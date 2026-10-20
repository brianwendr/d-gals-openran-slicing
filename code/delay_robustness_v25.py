from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from rebuilt_study import Config, TraceFactory, GraphResidualNet
from fast_eval import FastSimulator
from wireless_realism_v24 import make_uma_eta

MEAN_E2_SLOTS = 8.0
SEEDS = list(range(501, 509))
SLOTS = 300


def load_gat() -> GraphResidualNet:
    g = GraphResidualNet()
    g.load_state_dict(torch.load(ROOT / 'models' / 'gat_residual.pt', map_location='cpu'))
    g.eval()
    return g


def sample_e2_family(rng: np.random.Generator, family: str, shape, mean_slots: float = MEAN_E2_SLOTS):
    if family == 'Deterministic':
        return np.full(shape, int(round(mean_slots)), dtype=int)
    if family == 'Poisson':
        return rng.poisson(mean_slots, size=shape).astype(int)
    if family == 'Geometric':
        p = 1.0 / (mean_slots + 1.0)
        return (rng.geometric(p, size=shape) - 1).astype(int)
    if family == 'NegBin-heavy':
        # Negative-binomial stress with the same theoretical mean but higher variance than geometric.
        n = 0.5
        p = n / (n + mean_slots)
        return rng.negative_binomial(n, p, size=shape).astype(int)
    raise ValueError(family)


def paired_summary(df: pd.DataFrame, group_cols):
    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        r = {k: v for k, v in zip(group_cols, keys)}
        for method in ['Local-Lyapunov', 'D-GALS']:
            z = g[g.method == method]
            vals = z.sla_violation_pct.to_numpy(float)
            r[f'{method}_mean'] = vals.mean()
            r[f'{method}_sd'] = vals.std(ddof=1)
            r[f'{method}_ci95'] = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))
        piv = g.pivot(index='seed', columns='method', values='sla_violation_pct')
        diff = piv['Local-Lyapunov'] - piv['D-GALS']
        r['paired_gain_pp_mean'] = diff.mean()
        r['paired_gain_pp_sd'] = diff.std(ddof=1)
        r['relative_reduction_pct'] = 100.0 * diff.mean() / max(piv['Local-Lyapunov'].mean(), 1e-12)
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    data = ROOT / 'data'
    figs = ROOT / 'figures'
    data.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    gat = load_gat()

    # A. Matched-mean E2 delay-family sensitivity.
    fam_rows = []
    for seed in SEEDS:
        cfg = Config(cells=4, slots=SLOTS, seed=seed,
                     mean_e2_delay_slots=MEAN_E2_SLOTS,
                     mean_shared_delay_slots=1.0, traffic_corr=.75)
        base = TraceFactory(cfg).make()
        base['eta'] = make_uma_eta(seed + 31000, 4, SLOTS)
        for family in ['Deterministic', 'Poisson', 'Geometric', 'NegBin-heavy']:
            tr = {k: np.array(v, copy=True) for k, v in base.items()}
            rng = np.random.default_rng(seed * 1009 + len(family) * 37)
            tr['e2_delay'] = sample_e2_family(rng, family, tr['e2_delay'].shape)
            realized_mean = float(tr['e2_delay'].mean() * cfg.slot_ms)
            realized_sd = float(tr['e2_delay'].std(ddof=1) * cfg.slot_ms)
            for method in ['Local-Lyapunov', 'D-GALS']:
                m, _ = FastSimulator(cfg, tr, gat=gat).run(method)
                fam_rows.append({
                    'family': family,
                    'seed': seed,
                    'method': method,
                    'realized_e2_mean_ms': realized_mean,
                    'realized_e2_sd_ms': realized_sd,
                    'sla_violation_pct': 100.0 * m.sla_violation_rate,
                    'throughput_mbps': m.throughput_mbps,
                    'p95_delay_ms': m.p95_delay_ms,
                })
    fam = pd.DataFrame(fam_rows)
    fam.to_csv(data / 'delay_family_sensitivity_seed.csv', index=False)
    fam_sum = paired_summary(fam, ['family'])
    aux = fam.groupby('family').agg(
        realized_e2_mean_ms=('realized_e2_mean_ms', 'mean'),
        realized_e2_sd_ms=('realized_e2_sd_ms', 'mean')
    ).reset_index()
    fam_sum = fam_sum.merge(aux, on='family')
    order = ['Deterministic', 'Poisson', 'Geometric', 'NegBin-heavy']
    fam_sum['order'] = fam_sum['family'].map({v:i for i,v in enumerate(order)})
    fam_sum = fam_sum.sort_values('order').drop(columns='order')
    fam_sum.to_csv(data / 'delay_family_sensitivity_summary.csv', index=False)

    plt.figure(figsize=(7.2, 4.6))
    x = np.arange(len(order))
    for method in ['Local-Lyapunov', 'D-GALS']:
        vals=[]; errs=[]
        for family in order:
            z=fam[(fam.family==family)&(fam.method==method)].sla_violation_pct.to_numpy(float)
            vals.append(z.mean()); errs.append(1.96*z.std(ddof=1)/np.sqrt(len(z)))
        plt.errorbar(x, vals, yerr=errs, marker='o', capsize=3, label=method)
    plt.xticks(x, ['Deterministic','Poisson','Geometric','NegBin\nheavy-tail'])
    plt.xlabel('E2 age distribution (matched theoretical mean: 80 ms)')
    plt.ylabel('SLA violation rate (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs / 'supp_figure_s1_delay_family_sensitivity.png', dpi=320)
    plt.close()

    # B. Shared-context delay sensitivity, keeping E2 trace/workload/channel fixed within seed.
    sc_rows = []
    for seed in SEEDS:
        cfg = Config(cells=4, slots=SLOTS, seed=seed,
                     mean_e2_delay_slots=MEAN_E2_SLOTS,
                     mean_shared_delay_slots=1.0, traffic_corr=.75)
        base = TraceFactory(cfg).make()
        base['eta'] = make_uma_eta(seed + 41000, 4, SLOTS)
        for shared_mean in [0.0, 1.0, 2.0, 4.0, 8.0]:
            tr = {k: np.array(v, copy=True) for k, v in base.items()}
            rng = np.random.default_rng(seed * 2027 + int(shared_mean * 101))
            if shared_mean == 0:
                tr['shared_delay'] = np.zeros_like(tr['shared_delay'], dtype=int)
            else:
                p = 1.0 / (shared_mean + 1.0)
                tr['shared_delay'] = (rng.geometric(p, size=tr['shared_delay'].shape) - 1).astype(int)
            realized = float(tr['shared_delay'].mean() * cfg.slot_ms)
            for method in ['Local-Lyapunov', 'D-GALS']:
                m, _ = FastSimulator(cfg, tr, gat=gat).run(method)
                sc_rows.append({
                    'shared_delay_target_ms': shared_mean * cfg.slot_ms,
                    'shared_delay_realized_ms': realized,
                    'seed': seed,
                    'method': method,
                    'sla_violation_pct': 100.0 * m.sla_violation_rate,
                    'throughput_mbps': m.throughput_mbps,
                    'p95_delay_ms': m.p95_delay_ms,
                })
    sc = pd.DataFrame(sc_rows)
    sc.to_csv(data / 'shared_context_delay_sensitivity_seed.csv', index=False)
    sc_sum = paired_summary(sc, ['shared_delay_target_ms'])
    aux = sc.groupby('shared_delay_target_ms').agg(
        shared_delay_realized_ms=('shared_delay_realized_ms', 'mean')
    ).reset_index()
    sc_sum = sc_sum.merge(aux, on='shared_delay_target_ms')
    sc_sum.to_csv(data / 'shared_context_delay_sensitivity_summary.csv', index=False)

    plt.figure(figsize=(7.2, 4.6))
    for method in ['Local-Lyapunov', 'D-GALS']:
        z = sc[sc.method == method].groupby('shared_delay_target_ms').sla_violation_pct.agg(['mean','std','count']).reset_index()
        err = 1.96 * z['std'] / np.sqrt(z['count'])
        plt.errorbar(z.shared_delay_target_ms, z['mean'], yerr=err, marker='o', capsize=3, label=method)
    plt.xlabel('Mean shared-context age target (ms)')
    plt.ylabel('SLA violation rate (%)')
    plt.xticks([0,10,20,40,80])
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs / 'supp_figure_s2_shared_context_delay_sensitivity.png', dpi=320)
    plt.close()

    print('\nMatched-mean E2 delay families\n', fam_sum.to_string(index=False))
    print('\nShared-context delay sensitivity\n', sc_sum.to_string(index=False))

if __name__ == '__main__':
    main()
