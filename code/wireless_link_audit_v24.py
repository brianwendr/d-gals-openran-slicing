from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

FC_GHZ = 3.5
BW_HZ = 30e6
TX_DBM = 46.0
NF_DB = 7.0
H_BS = 25.0
H_UT = 1.5
NOISE_DBM = -174.0 + 10*np.log10(BW_HZ) + NF_DB


def dbm_to_mw(x):
    return 10 ** (x / 10.0)


def los_probability(d):
    d = np.maximum(d, 10.0)
    return np.minimum(18.0/d, 1.0) * (1.0 - np.exp(-d/63.0)) + np.exp(-d/63.0)


def uma_pathloss_db(d2d, los):
    d3d = np.sqrt(np.maximum(d2d, 1.0)**2 + (H_BS-H_UT)**2)
    pl_los = 28.0 + 22.0*np.log10(d3d) + 20.0*np.log10(FC_GHZ)
    pl_nlos = 13.54 + 39.08*np.log10(d3d) + 20.0*np.log10(FC_GHZ) - 0.6*(H_UT-1.5)
    return np.where(los, pl_los, np.maximum(pl_los, pl_nlos))


def bs_positions(n, isd=500.0):
    # Preserve nearest-neighbor inter-site distance under topology scaling.
    radius = isd / (2.0 * np.sin(np.pi / n))
    a = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([radius*np.cos(a), radius*np.sin(a)])


def audit_topology(cells: int, seed: int, samples: int = 5000):
    rng = np.random.default_rng(seed)
    bs = bs_positions(cells)
    serving_distance=[]; sinr_db=[]; eta=[]; serving_los=[]
    noise=dbm_to_mw(NOISE_DBM); tx=dbm_to_mw(TX_DBM)
    for _ in range(samples):
        b=int(rng.integers(0,cells))
        ang=rng.uniform(0,2*np.pi); rad=rng.uniform(40,220)
        ue=bs[b] + rad*np.array([np.cos(ang),np.sin(ang)])
        d=np.linalg.norm(bs-ue,axis=1)
        los=rng.random(cells)<los_probability(d)
        pl=uma_pathloss_db(d,los)
        shadow=rng.normal(0,np.where(los,4.0,6.0))
        fading=rng.exponential(1.0,cells)
        rx=tx*10**(-(pl+shadow)/10)*fading
        sig=rx[b]; interf=max(rx.sum()-sig,0.0)
        s=sig/max(interf+noise,1e-18)
        serving_distance.append(rad)
        sinr_db.append(10*np.log10(max(s,1e-12)))
        eta.append(np.clip(np.log2(1+s),0.05,6.0))
        serving_los.append(float(los[b]))
    serving_distance=np.asarray(serving_distance); sinr_db=np.asarray(sinr_db); eta=np.asarray(eta); serving_los=np.asarray(serving_los)
    return {
        'cells': cells,
        'samples': samples,
        'serving_distance_median_m': float(np.median(serving_distance)),
        'serving_distance_p95_m': float(np.percentile(serving_distance,95)),
        'serving_los_fraction': float(np.mean(serving_los)),
        'sinr_db_p05': float(np.percentile(sinr_db,5)),
        'sinr_db_median': float(np.median(sinr_db)),
        'sinr_db_p95': float(np.percentile(sinr_db,95)),
        'sinr_below_0db_fraction': float(np.mean(sinr_db<0)),
        'spectral_efficiency_median_bpshz': float(np.median(eta)),
        'spectral_efficiency_p05_bpshz': float(np.percentile(eta,5)),
    }


def main():
    root=Path(__file__).resolve().parents[1]
    rows=[]
    for cells in [4,8,16]:
        for seed in range(501,509):
            rows.append(audit_topology(cells,seed,5000))
    seed_df=pd.DataFrame(rows)
    seed_df.to_csv(root/'data'/'uma_link_budget_audit_seed.csv',index=False)
    out=[]
    metrics=[c for c in seed_df.columns if c not in ['cells','samples']]
    for cells,g in seed_df.groupby('cells'):
        r={'cells':cells,'samples_per_seed':int(g['samples'].iloc[0]),'seeds':len(g)}
        for m in metrics:
            r[m+'_mean']=float(g[m].mean())
            r[m+'_sd']=float(g[m].std(ddof=1))
        out.append(r)
    pd.DataFrame(out).to_csv(root/'data'/'uma_link_budget_audit_summary.csv',index=False)
    print(pd.DataFrame(out).to_string(index=False))

if __name__=='__main__':
    main()
