from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'


def holm_adjust(pvals):
    pvals=np.asarray(pvals,float)
    order=np.argsort(pvals)
    adj=np.empty_like(pvals)
    running=0.0
    m=len(pvals)
    for rank,idx in enumerate(order):
        val=(m-rank)*pvals[idx]
        running=max(running,val)
        adj[idx]=min(1.0,running)
    return adj


def bootstrap_ci(x, B=20000, seed=20261020):
    x=np.asarray(x,float)
    rng=np.random.default_rng(seed)
    n=len(x)
    means=np.empty(B)
    for b in range(B):
        means[b]=rng.choice(x,size=n,replace=True).mean()
    return np.quantile(means,[.025,.975])


def primary_audit():
    df=pd.read_csv(DATA/'seed_level_main.csv')
    rows=[]
    raw=[]
    for delay in [20,40,80]:
        p=df[(df.delay_ms==delay)&(df.method.isin(['Local-Lyapunov','D-GALS']))]
        piv=p.pivot(index='seed',columns='method',values='sla_violation_rate')*100
        diff=piv['Local-Lyapunov']-piv['D-GALS']
        w=wilcoxon(diff,alternative='two-sided',method='exact')
        lo,hi=bootstrap_ci(diff,seed=20261020+delay)
        rows.append({'scenario':'primary','condition':f'E2 {delay} ms','n_pairs':len(diff),'paired_gain_pp_mean':diff.mean(),'paired_gain_pp_sd':diff.std(ddof=1),'paired_dz':diff.mean()/diff.std(ddof=1),'bootstrap95_lo_pp':lo,'bootstrap95_hi_pp':hi,'wilcoxon_two_sided_p':w.pvalue})
        raw.append(w.pvalue)
    adj=holm_adjust(raw)
    for r,a in zip(rows,adj): r['holm_adjusted_p']=a
    return rows


def robustness_audit():
    rows=[]
    for fn,scenario,group_col in [
        ('delay_family_sensitivity_seed.csv','delay-family','family'),
        ('shared_context_delay_sensitivity_seed.csv','shared-context','shared_delay_target_ms')]:
        df=pd.read_csv(DATA/fn)
        groups=list(df[group_col].drop_duplicates())
        raw=[]; local=[]
        for i,gval in enumerate(groups):
            g=df[df[group_col]==gval]
            piv=g.pivot(index='seed',columns='method',values='sla_violation_pct')
            diff=piv['Local-Lyapunov']-piv['D-GALS']
            w=wilcoxon(diff,alternative='two-sided',method='exact')
            lo,hi=bootstrap_ci(diff,seed=20262000+i)
            local.append({'scenario':scenario,'condition':str(gval),'n_pairs':len(diff),'paired_gain_pp_mean':diff.mean(),'paired_gain_pp_sd':diff.std(ddof=1),'paired_dz':diff.mean()/diff.std(ddof=1) if diff.std(ddof=1)>0 else np.inf,'bootstrap95_lo_pp':lo,'bootstrap95_hi_pp':hi,'wilcoxon_two_sided_p':w.pvalue})
            raw.append(w.pvalue)
        adj=holm_adjust(raw)
        for r,a in zip(local,adj): r['holm_adjusted_p']=a
        rows.extend(local)
    return rows

rows=primary_audit()+robustness_audit()
out=pd.DataFrame(rows)
out.to_csv(DATA/'paired_statistical_audit_v25.csv',index=False)
print(out.to_string(index=False))
