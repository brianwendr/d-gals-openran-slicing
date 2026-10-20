from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from rebuilt_study import Config, TraceFactory, GraphResidualNet, FLOORS, BASE_ARR, CRIT, SLA_MS, sigmoid, ring_neighbors

FC_GHZ = 3.5
BW_HZ = 30e6
TX_DBM = 46.0
NF_DB = 7.0
H_BS = 25.0
H_UT = 1.5
NOISE_DBM = -174.0 + 10*np.log10(BW_HZ) + NF_DB
K = 3


def dbm_to_mw(x): return 10 ** (x / 10.0)

def los_probability(d):
    d=np.maximum(d,10.0)
    return np.minimum(18.0/d,1.0)*(1.0-np.exp(-d/63.0))+np.exp(-d/63.0)

def uma_pathloss_db(d2d,los):
    d3d=np.sqrt(np.maximum(d2d,1.0)**2+(H_BS-H_UT)**2)
    pl_los=28.0+22.0*np.log10(d3d)+20.0*np.log10(FC_GHZ)
    pl_nlos=13.54+39.08*np.log10(d3d)+20.0*np.log10(FC_GHZ)-0.6*(H_UT-1.5)
    pl_nlos=np.maximum(pl_los,pl_nlos)
    return np.where(los,pl_los,pl_nlos)

def bs_positions(n, isd=500.0):
    # Keep nearest-neighbor inter-site distance approximately constant as topology size changes.
    radius = isd / (2.0 * np.sin(np.pi / n))
    a=np.linspace(0,2*np.pi,n,endpoint=False)
    return np.column_stack([radius*np.cos(a),radius*np.sin(a)])

def make_uma_eta(seed,cells,slots):
    rng=np.random.default_rng(seed); bs=bs_positions(cells)
    # slow bounded mobility for one aggregate endpoint per cell-slice
    ue=np.zeros((cells,K,2)); vel=np.zeros_like(ue)
    for b in range(cells):
        ang=rng.uniform(0,2*np.pi,K); rad=rng.uniform(40,180,K)
        ue[b,:,0]=bs[b,0]+rad*np.cos(ang); ue[b,:,1]=bs[b,1]+rad*np.sin(ang)
    eta=np.zeros((slots,cells,K)); noise=dbm_to_mw(NOISE_DBM); tx=dbm_to_mw(TX_DBM)
    for t in range(slots):
        vel=0.85*vel+rng.normal(0,2.0,vel.shape); ue=ue+vel
        # project endpoints back into 220 m serving-cell radius
        for b in range(cells):
            dv=ue[b]-bs[b]; rr=np.linalg.norm(dv,axis=1)
            mask=rr>220
            if np.any(mask): ue[b,mask]=bs[b]+dv[mask]/rr[mask,None]*220
        for b in range(cells):
            for k in range(K):
                d=np.linalg.norm(bs-ue[b,k],axis=1)
                los=rng.random(cells)<los_probability(d)
                pl=uma_pathloss_db(d,los)
                shadow=rng.normal(0,np.where(los,4.0,6.0))
                fading=rng.exponential(1.0,cells)
                rx=tx*10**(-(pl+shadow)/10)*fading
                sig=rx[b]; interf=max(rx.sum()-sig,0.0)
                sinr=sig/max(interf+noise,1e-18)
                eta[t,b,k]=np.clip(np.log2(1+sinr),0.05,6.0)
    return eta

def load_coupled_delay(rng,arrivals,base_mean_slots):
    s,c,k=arrivals.shape
    p=1/(base_mean_slots+1) if base_mean_slots>0 else 1.0
    base=(rng.geometric(p,size=(s,c,k))-1) if base_mean_slots>0 else np.zeros((s,c,k),int)
    load=arrivals.sum(2); q50=np.quantile(load,.5); q85=np.quantile(load,.85)
    extra=np.zeros((s,c),int)
    mod=(load>=q50)&(load<q85); high=load>=q85
    extra[mod]=rng.binomial(1,.45,mod.sum())
    extra[high]=1+rng.binomial(2,.55,high.sum())
    return base+extra[:,:,None]

def normalize_with_floors(score):
    score=np.maximum(score,1e-9); w=score/score.sum(axis=1,keepdims=True)
    x=FLOORS[None,:]+(1-FLOORS.sum())*w
    return x/x.sum(axis=1,keepdims=True)

class FastReplay:
    def __init__(self,cfg,trace,gat):
        self.cfg=cfg; self.trace=trace; self.gat=gat; self.neigh=ring_neighbors(cfg.cells)
    def obs_index(self,t,d): return np.maximum(0,t-d)
    def run(self,method):
        c,s=self.cfg.cells,self.cfg.slots
        q=np.zeros((c,K)); qhist=np.zeros((s+1,c,K)); pihist=np.zeros((s+1,c,K))
        delays=[]; vios=[]; served_total=0.0; energy=0.0; qmeans=[]; dt=[]
        for t in range(s):
            qhist[t]=q
            d=self.trace['e2_delay'][t].astype(int)
            j=self.obs_index(t,d)
            # gather delayed q/arrival/eta
            bi=np.arange(c)[:,None]; ki=np.arange(K)[None,:]
            qobs=qhist[j,bi,ki]
            aobs=self.trace['arrival'][j,bi,ki]
            etaobs=self.trace['eta'][j,bi,ki]
            pihist[t]=sigmoid((qobs/(BASE_ARR[None,:]+0.08)-1.0)*0.8)
            qhat=qobs.copy()
            if method=='D-GALS':
                locals_=[]; nbs=[]
                for b in range(c):
                    for k in range(K):
                        dd=d[b,k]
                        local=[qobs[b,k]/2.0,aobs[b,k]/0.5,etaobs[b,k]/3.0,min(dd,12)/6.0,CRIT[k]/3.0]
                        nbrows=[]
                        for nb in self.neigh[b]:
                            sd=int(self.trace['shared_delay'][t,nb,k]); jj=max(0,t-sd)
                            nbrows.append([pihist[jj,nb,k],min(sd,12)/6.0])
                        locals_.append(local); nbs.append(nbrows if nbrows else [[0,0]])
                L=torch.tensor(np.asarray(locals_),dtype=torch.float32)
                N=torch.tensor(np.asarray(nbs),dtype=torch.float32)
                with torch.no_grad(): pred,_=self.gat(L,N)
                pred=pred.numpy().reshape(c,K)*2.0
                qhat=np.maximum(0,qobs+np.where(d>0,pred,0.0))
            qstate=qhat if method=='D-GALS' else qobs
            score=aobs+1.15*CRIT[None,:]*(qstate/(0.12+BASE_ARR[None,:]))
            import time
            st=time.perf_counter(); alloc=normalize_with_floors(score); dt.append((time.perf_counter()-st)*1000/max(c,1))
            qpre=q+self.trace['arrival'][t]
            eta_true=self.trace['eta'][t]
            cap=self.cfg.capacity_scale*(0.70+0.30*eta_true/np.maximum(eta_true.mean(axis=1,keepdims=True),1e-6))
            service=np.minimum(qpre,alloc*cap); q=np.maximum(qpre-service,0); qhist[t+1]=q
            dly=q/np.maximum(BASE_ARR[None,:],0.03)*self.cfg.slot_ms
            vio=dly>SLA_MS[None,:]
            delays.extend(dly.ravel()); vios.extend(vio.ravel()); served_total+=service.sum(); qmeans.append(q.mean())
            energy+=float(np.sum(alloc*(0.9+0.06*eta_true)))
        return {
            'throughput_mbps':served_total/s*1000/self.cfg.slot_ms,
            'sla_violation_pct':100*float(np.mean(vios)),
            'reliability_pct':100*(1-float(np.mean(vios))),
            'p95_delay_ms':float(np.percentile(delays,95)),
            'decision_ms':float(np.mean(dt)),
        }

def load_gat(root):
    g=GraphResidualNet(); g.load_state_dict(torch.load(root/'models'/'gat_residual.pt',map_location='cpu')); g.eval(); return g

def summarize(df,cols):
    rows=[]
    for keys,g in df.groupby(cols):
        if not isinstance(keys,tuple): keys=(keys,)
        r={c:v for c,v in zip(cols,keys)}
        for v in ['throughput_mbps','sla_violation_pct','reliability_pct','p95_delay_ms','decision_ms']:
            x=g[v].to_numpy(float); r[v+'_mean']=x.mean(); r[v+'_sd']=x.std(ddof=1)
        rows.append(r)
    return pd.DataFrame(rows)

def main():
    root=Path(__file__).resolve().parents[1]; data=root/'data'; figs=root/'figures'; gat=load_gat(root)
    seeds=list(range(301,309))

    # Standard four-cell UMa controller replay at 20 and 80 ms mean E2 delay.
    rows=[]
    for delay_ms in [20,80]:
        for seed in seeds:
            cfg=Config(cells=4,slots=300,seed=seed,mean_e2_delay_slots=delay_ms/10,mean_shared_delay_slots=1.0,traffic_corr=.75)
            trace=TraceFactory(cfg).make(); trace['eta']=make_uma_eta(seed+7000,4,cfg.slots)
            for method in ['Local-Lyapunov','D-GALS']:
                r=FastReplay(cfg,trace,gat).run(method); rows.append({'cells':4,'e2_delay_ms':delay_ms,'seed':seed,'method':method,**r})
    uma=pd.DataFrame(rows); uma.to_csv(data/'uma_controller_replay_seed.csv',index=False)
    us=summarize(uma,['cells','e2_delay_ms','method']); us.to_csv(data/'uma_controller_replay_summary.csv',index=False)
    plt.figure(figsize=(7,4.5))
    for method in ['Local-Lyapunov','D-GALS']:
        z=us[us.method==method].sort_values('e2_delay_ms'); plt.errorbar(z.e2_delay_ms,z.sla_violation_pct_mean,yerr=1.96*z.sla_violation_pct_sd/np.sqrt(len(seeds)),marker='o',capsize=3,label=method)
    plt.xlabel('Mean E2 reporting delay (ms)'); plt.ylabel('SLA violation rate (%)'); plt.xticks([20,80]); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(figs/'figure6_uma_replay_v24.png',dpi=320); plt.close()

    rows=[]
    # topology scale at 80ms E2 stress; nearest-neighbor inter-site distance is held at 500 m.
    for cells in [4,8,16]:
        for seed in seeds:
            cfg=Config(cells=cells,slots=300,seed=seed,mean_e2_delay_slots=8.0,mean_shared_delay_slots=1.0,traffic_corr=.75)
            trace=TraceFactory(cfg).make(); trace['eta']=make_uma_eta(seed+9000,cells,cfg.slots)
            for method in ['Local-Lyapunov','D-GALS']:
                r=FastReplay(cfg,trace,gat).run(method); rows.append({'cells':cells,'e2_delay_ms':80,'seed':seed,'method':method,**r})
    topo=pd.DataFrame(rows); topo.to_csv(data/'uma_topology_replay_seed.csv',index=False); ts=summarize(topo,['cells','e2_delay_ms','method']); ts.to_csv(data/'uma_topology_replay_summary.csv',index=False)

    rows=[]
    # load-coupled delay under 4-cell UMa replay
    for base_ms in [20,80]:
        for seed in seeds:
            cfg=Config(cells=4,slots=300,seed=seed,mean_e2_delay_slots=base_ms/10,mean_shared_delay_slots=1.0,traffic_corr=.75)
            trace=TraceFactory(cfg).make(); trace['eta']=make_uma_eta(seed+12000,4,cfg.slots)
            rng=np.random.default_rng(seed+22000+base_ms); trace['e2_delay']=load_coupled_delay(rng,trace['arrival'],base_ms/10)
            realized=float(trace['e2_delay'].mean()*cfg.slot_ms)
            corr=float(np.corrcoef(trace['arrival'].sum(2).ravel(),trace['e2_delay'].mean(2).ravel())[0,1])
            for method in ['Local-Lyapunov','D-GALS']:
                r=FastReplay(cfg,trace,gat).run(method); rows.append({'base_e2_delay_ms':base_ms,'realized_e2_delay_ms':realized,'arrival_delay_corr':corr,'seed':seed,'method':method,**r})
    lc=pd.DataFrame(rows); lc.to_csv(data/'load_coupled_e2_seed.csv',index=False); ls=summarize(lc,['base_e2_delay_ms','method'])
    aux=lc.groupby(['base_e2_delay_ms','method']).agg(realized_e2_delay_ms_mean=('realized_e2_delay_ms','mean'),arrival_delay_corr_mean=('arrival_delay_corr','mean')).reset_index(); ls=ls.merge(aux,on=['base_e2_delay_ms','method']); ls.to_csv(data/'load_coupled_e2_summary.csv',index=False)

    # standalone plots (no subplots)
    g=ts[ts.e2_delay_ms==80]
    plt.figure(figsize=(7,4.5))
    for method in ['Local-Lyapunov','D-GALS']:
        z=g[g.method==method].sort_values('cells'); plt.errorbar(z.cells,z.sla_violation_pct_mean,yerr=1.96*z.sla_violation_pct_sd/np.sqrt(len(seeds)),marker='o',capsize=3,label=method)
    plt.xlabel('Number of cells'); plt.ylabel('SLA violation rate (%)'); plt.xticks([4,8,16]); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(figs/'figure7_uma_topology_scaling.png',dpi=320); plt.close()
    plt.figure(figsize=(7,4.5))
    for method in ['Local-Lyapunov','D-GALS']:
        z=ls[ls.method==method].sort_values('realized_e2_delay_ms_mean'); plt.errorbar(z.realized_e2_delay_ms_mean,z.sla_violation_pct_mean,yerr=1.96*z.sla_violation_pct_sd/np.sqrt(len(seeds)),marker='o',capsize=3,label=method)
    plt.xlabel('Realized mean E2 reporting delay (ms)'); plt.ylabel('SLA violation rate (%)'); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(figs/'figure8_load_coupled_e2.png',dpi=320); plt.close()
    print('\nUMA 4-CELL\n',us.to_string(index=False)); print('\nTOPOLOGY\n',ts.to_string(index=False)); print('\nLOAD-COUPLED\n',ls.to_string(index=False))

if __name__=='__main__': main()
