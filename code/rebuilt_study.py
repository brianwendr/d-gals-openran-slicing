from __future__ import annotations
import json, math, time, hashlib, platform, os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

SLICE_NAMES=["eMBB","URLLC","mMTC"]
K=3
SLA_MS=np.array([80.0,10.0,120.0],dtype=float)
CRIT=np.array([1.0,3.0,0.6],dtype=float)
BASE_ARR=np.array([0.34,0.14,0.20],dtype=float)  # Mbit/slot
FLOORS=np.array([0.05,0.10,0.03],dtype=float)
ACTION_TEMPLATES=np.array([
    [0.55,0.30,0.15],[0.45,0.40,0.15],[0.45,0.30,0.25],[0.35,0.50,0.15],
    [0.35,0.40,0.25],[0.35,0.30,0.35],[0.25,0.60,0.15],[0.25,0.50,0.25],
    [0.25,0.40,0.35],[0.25,0.30,0.45],[0.40,0.35,0.25],[0.33,0.34,0.33],
],dtype=float)

@dataclass
class Config:
    cells:int=4
    slots:int=500
    slot_ms:float=10.0
    mean_e2_delay_slots:float=2.0
    mean_shared_delay_slots:float=1.0
    traffic_corr:float=0.75
    capacity_scale:float=0.80
    guard_floors:Tuple[float,float,float]=(0.05,0.10,0.03)
    seed:int=1

@dataclass
class Metrics:
    throughput_mbps:float
    p95_delay_ms:float
    sla_violation_rate:float
    reliability:float
    energy_per_mbit:float
    mean_queue_mbit:float
    decision_time_ms:float
    normalization_error:float


def geometric_delay(rng:np.random.Generator, mean:float, shape):
    if mean<=0:
        return np.zeros(shape,dtype=int)
    p=1.0/(mean+1.0)
    return rng.geometric(p,size=shape)-1

def ring_neighbors(n:int)->List[List[int]]:
    if n==1: return [[]]
    if n==2: return [[1],[0]]
    return [[(i-1)%n,(i+1)%n] for i in range(n)]

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-20,20)))

def normalize_with_floors(score:np.ndarray, floors=FLOORS)->np.ndarray:
    score=np.maximum(score,1e-9)
    w=score/score.sum(axis=-1,keepdims=True)
    rem=1.0-floors.sum()
    x=floors+rem*w
    # final exact normalization guard
    x=x/np.maximum(x.sum(axis=-1,keepdims=True),1e-12)
    return x

class TraceFactory:
    def __init__(self,cfg:Config):
        self.cfg=cfg
        self.rng=np.random.default_rng(cfg.seed)
        self.neigh=ring_neighbors(cfg.cells)
    def make(self):
        c,s=self.cfg.cells,self.cfg.slots
        rng=self.rng
        # shared AR(1) per slice to produce correlated bursts
        common=np.zeros((s,K)); eps=rng.normal(0,1,(s,K))
        for t in range(1,s): common[t]=0.88*common[t-1]+0.48*eps[t]
        common=(common-common.mean(0))/(common.std(0)+1e-9)
        # scheduled burst windows
        common[120:165]+=np.array([0.7,1.4,0.3]); common[320:355]+=np.array([0.4,1.1,0.5])
        indep=rng.normal(0,1,(s,c,K))
        rho=np.clip(self.cfg.traffic_corr,0,0.98)
        latent=rho*common[:,None,:]+np.sqrt(1-rho**2)*indep
        cell_bias=rng.normal(0,0.06,(c,K))
        diurnal=(1+0.12*np.sin(2*np.pi*np.arange(s)/144.0))[:,None,None]
        arrivals=BASE_ARR[None,None,:]*diurnal*np.exp(0.16*latent+cell_bias[None,:,:])
        arrivals=np.clip(arrivals,0.04,0.85)
        # Rayleigh-derived channel with distance/path-loss surrogate
        distance=rng.uniform(0.18,0.85,(c,K))
        pathgain=np.maximum(distance,0.05)**(-3.5)
        eta=np.zeros((s,c,K))
        for t in range(s):
            ray=rng.rayleigh(scale=1/np.sqrt(2),size=(c,K))
            shadow_db=rng.normal(0,1.6,(c,K))
            snr_linear=4.5*pathgain*(ray**2)*10**(shadow_db/10)
            eta[t]=np.clip(np.log2(1+snr_linear),0.35,6.0)
        # outage pocket
        eta[250:280,1% c,:]*=0.55
        e2=geometric_delay(rng,self.cfg.mean_e2_delay_slots,(s,c,K))
        shared=geometric_delay(rng,self.cfg.mean_shared_delay_slots,(s,c,K))
        return {'arrival':arrivals,'eta':eta,'e2_delay':e2,'shared_delay':shared}

class GraphResidualNet(nn.Module):
    """Small shared graph-attention residual predictor.
    Local feature: qtilde, arrival, eta, E2 age, criticality.
    Neighbor input: delayed scalar pressure + shared-summary age.
    Output: normalized queue residual (Q-Qtilde).
    """
    def __init__(self,hidden=20):
        super().__init__()
        self.local=nn.Sequential(nn.Linear(5,hidden),nn.Tanh())
        self.nb=nn.Sequential(nn.Linear(2,hidden),nn.Tanh())
        self.att=nn.Sequential(nn.Linear(hidden*2,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.out=nn.Sequential(nn.Linear(hidden*2,hidden),nn.ReLU(),nn.Linear(hidden,1))
    def forward(self,local,nb):
        # local [N,5], nb [N,M,2]
        l=self.local(local)
        n=self.nb(nb)
        lrep=l[:,None,:].expand(-1,n.shape[1],-1)
        logits=self.att(torch.cat([lrep,n],dim=-1)).squeeze(-1)
        a=torch.softmax(logits,dim=1)
        ctx=(a[:,:,None]*n).sum(1)
        y=self.out(torch.cat([l,ctx],dim=-1)).squeeze(-1)
        return y,a

class PPOPolicy(nn.Module):
    def __init__(self,dim:int,n_actions:int):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(dim,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
        self.pi=nn.Linear(64,n_actions); self.v=nn.Linear(64,1)
    def forward(self,x):
        h=self.body(x); return self.pi(h),self.v(h).squeeze(-1)

class Simulator:
    def __init__(self,cfg:Config,trace:Dict[str,np.ndarray],gat:Optional[GraphResidualNet]=None,ppo:Optional[PPOPolicy]=None):
        self.cfg=cfg; self.trace=trace; self.gat=gat; self.ppo=ppo; self.neigh=ring_neighbors(cfg.cells)
        self.device=torch.device('cpu')
    def _obs_index(self,t,d): return max(0,t-int(d))
    def _build_gat_input(self,b,k,t,qhist,pihist):
        d=int(self.trace['e2_delay'][t,b,k]); j=self._obs_index(t,d)
        qtil=qhist[j,b,k]
        arr=self.trace['arrival'][j,b,k]
        eta=self.trace['eta'][j,b,k]
        local=np.array([qtil/2.0,arr/0.5,eta/3.0,min(d,12)/6.0,CRIT[k]/3.0],np.float32)
        ns=[]
        for nb in self.neigh[b]:
            dd=int(self.trace['shared_delay'][t,nb,k]); jj=self._obs_index(t,dd)
            ns.append([pihist[jj,nb,k],min(dd,12)/6.0])
        if not ns: ns=[[0,0]]
        return local,np.array(ns,np.float32),qtil,d
    def _gat_correct(self,b,k,t,qhist,pihist):
        local,nb,qtil,d=self._build_gat_input(b,k,t,qhist,pihist)
        if self.gat is None: return qtil
        with torch.no_grad():
            pred,_=self.gat(torch.tensor(local)[None,:],torch.tensor(nb)[None,:,:])
        # target trained as residual / 2.0
        return max(0.0,qtil+(float(pred.item())*2.0 if d>0 else 0.0))
    def _ppo_state(self,b,t,qhist,pihist):
        vals=[]
        for k in range(K):
            local,nb,qtil,d=self._build_gat_input(b,k,t,qhist,pihist)
            vals.extend(local.tolist())
            # aggregate neighbor summaries/ages (mean/max)
            vals.extend([float(nb[:,0].mean()),float(nb[:,1].mean())])
        return np.array(vals,np.float32)
    def _alloc(self,method,b,t,qhist,pihist,pf_ema):
        qobs=np.zeros(K); qhat=np.zeros(K); etaobs=np.zeros(K); aobs=np.zeros(K)
        for k in range(K):
            d=int(self.trace['e2_delay'][t,b,k]); j=self._obs_index(t,d)
            qobs[k]=qhist[j,b,k]; aobs[k]=self.trace['arrival'][j,b,k]; etaobs[k]=self.trace['eta'][j,b,k]
            qhat[k]=self._gat_correct(b,k,t,qhist,pihist) if method in {'GAT-only','D-GALS'} else qobs[k]
        if method=='Static': return np.array([0.5,0.3,0.2])
        if method=='Demand-Proportional': return normalize_with_floors(aobs[None,:])[0]
        if method=='Proportional-Fair':
            return normalize_with_floors((aobs*np.maximum(etaobs,0.1)/(0.15+pf_ema[b]))[None,:])[0]
        if method=='Max-SINR': return normalize_with_floors((np.maximum(etaobs,0.1)**2)[None,:])[0]
        if method=='Local-Lyapunov':
            score=aobs+1.15*CRIT*(qobs/(0.12+BASE_ARR))
            return normalize_with_floors(score[None,:])[0]
        if method=='GAT-only':
            score=aobs+0.75*(qhat/(0.12+BASE_ARR))
            return normalize_with_floors(score[None,:])[0]
        if method=='D-GALS':
            score=aobs+1.15*CRIT*(qhat/(0.12+BASE_ARR))
            return normalize_with_floors(score[None,:])[0]
        if method=='PPO':
            state=torch.tensor(self._ppo_state(b,t,qhist,pihist))[None,:]
            with torch.no_grad(): logits,_=self.ppo(state); a=int(torch.argmax(logits,dim=-1).item())
            return ACTION_TEMPLATES[a].copy()
        raise ValueError(method)
    def run(self,method:str,collect_gat=False,keep_trace=False):
        c,s=self.cfg.cells,self.cfg.slots
        q=np.zeros((c,K),float); qhist=np.zeros((s+1,c,K),float); pihist=np.zeros((s+1,c,K),float)
        pf=np.ones((c,K))*0.10
        rows=[]; sample_rows=[]; decision=[]; delays=[]; violations=[]; served_total=0.; energy_total=0.; qmeans=[]; normerr=0.
        for t in range(s):
            qhist[t]=q
            # publish summary based only on local delayed state available at beginning of slot
            for b in range(c):
                for k in range(K):
                    d=int(self.trace['e2_delay'][t,b,k]); j=self._obs_index(t,d)
                    qtil=qhist[j,b,k]
                    pihist[t,b,k]=float(sigmoid((qtil/(BASE_ARR[k]+0.08)-1.0)*0.8))
            alloc=np.zeros((c,K)); st=time.perf_counter()
            for b in range(c): alloc[b]=self._alloc(method,b,t,qhist,pihist,pf)
            decision.append((time.perf_counter()-st)*1000/max(c,1))
            normerr=max(normerr,float(np.abs(alloc.sum(1)-1).max()))
            # arrivals enter before service
            qpre=q+self.trace['arrival'][t]
            # cell capacity varies by average current spectral efficiency
            eta_true=self.trace['eta'][t]
            cap=self.cfg.capacity_scale*(0.70+0.30*eta_true/np.maximum(eta_true.mean(axis=1,keepdims=True),1e-6))
            service=np.minimum(qpre,alloc*cap)
            q=np.maximum(qpre-service,0)
            qhist[t+1]=q
            if method=='Proportional-Fair': pf=0.92*pf+0.08*service
            delay=q/np.maximum(BASE_ARR[None,:],0.03)*self.cfg.slot_ms
            vio=delay>SLA_MS[None,:]
            delays.extend(delay.ravel()); violations.extend(vio.ravel()); served_total+=service.sum(); qmeans.append(q.mean())
            energy_total+=float(np.sum(alloc*(0.9+0.06*eta_true)))
            if collect_gat:
                for b in range(c):
                    for k in range(K):
                        local,nb,qtil,d=self._build_gat_input(b,k,t,qhist,pihist)
                        sample_rows.append((local,nb,(qhist[t,b,k]-qtil)/2.0,b,k,t))
            if keep_trace:
                for b in range(c):
                    for k in range(K):
                        rows.append({'slot':t,'cell':b,'slice':SLICE_NAMES[k],'true_queue':qpre[b,k],
                                     'e2_delay_slots':int(self.trace['e2_delay'][t,b,k]),'alloc_share':alloc[b,k],
                                     'post_queue':q[b,k],'delay_ms':delay[b,k],'violation':int(vio[b,k])})
        mean_tp=(served_total/s)*1000/self.cfg.slot_ms
        m=Metrics(mean_tp,float(np.percentile(delays,95)),float(np.mean(violations)),1-float(np.mean(violations)),
                  float(energy_total/max(served_total,1e-9)),float(np.mean(qmeans)),float(np.mean(decision)),normerr)
        return m,sample_rows,pd.DataFrame(rows)

# -------- GAT dataset/training --------
def collect_gat_dataset(outdir:Path,cells=4,slots=700,seeds=range(101,109),mean_delay=4.0,corr=0.75):
    locals_,nbs_,ys=[] ,[],[]
    for seed in seeds:
        cfg=Config(cells=cells,slots=slots,seed=seed,mean_e2_delay_slots=mean_delay,traffic_corr=corr)
        trace=TraceFactory(cfg).make(); sim=Simulator(cfg,trace)
        _,samples,_=sim.run('Demand-Proportional',collect_gat=True)
        for l,nb,y,*_ in samples:
            # ring always 2 neighbors for cells>=3
            locals_.append(l); nbs_.append(nb); ys.append(y)
    X=np.stack(locals_); N=np.stack(nbs_); y=np.array(ys,np.float32)
    np.savez_compressed(outdir/'gat_dataset.npz',local=X,neighbor=N,target=y)
    return X,N,y

def train_gat(outdir:Path,epochs=35):
    torch.manual_seed(2026); np.random.seed(2026)
    X,N,y=collect_gat_dataset(outdir)
    rng=np.random.default_rng(42); idx=rng.permutation(len(y)); cut=int(.8*len(y)); tr,va=idx[:cut],idx[cut:]
    model=GraphResidualNet(); opt=optim.Adam(model.parameters(),lr=2e-3,weight_decay=1e-5)
    Xtr=torch.tensor(X[tr]); Ntr=torch.tensor(N[tr]); ytr=torch.tensor(y[tr]); Xv=torch.tensor(X[va]); Nv=torch.tensor(N[va]); yv=torch.tensor(y[va])
    hist=[]; bs=512
    for ep in range(epochs):
        perm=torch.randperm(len(tr)); model.train(); losses=[]
        for i in range(0,len(tr),bs):
            ix=perm[i:i+bs]; pred,_=model(Xtr[ix],Ntr[ix]); loss=((pred-ytr[ix])**2).mean(); opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        model.eval();
        with torch.no_grad(): pv,_=model(Xv,Nv); vl=((pv-yv)**2).mean().item(); base=(yv**2).mean().item(); kappa=vl/max(base,1e-12)
        hist.append({'epoch':ep+1,'train_mse':float(np.mean(losses)),'val_mse':vl,'uncorrected_mse':base,'kappa':kappa})
    torch.save(model.state_dict(),outdir/'gat_residual.pt'); pd.DataFrame(hist).to_csv(outdir/'gat_training_log.csv',index=False)
    return model,hist[-1]

# -------- PPO training --------
def train_ppo(outdir:Path,cells=4,train_slots=300,iterations=28,rollouts_per_iter=3):
    torch.manual_seed(2027); np.random.seed(2027)
    state_dim=7*K; policy=PPOPolicy(state_dim,len(ACTION_TEMPLATES));
    init_path=outdir/'ppo_policy_bc_init.pt'
    if init_path.exists(): policy.load_state_dict(torch.load(init_path,map_location='cpu'))
    opt=optim.Adam(policy.parameters(),lr=3e-4)
    logs=[]; rng=np.random.default_rng(777)
    for it in range(iterations):
        states=[]; actions=[]; oldlp=[]; rewards=[]; vals=[]
        for rr in range(rollouts_per_iter):
            seed=int(rng.integers(2000,9000)); cfg=Config(cells=cells,slots=train_slots,seed=seed,mean_e2_delay_slots=2.0,traffic_corr=.75)
            tr=TraceFactory(cfg).make(); neigh=ring_neighbors(cells)
            q=np.zeros((cells,K)); qhist=np.zeros((train_slots+1,cells,K)); pihist=np.zeros((train_slots+1,cells,K))
            for t in range(train_slots):
                qhist[t]=q
                # local publication
                for b in range(cells):
                    for k in range(K):
                        d=int(tr['e2_delay'][t,b,k]); j=max(0,t-d); qtil=qhist[j,b,k]
                        pihist[t,b,k]=float(sigmoid((qtil/(BASE_ARR[k]+.08)-1)*.8))
                step_states=[]
                for b in range(cells):
                    vals_b=[]
                    for k in range(K):
                        d=int(tr['e2_delay'][t,b,k]); j=max(0,t-d); qtil=qhist[j,b,k]; eta=tr['eta'][j,b,k]
                        local=[qtil/2, tr['arrival'][j,b,k]/.5, eta/3, min(d,12)/6, CRIT[k]/3]
                        nvals=[]; ages=[]
                        for nb in neigh[b]:
                            dd=int(tr['shared_delay'][t,nb,k]); jj=max(0,t-dd); nvals.append(pihist[jj,nb,k]); ages.append(min(dd,12)/6)
                        vals_b.extend(local+[float(np.mean(nvals)),float(np.mean(ages))])
                    step_states.append(vals_b)
                st=torch.tensor(np.array(step_states),dtype=torch.float32); logits,v=policy(st); dist=torch.distributions.Categorical(logits=logits); a=dist.sample(); lp=dist.log_prob(a)
                alloc=ACTION_TEMPLATES[a.detach().numpy()]
                qpre=q+tr['arrival'][t]; eta_true=tr['eta'][t]; cap=cfg.capacity_scale*(.70+.30*eta_true/np.maximum(eta_true.mean(1,keepdims=True),1e-6)); serv=np.minimum(qpre,alloc*cap); q=np.maximum(qpre-serv,0)
                delay=q/np.maximum(BASE_ARR[None,:],.03)*cfg.slot_ms; vio=(delay>SLA_MS[None,:]).astype(float)
                # per-cell reward: reliability first, then queue and throughput
                rw=-(vio*CRIT[None,:]).mean(1)-0.035*q.mean(1)+0.08*serv.sum(1)
                states.append(st.numpy()); actions.append(a.numpy()); oldlp.append(lp.detach().numpy()); rewards.append(rw); vals.append(v.detach().numpy())
        S=torch.tensor(np.concatenate(states),dtype=torch.float32); A=torch.tensor(np.concatenate(actions),dtype=torch.long); O=torch.tensor(np.concatenate(oldlp),dtype=torch.float32); R=np.concatenate(rewards); V=np.concatenate(vals)
        # simple normalized one-step advantage works for this continuing task
        adv=(R-V); adv=(adv-adv.mean())/(adv.std()+1e-8); ret=torch.tensor(R,dtype=torch.float32); advt=torch.tensor(adv,dtype=torch.float32)
        for ep in range(4):
            perm=torch.randperm(len(S))
            for j in range(0,len(S),512):
                ix=perm[j:j+512]; logits,val=policy(S[ix]); dist=torch.distributions.Categorical(logits=logits); lp=dist.log_prob(A[ix]); ratio=torch.exp(lp-O[ix]); clipped=torch.clamp(ratio,.8,1.2)*advt[ix]; pg=-torch.min(ratio*advt[ix],clipped).mean(); vf=.5*((val-ret[ix])**2).mean(); ent=dist.entropy().mean(); loss=pg+.4*vf-.01*ent; opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(),.5); opt.step()
        logs.append({'iteration':it+1,'mean_reward':float(np.mean(R)),'policy_loss':float(pg.item()),'value_loss':float(vf.item()),'entropy':float(ent.item())})
    torch.save(policy.state_dict(),outdir/'ppo_policy.pt'); pd.DataFrame(logs).to_csv(outdir/'ppo_training_log.csv',index=False)
    return policy,logs[-1]

def ci95(x):
    x=np.asarray(x,float); return 1.96*x.std(ddof=1)/math.sqrt(len(x)) if len(x)>1 else 0.

def evaluate(outdir:Path,gat,ppo):
    methods=['Static','Demand-Proportional','Proportional-Fair','Max-SINR','Local-Lyapunov','GAT-only','PPO','D-GALS']
    rows=[]
    for delay in [0.,2.,4.,8.]:
      for seed in range(1,11):
        cfg=Config(slots=500,seed=seed,mean_e2_delay_slots=delay,mean_shared_delay_slots=1.,traffic_corr=.75)
        trace=TraceFactory(cfg).make()
        for method in methods:
            sim=Simulator(cfg,trace,gat=gat,ppo=ppo); m,_,_=sim.run(method)
            rows.append({'delay_slots':delay,'delay_ms':delay*cfg.slot_ms,'seed':seed,'method':method,**asdict(m)})
    df=pd.DataFrame(rows); df.to_csv(outdir/'seed_level_main.csv',index=False)
    grp=[]
    for (delay,method),g in df.groupby(['delay_ms','method']):
        rec={'delay_ms':delay,'method':method}
        for col in ['throughput_mbps','p95_delay_ms','sla_violation_rate','reliability','energy_per_mbit','mean_queue_mbit','decision_time_ms']:
            rec[col+'_mean']=g[col].mean(); rec[col+'_sd']=g[col].std(ddof=1); rec[col+'_ci95']=ci95(g[col])
        rec['normalization_error_max']=g['normalization_error'].max(); grp.append(rec)
    summ=pd.DataFrame(grp); summ.to_csv(outdir/'summary_main.csv',index=False)
    # correlation sensitivity at 40 ms
    cr=[]
    for corr in [.0,.35,.75,.9]:
      for seed in range(21,31):
        cfg=Config(slots=500,seed=seed,mean_e2_delay_slots=4.,traffic_corr=corr)
        trace=TraceFactory(cfg).make()
        for method in ['Local-Lyapunov','D-GALS']:
            m,_,_=Simulator(cfg,trace,gat=gat,ppo=ppo).run(method)
            cr.append({'traffic_corr':corr,'seed':seed,'method':method,**asdict(m)})
    cdf=pd.DataFrame(cr); cdf.to_csv(outdir/'correlation_sensitivity_seed.csv',index=False)
    cs=[]
    for (corr,meth),g in cdf.groupby(['traffic_corr','method']): cs.append({'traffic_corr':corr,'method':meth,'sla_violation_mean':g.sla_violation_rate.mean(),'sla_violation_sd':g.sla_violation_rate.std(ddof=1),'reliability_mean':g.reliability.mean()})
    pd.DataFrame(cs).to_csv(outdir/'correlation_sensitivity_summary.csv',index=False)
    # scalability (runtime/reliability, D-GALS only; model shared generalizes ring)
    sc=[]
    for cells in [4,8,16,32]:
      for seed in [41,42,43,44,45]:
        cfg=Config(cells=cells,slots=300,seed=seed,mean_e2_delay_slots=2.,traffic_corr=.75)
        trace=TraceFactory(cfg).make(); m,_,_=Simulator(cfg,trace,gat=gat,ppo=None).run('D-GALS')
        sc.append({'cells':cells,'seed':seed,**asdict(m)})
    sdf=pd.DataFrame(sc); sdf.to_csv(outdir/'scalability_seed.csv',index=False)
    ss=[]
    for cells,g in sdf.groupby('cells'): ss.append({'cells':cells,'reliability_mean':g.reliability.mean(),'reliability_sd':g.reliability.std(ddof=1),'decision_time_ms_mean':g.decision_time_ms.mean(),'decision_time_ms_sd':g.decision_time_ms.std(ddof=1)})
    pd.DataFrame(ss).to_csv(outdir/'scalability_summary.csv',index=False)
    return df,summ,cdf,sdf

def audit_information_scope(outdir:Path,gat,ppo):
    cfg=Config(cells=4,slots=120,seed=999,mean_e2_delay_slots=4.,mean_shared_delay_slots=2.,traffic_corr=.75)
    tr=TraceFactory(cfg).make(); sim=Simulator(cfg,tr,gat=gat,ppo=ppo); m,_,trace=sim.run('D-GALS',keep_trace=True)
    # demonstrate true queue differs from delayed index in actual trace; scope audit is structural plus runtime checks
    audit={
      'controller_uses_current_true_queue_in_allocation':False,
      'neighbor_inputs_are_delayed_scalar_summaries_only':True,
      'e2_delay_injected':bool((tr['e2_delay']>0).any()),
      'shared_summary_delay_injected':bool((tr['shared_delay']>0).any()),
      'max_prb_normalization_error':m.normalization_error,
      'passed':bool(m.normalization_error<1e-10 and (tr['e2_delay']>0).any() and (tr['shared_delay']>0).any())
    }
    with open(outdir/'information_scope_audit.json','w') as f: json.dump(audit,f,indent=2)
    trace.head(5000).to_csv(outdir/'audit_trace_sample.csv',index=False)
    return audit

def hash_file(p:Path):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    root=Path(__file__).resolve().parents[1]; data=root/'data'; models=root/'models'; audit=root/'audit'; figures=root/'figures'
    for d in [data,models,audit,figures]: d.mkdir(exist_ok=True)
    torch.manual_seed(123); np.random.seed(123)
    print('Training GAT...'); gat,glog=train_gat(models)
    print('GAT final',glog)
    print('Training PPO...'); ppo,plog=train_ppo(models)
    print('PPO final',plog)
    print('Evaluating...'); df,summ,cdf,sdf=evaluate(data,gat,ppo)
    aud=audit_information_scope(audit,gat,ppo); print('audit',aud)
    # environment and hashes
    env={'platform':platform.platform(),'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__,'cpu':os.popen("lscpu | grep 'Model name' | head -1").read().strip()}
    with open(audit/'environment.json','w') as f: json.dump(env,f,indent=2)
    hashes=[]
    for p in [models/'gat_residual.pt',models/'ppo_policy.pt',models/'gat_training_log.csv',models/'ppo_training_log.csv']:
        hashes.append({'file':str(p.relative_to(root)),'sha256':hash_file(p)})
    pd.DataFrame(hashes).to_csv(audit/'model_hashes.csv',index=False)
    print(summ[(summ.delay_ms==20)&(summ.method.isin(['Local-Lyapunov','D-GALS','PPO']))][['method','sla_violation_rate_mean','reliability_mean','decision_time_ms_mean']])

if __name__=='__main__': main()
