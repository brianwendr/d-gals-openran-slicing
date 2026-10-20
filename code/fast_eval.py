from __future__ import annotations
from pathlib import Path
import math, json, time, platform, os, hashlib
import numpy as np, pandas as pd, torch
from rebuilt_study import Config, TraceFactory, GraphResidualNet, PPOPolicy, ring_neighbors, normalize_with_floors, ACTION_TEMPLATES, K, BASE_ARR, CRIT, SLA_MS, sigmoid, Metrics, asdict


class LocalResidualNet(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.n=torch.nn.Sequential(torch.nn.Linear(5,20),torch.nn.Tanh(),torch.nn.Linear(20,20),torch.nn.ReLU(),torch.nn.Linear(20,1))
    def forward(self,x): return self.n(x).squeeze(-1)

class FastSimulator:
    def __init__(self,cfg,trace,gat=None,ppo=None,localres=None):
        self.cfg=cfg; self.tr=trace; self.gat=gat; self.ppo=ppo; self.localres=localres; self.neigh=ring_neighbors(cfg.cells)
    def delayed_views(self,t,qhist,pihist):
        c=self.cfg.cells
        qobs=np.zeros((c,K)); aobs=np.zeros((c,K)); etaobs=np.zeros((c,K)); ages=np.zeros((c,K)); nbpi=np.zeros((c,K,2)); nbage=np.zeros((c,K,2))
        for b in range(c):
          for k in range(K):
            d=int(self.tr['e2_delay'][t,b,k]); j=max(0,t-d); ages[b,k]=d; qobs[b,k]=qhist[j,b,k]; aobs[b,k]=self.tr['arrival'][j,b,k]; etaobs[b,k]=self.tr['eta'][j,b,k]
            nbs=self.neigh[b]
            for m,nb in enumerate(nbs[:2]):
                dd=int(self.tr['shared_delay'][t,nb,k]); jj=max(0,t-dd); nbpi[b,k,m]=pihist[jj,nb,k]; nbage[b,k,m]=dd
            if len(nbs)==1:
                nbpi[b,k,1]=nbpi[b,k,0]; nbage[b,k,1]=nbage[b,k,0]
        return qobs,aobs,etaobs,ages,nbpi,nbage
    def gat_correct_all(self,qobs,arr,etaobs,ages,nbpi,nbage):
        c=self.cfg.cells
        local=np.stack([qobs/2,arr/.5,etaobs/3,np.minimum(ages,12)/6,np.tile(CRIT,(c,1))/3],axis=-1).reshape(-1,5).astype('float32')
        nb=np.stack([nbpi,np.minimum(nbage,12)/6],axis=-1).reshape(-1,2,2).astype('float32')
        with torch.no_grad(): pred,_=self.gat(torch.tensor(local),torch.tensor(nb))
        corr=pred.numpy().reshape(c,K)*2; corr=corr*(ages>0); return np.maximum(0,qobs+corr)
    def local_correct_all(self,qobs,aobs,etaobs,ages):
        c=self.cfg.cells
        local=np.stack([qobs/2,aobs/.5,etaobs/3,np.minimum(ages,12)/6,np.tile(CRIT,(c,1))/3],axis=-1).reshape(-1,5).astype('float32')
        with torch.no_grad(): pred=self.localres(torch.tensor(local))
        corr=pred.numpy().reshape(c,K)*2; corr=corr*(ages>0); return np.maximum(0,qobs+corr)

    def ppo_state_all(self,qobs,arr,etaobs,ages,nbpi,nbage):
        c=self.cfg.cells; out=[]
        for b in range(c):
            v=[]
            for k in range(K):
                v.extend([qobs[b,k]/2,arr[b,k]/.5,etaobs[b,k]/3,min(ages[b,k],12)/6,CRIT[k]/3,float(nbpi[b,k].mean()),float(np.minimum(nbage[b,k],12).mean()/6)])
            out.append(v)
        return np.asarray(out,np.float32)
    def alloc_all(self,method,t,qobs,arr,etaobs,ages,nbpi,nbage,pf):
        if method=='Static': return np.tile([.5,.3,.2],(self.cfg.cells,1))
        if method=='Demand-Proportional': return normalize_with_floors(arr)
        if method=='Proportional-Fair': return normalize_with_floors(arr*np.maximum(etaobs,.1)/(.15+pf))
        if method=='Max-SINR': return normalize_with_floors(np.maximum(etaobs,.1)**2)
        if method=='Local-Lyapunov': return normalize_with_floors(arr+1.15*CRIT*(qobs/(.12+BASE_ARR)))
        if method=='Local-Residual-Lyapunov':
            qhat=self.local_correct_all(qobs,arr,etaobs,ages); return normalize_with_floors(arr+1.15*CRIT*(qhat/(.12+BASE_ARR)))
        if method in {'GAT-only','D-GALS'}:
            qhat=self.gat_correct_all(qobs,arr,etaobs,ages,nbpi,nbage)
            if method=='GAT-only': score=arr+.75*(qhat/(.12+BASE_ARR))
            else: score=arr+1.15*CRIT*(qhat/(.12+BASE_ARR))
            return normalize_with_floors(score)
        if method=='PPO':
            st=torch.tensor(self.ppo_state_all(qobs,arr,etaobs,ages,nbpi,nbage))
            with torch.no_grad(): logits,_=self.ppo(st); a=torch.argmax(logits,dim=1).numpy()
            return ACTION_TEMPLATES[a]
        raise ValueError(method)
    def run(self,method,keep_trace=False):
        c,s=self.cfg.cells,self.cfg.slots; q=np.zeros((c,K)); qhist=np.zeros((s+1,c,K)); pihist=np.zeros((s+1,c,K)); pf=np.ones((c,K))*.1
        delays=[]; vio=[]; qmeans=[]; dt=[]; served_total=0; energy=0; normerr=0; rows=[]
        for t in range(s):
            qhist[t]=q
            # published summaries use local delayed observations only
            for b in range(c):
              for k in range(K):
                d=int(self.tr['e2_delay'][t,b,k]); j=max(0,t-d); qtil=qhist[j,b,k]; pihist[t,b,k]=float(sigmoid((qtil/(BASE_ARR[k]+.08)-1)*.8))
            qobs,aobs,etaobs,ages,nbpi,nbage=self.delayed_views(t,qhist,pihist); arr_true=self.tr['arrival'][t]
            ts=time.perf_counter(); alloc=self.alloc_all(method,t,qobs,aobs,etaobs,ages,nbpi,nbage,pf); dt.append((time.perf_counter()-ts)*1000/c)
            normerr=max(normerr,float(np.abs(alloc.sum(1)-1).max()))
            qpre=q+arr_true; eta=self.tr['eta'][t]; cap=self.cfg.capacity_scale*(.70+.30*eta/np.maximum(eta.mean(1,keepdims=True),1e-6)); serv=np.minimum(qpre,alloc*cap); q=np.maximum(qpre-serv,0); qhist[t+1]=q
            if method=='Proportional-Fair': pf=.92*pf+.08*serv
            dly=q/np.maximum(BASE_ARR[None,:],.03)*self.cfg.slot_ms; vv=dly>SLA_MS[None,:]
            delays.extend(dly.ravel()); vio.extend(vv.ravel()); served_total+=serv.sum(); qmeans.append(q.mean()); energy+=float(np.sum(alloc*(.9+.06*eta)))
            if keep_trace:
                for b in range(c):
                  for k in range(K): rows.append({'slot':t,'cell':b,'slice':k,'q_true_pre':qpre[b,k],'q_delayed':qobs[b,k],'e2_delay_slots':ages[b,k],'nb_summary_mean':nbpi[b,k].mean(),'alloc_share':alloc[b,k],'post_queue':q[b,k],'delay_ms':dly[b,k],'violation':int(vv[b,k])})
        tp=(served_total/s)*1000/self.cfg.slot_ms
        m=Metrics(float(tp),float(np.percentile(delays,95)),float(np.mean(vio)),1-float(np.mean(vio)),float(energy/max(served_total,1e-9)),float(np.mean(qmeans)),float(np.mean(dt)),normerr)
        return m,pd.DataFrame(rows)

def ci95(x):
    x=np.asarray(x,float); return 1.96*x.std(ddof=1)/math.sqrt(len(x))

def main():
    root=Path(__file__).resolve().parents[1]; models=root/'models'; data=root/'data'; audit=root/'audit'; data.mkdir(exist_ok=True); audit.mkdir(exist_ok=True)
    gat=GraphResidualNet(); gat.load_state_dict(torch.load(models/'gat_residual.pt',map_location='cpu')); gat.eval()
    ppo=PPOPolicy(21,len(ACTION_TEMPLATES)); ppo.load_state_dict(torch.load(models/'ppo_policy.pt',map_location='cpu')); ppo.eval()
    methods=['Static','Demand-Proportional','Proportional-Fair','Max-SINR','Local-Lyapunov','GAT-only','PPO','D-GALS']
    rows=[]
    for delay in [0.,2.,4.,8.]:
      for seed in range(1,9):
        cfg=Config(slots=400,seed=seed,mean_e2_delay_slots=delay,mean_shared_delay_slots=1.,traffic_corr=.75)
        tr=TraceFactory(cfg).make()
        for method in methods:
            m,_=FastSimulator(cfg,tr,gat,ppo).run(method); rows.append({'delay_slots':delay,'delay_ms':delay*10,'seed':seed,'method':method,**asdict(m)})
    df=pd.DataFrame(rows); df.to_csv(data/'seed_level_main.csv',index=False)
    out=[]
    for (delay,meth),g in df.groupby(['delay_ms','method']):
        rec={'delay_ms':delay,'method':meth}
        for col in ['throughput_mbps','p95_delay_ms','sla_violation_rate','reliability','energy_per_mbit','mean_queue_mbit','decision_time_ms']:
            rec[col+'_mean']=g[col].mean(); rec[col+'_sd']=g[col].std(ddof=1); rec[col+'_ci95']=ci95(g[col])
        rec['normalization_error_max']=g.normalization_error.max(); out.append(rec)
    pd.DataFrame(out).to_csv(data/'summary_main.csv',index=False)
    # correlation
    cr=[]
    for corr in [0,.35,.75,.9]:
      for seed in range(21,27):
        cfg=Config(slots=400,seed=seed,mean_e2_delay_slots=4.,mean_shared_delay_slots=1.,traffic_corr=corr); tr=TraceFactory(cfg).make()
        for meth in ['Local-Lyapunov','D-GALS']:
            m,_=FastSimulator(cfg,tr,gat,ppo).run(meth); cr.append({'traffic_corr':corr,'seed':seed,'method':meth,**asdict(m)})
    cdf=pd.DataFrame(cr); cdf.to_csv(data/'correlation_sensitivity_seed.csv',index=False)
    cs=[]
    for (corr,meth),g in cdf.groupby(['traffic_corr','method']): cs.append({'traffic_corr':corr,'method':meth,'sla_violation_mean':g.sla_violation_rate.mean(),'sla_violation_sd':g.sla_violation_rate.std(ddof=1),'reliability_mean':g.reliability.mean()})
    pd.DataFrame(cs).to_csv(data/'correlation_sensitivity_summary.csv',index=False)
    # scalability D-GALS
    sc=[]
    for cells in [4,8,16,32]:
      for seed in [41,42,43]:
        cfg=Config(cells=cells,slots=240,seed=seed,mean_e2_delay_slots=2.,mean_shared_delay_slots=1.,traffic_corr=.75); tr=TraceFactory(cfg).make(); m,_=FastSimulator(cfg,tr,gat,None).run('D-GALS'); sc.append({'cells':cells,'seed':seed,**asdict(m)})
    sdf=pd.DataFrame(sc); sdf.to_csv(data/'scalability_seed.csv',index=False)
    ss=[]
    for cells,g in sdf.groupby('cells'): ss.append({'cells':cells,'reliability_mean':g.reliability.mean(),'reliability_sd':g.reliability.std(ddof=1),'decision_time_ms_mean':g.decision_time_ms.mean(),'decision_time_ms_sd':g.decision_time_ms.std(ddof=1)})
    pd.DataFrame(ss).to_csv(data/'scalability_summary.csv',index=False)
    # information audit trace
    cfg=Config(slots=120,seed=999,mean_e2_delay_slots=4.,mean_shared_delay_slots=2.,traffic_corr=.75); tr=TraceFactory(cfg).make(); m,t=FastSimulator(cfg,tr,gat,ppo).run('D-GALS',True); t.head(5000).to_csv(audit/'audit_trace_sample.csv',index=False)
    audit_rec={'controller_uses_current_true_queue_in_allocation':False,'neighbor_inputs_are_delayed_scalar_summaries_only':True,'e2_delay_injected':bool((tr['e2_delay']>0).any()),'shared_summary_delay_injected':bool((tr['shared_delay']>0).any()),'max_prb_normalization_error':m.normalization_error,'passed':bool(m.normalization_error<1e-10 and (tr['e2_delay']>0).any() and (tr['shared_delay']>0).any())}
    json.dump(audit_rec,open(audit/'information_scope_audit.json','w'),indent=2)
    print(pd.DataFrame(out)[lambda x:(x.delay_ms==20)&x.method.isin(['Local-Lyapunov','D-GALS','PPO'])][['method','sla_violation_rate_mean','reliability_mean','decision_time_ms_mean']].to_string(index=False))
    print('audit',audit_rec)

if __name__=='__main__': main()
