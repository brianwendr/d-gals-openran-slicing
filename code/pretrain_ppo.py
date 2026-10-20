from pathlib import Path
import sys, numpy as np, torch
import torch.nn as nn, torch.optim as optim
sys.path.insert(0,str(Path(__file__).resolve().parent))
from rebuilt_study import Config, TraceFactory, PPOPolicy, ACTION_TEMPLATES, K, BASE_ARR, CRIT, sigmoid, ring_neighbors, normalize_with_floors, train_ppo

root=Path(__file__).resolve().parents[1]; models=root/'models'
torch.manual_seed(2028); np.random.seed(2028)
policy=PPOPolicy(21,len(ACTION_TEMPLATES)); opt=optim.Adam(policy.parameters(),lr=1e-3)
X=[];Y=[]
for seed in range(300,308):
    cfg=Config(cells=4,slots=260,seed=seed,mean_e2_delay_slots=2.,mean_shared_delay_slots=1.,traffic_corr=.75)
    tr=TraceFactory(cfg).make(); c=cfg.cells; neigh=ring_neighbors(c); q=np.zeros((c,K)); qhist=np.zeros((cfg.slots+1,c,K)); pihist=np.zeros((cfg.slots+1,c,K))
    for t in range(cfg.slots):
        qhist[t]=q
        states=[]; expert=[]
        for b in range(c):
          for k in range(K):
            d=int(tr['e2_delay'][t,b,k]); j=max(0,t-d); qtil=qhist[j,b,k]; pihist[t,b,k]=float(sigmoid((qtil/(BASE_ARR[k]+.08)-1)*.8))
        for b in range(c):
          vals=[]; qobs=np.zeros(K)
          for k in range(K):
            d=int(tr['e2_delay'][t,b,k]); j=max(0,t-d); qtil=qhist[j,b,k]; eta=tr['eta'][j,b,k]; qobs[k]=qtil
            nvals=[]; ages=[]
            for nb in neigh[b]:
              dd=int(tr['shared_delay'][t,nb,k]); jj=max(0,t-dd); nvals.append(pihist[jj,nb,k]); ages.append(min(dd,12)/6)
            vals.extend([qtil/2,tr['arrival'][j,b,k]/.5,eta/3,min(d,12)/6,CRIT[k]/3,float(np.mean(nvals)),float(np.mean(ages))])
          aobs=np.array([tr['arrival'][max(0,t-int(tr['e2_delay'][t,b,kk])),b,kk] for kk in range(K)]); ex=normalize_with_floors((aobs+1.15*CRIT*(qobs/(.12+BASE_ARR)))[None,:])[0]
          label=int(np.argmin(((ACTION_TEMPLATES-ex[None,:])**2).sum(1)))
          X.append(vals);Y.append(label); expert.append(ACTION_TEMPLATES[label])
        alloc=np.asarray(expert); qpre=q+tr['arrival'][t]; eta=tr['eta'][t]; cap=cfg.capacity_scale*(.70+.30*eta/np.maximum(eta.mean(1,keepdims=True),1e-6)); q=np.maximum(qpre-np.minimum(qpre,alloc*cap),0)
X=torch.tensor(np.asarray(X),dtype=torch.float32);Y=torch.tensor(Y,dtype=torch.long)
for ep in range(15):
    perm=torch.randperm(len(X)); ls=[]
    for i in range(0,len(X),512):
        ix=perm[i:i+512]; logits,_=policy(X[ix]); loss=nn.functional.cross_entropy(logits,Y[ix]); opt.zero_grad(); loss.backward();opt.step();ls.append(loss.item())
    if ep in {0,4,9,14}:
      with torch.no_grad(): acc=(policy(X)[0].argmax(1)==Y).float().mean().item(); print(ep+1,np.mean(ls),acc)
torch.save(policy.state_dict(),models/'ppo_policy_bc_init.pt')
print('saved init')
