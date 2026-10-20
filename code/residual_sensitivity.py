from pathlib import Path
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from rebuilt_study import Config, TraceFactory, Simulator, GraphResidualNet
from fast_eval import LocalResidualNet
root=Path(__file__).resolve().parents[1];data=root/'data';models=root/'models'
g=GraphResidualNet();g.load_state_dict(torch.load(models/'gat_residual.pt',map_location='cpu'));g.eval()
l=LocalResidualNet();l.load_state_dict(torch.load(models/'local_residual.pt',map_location='cpu'));l.eval()
rows=[]
for corr in [0,.35,.75,.9]:
    L=[];N=[];Y=[]
    for seed in [501,502,503]:
        cfg=Config(cells=4,slots=350,seed=seed,mean_e2_delay_slots=4.,traffic_corr=corr,capacity_scale=.8)
        tr=TraceFactory(cfg).make(); _,samples,_=Simulator(cfg,tr).run('Demand-Proportional',collect_gat=True)
        for a,b,y,*_ in samples: L.append(a);N.append(b);Y.append(y)
    X=torch.tensor(np.stack(L));NB=torch.tensor(np.stack(N));y=torch.tensor(np.array(Y,np.float32))
    with torch.no_grad(): pg,_=g(X,NB);pl=l(X)
    base=float((y**2).mean()); mg=float(((pg-y)**2).mean()); ml=float(((pl-y)**2).mean())
    rows.append({'traffic_corr':corr,'uncorrected_mse':base,'graph_mse':mg,'local_mlp_mse':ml,'graph_kappa':mg/base,'local_kappa':ml/base})
pd.DataFrame(rows).to_csv(data/'residual_predictor_sensitivity.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
