from pathlib import Path
import json, sys, numpy as np, torch
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'code'))
from rebuilt_study import Config, TraceFactory, GraphResidualNet, PPOPolicy, ACTION_TEMPLATES
from fast_eval import FastSimulator

models=root/'models'; audit=root/'audit'
# Load supplied trained models
gat=GraphResidualNet(); gat.load_state_dict(torch.load(models/'gat_residual.pt',map_location='cpu')); gat.eval()
ppo=PPOPolicy(21,len(ACTION_TEMPLATES)); ppo.load_state_dict(torch.load(models/'ppo_policy.pt',map_location='cpu')); ppo.eval()

# A small deterministic audit trace with nonzero E2 and shared-summary delay
cfg=Config(cells=4,slots=80,seed=2468,mean_e2_delay_slots=3.0,mean_shared_delay_slots=1.0,traffic_corr=.75)
tr=TraceFactory(cfg).make()
assert (tr['e2_delay']>0).any(), 'E2 delay injection missing'
assert (tr['shared_delay']>0).any(), 'shared-summary delay injection missing'

# Run proposed method and verify normalized allocation + audit fields
m, trace = FastSimulator(cfg,tr,gat,ppo).run('D-GALS',keep_trace=True)
assert m.normalization_error < 1e-10, m.normalization_error
assert 'q_delayed' in trace and 'q_true_pre' in trace and 'e2_delay_slots' in trace
# There must be observations where a positive age produces a genuinely stale queue.
pos=trace[trace.e2_delay_slots>0]
assert len(pos)>0
assert np.any(np.abs(pos.q_true_pre.to_numpy()-pos.q_delayed.to_numpy())>1e-12), 'positive-age observations never differ from current state'

# Zero-delay equivalence: graph correction is gated off, so D-GALS == Local-Lyapunov.
cfg0=Config(cells=4,slots=60,seed=1357,mean_e2_delay_slots=0.0,mean_shared_delay_slots=1.0,traffic_corr=.75)
tr0=TraceFactory(cfg0).make()
ml,_=FastSimulator(cfg0,tr0,gat,ppo).run('Local-Lyapunov')
mg,_=FastSimulator(cfg0,tr0,gat,ppo).run('D-GALS')
assert abs(ml.sla_violation_rate-mg.sla_violation_rate)<1e-12
assert abs(ml.throughput_mbps-mg.throughput_mbps)<1e-12

# Model input/output sanity checks.
with torch.no_grad():
    pred,att=gat(torch.zeros((2,5)),torch.zeros((2,2,2)))
    logits,val=ppo(torch.zeros((2,21)))
assert pred.shape==(2,) and att.shape==(2,2)
assert logits.shape==(2,len(ACTION_TEMPLATES)) and val.shape==(2,)

report={
    'e2_delay_injected': True,
    'shared_summary_delay_injected': True,
    'positive_age_differs_from_current_state': True,
    'zero_delay_dgals_equals_local_lyapunov': True,
    'gat_checkpoint_loads_and_runs': True,
    'ppo_checkpoint_loads_and_runs': True,
    'max_prb_normalization_error': float(m.normalization_error),
    'status':'PASS'
}
(audit/'core_repairs_smoke_test.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
