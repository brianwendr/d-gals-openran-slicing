from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]
a=json.load(open(root/'audit'/'information_scope_audit.json'))
assert a['controller_uses_current_true_queue_in_allocation'] is False
assert a['neighbor_inputs_are_delayed_scalar_summaries_only'] is True
assert a['e2_delay_injected'] is True
assert a['shared_summary_delay_injected'] is True
assert a['max_prb_normalization_error'] < 1e-10
print('scope/normalization audit: PASS')
