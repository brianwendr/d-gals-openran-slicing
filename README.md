# Online Resource 1 — D-GALS Open RAN Slicing Reproduction Package

This package contains executable controller code, trained checkpoints, seed-level data, figures, and audit material for the manuscript. It excludes the manuscript and editorial documents.

- **3.5-GHz UMa-inspired full-frequency-reuse replay.** `code/wireless_realism_v24.py` uses LOS/NLOS path loss based on the 3GPP TR 38.901 UMa form, log-normal shadowing, Rayleigh power fading, thermal noise, explicit inter-cell interference, bounded mobility, and spectral-efficiency traces consumed by the same delayed-information controller.
- **Constant-ISD topology scaling.** The 4-, 8-, and 16-cell ring layouts keep the nearest-neighbor inter-site distance at approximately 500 m. This avoids conflating network size with artificial cell densification. The 80-ms stress case is replayed with eight independent seeds.
- **Standard four-cell wireless replay.** The four-cell UMa sensitivity is directly regenerated at 20 and 80 ms mean E2 delay, producing `data/uma_controller_replay_seed.csv` and `data/uma_controller_replay_summary.csv`.
- **Load-coupled E2 delay.** A geometric base delay is augmented by an exogenous traffic-load-dependent congestion increment. The identical delay trace is used for paired controllers.
- **Link-budget plausibility audit.** `code/wireless_link_audit_v24.py` reports serving-distance, LOS, SINR, and spectral-efficiency summaries at 4, 8, and 16 cells while preserving 500 m nearest-neighbor ISD. It is an audit only and does not feed the primary results.

These additions strengthen wireless plausibility and delay robustness checks, but they are **not** presented as a packet-level 5G NR emulator or an over-the-air O-RAN testbed.

## Quick verification

```bash
python -m pip install -r requirements.txt
python audit/smoke_test_core_repairs.py
```

The smoke test checks nonzero E2/shared-summary delay injection, genuinely stale observations, zero-delay D-GALS/Local-Lyapunov equivalence, model loading/inference, and exact PRB normalization.

## Regenerate the primary controller study

Using supplied checkpoints:

```bash
python code/run_full_study.py
```

To retrain the graph residual predictor and PPO first:

```bash
python code/run_full_study.py --retrain
```

## Regenerate the V24 wireless-realism sensitivity

```bash
python code/wireless_realism_v24.py
python code/wireless_link_audit_v24.py
python code/delay_robustness_v25.py
python code/statistical_audit_v25.py
```

The wireless-realism scripts regenerate:

- `data/uma_controller_replay_seed.csv`
- `data/uma_controller_replay_summary.csv`
- `data/uma_topology_replay_seed.csv`
- `data/uma_topology_replay_summary.csv`
- `data/load_coupled_e2_seed.csv`
- `data/load_coupled_e2_summary.csv`
- `data/uma_link_budget_audit_seed.csv`
- `data/uma_link_budget_audit_summary.csv`
- `figures/figure6_uma_replay_v24.png`
- `figures/figure7_uma_topology_scaling.png`
- `figures/figure8_load_coupled_e2.png`
- `data/delay_family_sensitivity_seed.csv`
- `data/delay_family_sensitivity_summary.csv`
- `data/shared_context_delay_sensitivity_seed.csv`
- `data/shared_context_delay_sensitivity_summary.csv`
- `data/paired_statistical_audit_v25.csv`
- `figures/supp_figure_s1_delay_family_sensitivity.png`
- `figures/supp_figure_s2_shared_context_delay_sensitivity.png`

## Main files

- `code/rebuilt_study.py`: delayed-observation simulator, trace generator, graph residual model, PPO model, and training functions.
- `code/fast_eval.py`: primary eight-seed evaluation, delay stress, residual/correlation sensitivity, scalability study, and information-scope audit.
- `code/run_full_study.py`: primary end-to-end reproduction entry point.
- `code/wireless_realism_v24.py`: UMa full-reuse replay, constant-ISD topology scaling, and load-coupled E2-delay sensitivity.
- `code/wireless_link_audit_v24.py`: independent link-budget plausibility audit.
- `code/delay_robustness_v25.py`: matched-mean E2 age-law and shared-context-age sensitivity.
- `code/statistical_audit_v25.py`: conservative paired statistical audit for primary and V25 sensitivity results.
- `models/gat_residual.pt`: trained graph residual predictor.
- `models/ppo_policy.pt`: PPO-fine-tuned policy/value checkpoint.
- `data/seed_level_main.csv`, `data/summary_main.csv`: primary seed-level and summary results.
- `data/residual_predictor_sensitivity.csv`: graph-vs-local residual-prediction sensitivity.
- `data/scalability_seed.csv`, `data/scalability_summary.csv`: runtime sensitivity.
- `data/uma_controller_replay_*`: standard four-cell UMa replay.
- `data/uma_topology_replay_*`: 4/8/16-cell constant-ISD topology replay.
- `data/load_coupled_e2_*`: load-dependent reporting-age sensitivity.
- `data/uma_link_budget_audit_*`: radio plausibility diagnostics.
- `audit/information_scope_audit.json`: information-boundary and PRB-normalization audit.
- `audit/core_repairs_smoke_test.json`: automated core-repair smoke-test result.

## Claim boundary

The mathematical claim is a stale-state mean-square bound plus an empirically measured residual-reduction factor. The package does **not** claim a universal `O(tau^2)` to `O(tau)` order reduction. The controller experiments establish robustness within the simulated information and radio models, including alternative synthetic E2 age laws and shared-context ages. They do not identify a measured E2/RIC transport distribution; packet-level NR timing, scheduler signaling, HARQ, and measured transport remain future validation targets.
