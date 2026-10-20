#!/usr/bin/env python3
"""Reproduce the rebuilt D-GALS study.

Default behavior uses the supplied trained checkpoints and regenerates evaluation
CSVs/figures. Pass --retrain to retrain GAT and PPO before evaluation.
"""
from pathlib import Path
import argparse, subprocess, sys, torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from rebuilt_study import GraphResidualNet, PPOPolicy, ACTION_TEMPLATES, train_gat, train_ppo

root=Path(__file__).resolve().parents[1]; models=root/'models'
ap=argparse.ArgumentParser();ap.add_argument('--retrain',action='store_true');args=ap.parse_args()
if args.retrain:
    print('Retraining graph residual predictor...')
    train_gat(models,epochs=30)
    print('Behavior-cloning warm start for PPO...')
    subprocess.run([sys.executable,str(root/'code'/'pretrain_ppo.py')],check=True)
    print('PPO fine-tuning...')
    train_ppo(models,cells=4,train_slots=220,iterations=8,rollouts_per_iter=1)
for p in ['gat_residual.pt','ppo_policy.pt']:
    if not (models/p).exists(): raise FileNotFoundError(models/p)
print('Regenerating evaluation outputs...')
subprocess.run([sys.executable,str(root/'code'/'fast_eval.py')],check=True)
print('Regenerating residual-predictor sensitivity...')
subprocess.run([sys.executable,str(root/'code'/'residual_sensitivity.py')],check=True)
print('Regenerating figures...')
subprocess.run([sys.executable,str(root/'code'/'make_figures.py')],check=True)
print('Done.')
