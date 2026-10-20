from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
root=Path(__file__).resolve().parents[1]; figdir=root/'figures';figdir.mkdir(exist_ok=True);data=root/'data'

def box(ax,x,y,w,h,text,fs=14,fc='white'):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.025',linewidth=1.6,edgecolor='black',facecolor=fc);ax.add_patch(p);ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs);return p

def arrow(ax,p1,p2,text=None,txy=None,style='->',ls='-',lw=1.7):
 a=FancyArrowPatch(p1,p2,arrowstyle=style,mutation_scale=15,linewidth=lw,linestyle=ls,color='black');ax.add_patch(a)
 if text and txy: ax.text(*txy,text,ha='center',va='center',fontsize=11)
# Fig 1 corrected architecture
fig,ax=plt.subplots(figsize=(14,8));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
box(ax,.31,.58,.38,.24,'near-RT RIC\nper-cell D-GALS xApps\n+ shared context (SDL/RMR)',20,'#f5f5f5')
box(ax,.05,.70,.18,.14,'Conflict\nmitigation /\nplatform policy',16,'#f5f5f5')
box(ax,.77,.70,.18,.14,'SMO / Non-RT RIC\nO1/A1 policy and\nmanagement context',15,'#f5f5f5')
for x,i in [(.06,1),(.39,2),(.72,3)]: box(ax,x,.12,.22,.17,f'O-DU / Cell {i}\nlocal queues\nPRB scheduler',16,'#f5f5f5')
arrow(ax,(.23,.77),(.31,.75),'platform checks',(.27,.84));arrow(ax,(.69,.75),(.77,.75),'A1/O1 context',(.73,.84));arrow(ax,(.77,.66),(.69,.66),'non-RT guidance',(.73,.60))
arrow(ax,(.17,.29),(.41,.58));arrow(ax,(.50,.29),(.50,.58));arrow(ax,(.83,.29),(.59,.58));ax.text(.50,.44,'Delayed E2 telemetry',ha='center',fontsize=13)
arrow(ax,(.28,.20),(.39,.20),'near-RT scalar summary',(.335,.25),style='<->');arrow(ax,(.61,.20),(.72,.20),'near-RT scalar summary',(.665,.25),style='<->')
ax.text(.50,.035,'O1 is not used for per-slot xApp coordination; near-RT summaries are modeled through RIC-internal shared context.',ha='center',fontsize=11)
fig.savefig(figdir/'figure1_corrected_architecture.png',dpi=300,bbox_inches='tight');plt.close(fig)
# Fig2 workflow
fig,ax=plt.subplots(figsize=(11,9));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
box(ax,.08,.81,.32,.11,'Delayed local E2 telemetry\n$\\tilde{Q},\\tilde{A},\\tilde{\\eta}$',17)
box(ax,.60,.81,.30,.11,'Delayed neighbor summaries\nfrom RIC shared context',16)
box(ax,.20,.60,.60,.12,'1. Build delayed features and compute\ngraph-attention weights $\\alpha$',17)
box(ax,.20,.40,.60,.12,'2. Predict stale-state residual and form\ncorrected queue $\\hat{Q}$',17)
box(ax,.20,.20,.60,.12,'3. Compute Lyapunov PRB shares\nand enforce exact normalization',17)
box(ax,.12,.035,.76,.085,'Publish a new scalar pressure summary for future slots.\nNo current neighbor queue is directly exposed.',13)
arrow(ax,(.24,.81),(.39,.72));arrow(ax,(.76,.81),(.61,.72));arrow(ax,(.50,.60),(.50,.52));arrow(ax,(.50,.40),(.50,.32));arrow(ax,(.50,.20),(.50,.12))
fig.savefig(figdir/'figure2_corrected_workflow.png',dpi=300,bbox_inches='tight');plt.close(fig)
# Fig3 latency stress with CI
s=pd.read_csv(data/'summary_main.csv');sel=s[s.method.isin(['Local-Lyapunov','D-GALS','PPO'])]
fig,ax=plt.subplots(figsize=(8,5.5))
for meth,g in sel.groupby('method'):
 g=g.sort_values('delay_ms'); y=g.sla_violation_rate_mean.values*100; e=g.sla_violation_rate_ci95.values*100; ax.errorbar(g.delay_ms,y,yerr=e,marker='o',capsize=3,label=meth)
ax.set_xlabel('Mean E2 delay (ms)');ax.set_ylabel('SLA violation rate (%)');ax.set_title('Latency-stress evaluation (mean ± 95% CI, 8 seeds)');ax.grid(True,alpha=.25);ax.legend();fig.tight_layout();fig.savefig(figdir/'figure3_latency_stress.png',dpi=300);plt.close(fig)
# Fig4 residual predictor sensitivity
r=pd.read_csv(data/'residual_predictor_sensitivity.csv');fig,ax=plt.subplots(figsize=(7.6,5.2));ax.plot(r.traffic_corr,r.graph_kappa,marker='o',label='Graph residual predictor');ax.plot(r.traffic_corr,r.local_kappa,marker='s',label='Local-only residual MLP');ax.axhline(1,ls='--',lw=1);ax.set_xlabel('Traffic correlation parameter');ax.set_ylabel(r'Residual MSE ratio $\kappa$');ax.set_title('Residual-prediction sensitivity');ax.set_ylim(.65,1.02);ax.grid(True,alpha=.25);ax.legend();fig.tight_layout();fig.savefig(figdir/'figure4_predictor_sensitivity.png',dpi=300);plt.close(fig)
# Fig5 scalability
sc=pd.read_csv(data/'scalability_summary.csv');fig,ax=plt.subplots(figsize=(7.6,5.2));ax.errorbar(sc.cells,sc.decision_time_total_ms_mean,yerr=sc.decision_time_total_ms_sd,marker='o',capsize=3);ax.axhline(10,ls='--',lw=1,label='10 ms decision slot');ax.set_xlabel('Number of cells');ax.set_ylabel('Total D-GALS compute time per slot (ms)');ax.set_title('Controller compute-time scaling on CPU');ax.set_xticks(sc.cells);ax.grid(True,alpha=.25);ax.legend();fig.tight_layout();fig.savefig(figdir/'figure5_scalability_runtime.png',dpi=300);plt.close(fig)
print('figures written to',figdir)
