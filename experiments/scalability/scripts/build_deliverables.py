#!/usr/bin/env python
"""Create numerical tables and editable figures from completed runs."""
import json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis_metrics import refresh_native_quality
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
names=['Ma_RNA_ATAC_K644_B512_seed1','GSE164378_RNA_ADT_K500_B512_seed1']
display={'Ma_GSE140203':'Ma (RNA + ATAC)','GSE164378':'GSE164378 (RNA + ADT)','D5_RNA_ADT':'D5 (RNA + ADT)','D11_RNA_ATAC':'D11 (RNA + ATAC)'}
rows=[];stage_rows=[];frozen=[]
for name in names:
 p=ROOT/'runs'/name
 assert (p/'completed.json').exists(),name+' incomplete'
 assert (p/'resource_summary.json').exists(),name+' monitoring incomplete'
 refresh_native_quality(p)
 cfg=json.loads((p/'effective_config.json').read_text());done=json.loads((p/'completed.json').read_text());res=json.loads((p/'resource_summary.json').read_text());st=pd.read_csv(p/'stages.csv')
 dataset=cfg['dataset'];gpu_alloc=int(st.peak_cuda_allocated_bytes.max());gpu_reserved=int(st.peak_cuda_reserved_bytes.max())
 row={'dataset':dataset,'cells':done['quality']['n_cells'],'K_requested':cfg['K'],'K_actual':done['quality']['K_actual'],'train_batch_size':cfg['train_batch_size'],'eval_batch_size':cfg['eval_batch_size'],'epochs_completed':int(st.phase.isin(['warm_one_epoch','train_one_epoch']).sum()),'seed':cfg['seed'],'original_features':' / '.join(f'{x["raw_shape"][1]:,}' for x in cfg['input_metadata']),'selected_features':' / '.join(f'{x["selected_shape"][1]:,}' for x in cfg['input_metadata']),'raw_dtype':' / '.join(x['raw_dtype'] for x in cfg['input_metadata']),'batch_sizes':f"{cfg['train_batch_size']} / {cfg['eval_batch_size']}",'profiled_process_wall_min':res['process_wall_seconds']/60,'core_pipeline_min':done['core_pipeline_seconds']/60,'main_CPU_peak_GiB':max(res['peaks']['main_rss_bytes'],done['main_ru_maxrss_bytes'])/1024**3,'tree_PSS_peak_GiB':res['peaks']['tree_pss_sum_bytes']/1024**3,'PyTorch_allocated_peak_GiB':gpu_alloc/1024**3,'PyTorch_reserved_peak_GiB':gpu_reserved/1024**3,'process_GPU_peak_GiB':res['peaks']['process_gpu_bytes']/1024**3,**{k:v for k,v in done['quality'].items() if k not in ['K_requested','K_actual']}}
 rows.append(row)
 stage_names={'Read and preprocess':['load_and_preprocess'],'Initialize':['anchor_initialization'],'Train':['warm_one_epoch','train_one_epoch'],'Infer':['inference'],'Aggregate':['aggregation'],'Write H5AD':['write_h5ad']}
 assigned=0
 for group,phases in stage_names.items():
  seconds=float(st.loc[st.phase.isin(phases),'seconds'].sum());assigned+=seconds
  stage_rows.append({'dataset':dataset,'stage':group,'seconds':seconds})
 stage_rows.append({'dataset':dataset,'stage':'Checkpoint and other','seconds':done['core_pipeline_seconds']-assigned})
 graph=pd.read_csv(p/'graph_timings.csv')
 for phase,g in graph.groupby('phase'):
  stage_rows.append({'dataset':dataset,'stage':'Graph GPU subset: '+phase,'seconds':g.gpu_ms.sum()/1000})
 if (p/'frozen_stability/completed.json').exists():
  f=pd.read_csv(p/'frozen_stability/stability.csv',dtype={'order_seed':str});frozen.append(f)
metrics=pd.DataFrame(rows);stages=pd.DataFrame(stage_rows)
metrics.to_csv(OUT/'Large_Data_Resources_and_Quality.csv',index=False);stages.to_csv(OUT/'Stage_Times_and_Graph_Subsets.csv',index=False)
def save(fig,name):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
 fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
 fig.savefig(OUT/(name+'.png'),dpi=240,bbox_inches='tight')
 plt.close(fig)
fig=plt.figure(figsize=(11,6));gs=fig.add_gridspec(2,2,width_ratios=[1.2,1])
ax=fig.add_subplot(gs[:,0]);small=stages[~stages.stage.str.startswith('Graph')]
bottom=np.zeros(2)
for label,color in zip(small.stage.unique(),['#8da0cb','#66c2a5','#b65d71','#fc8d62','#a6d854','#e5c494','#b3b3b3']):
 vals=np.array([small[(small.dataset==ds)&(small.stage==label)].seconds.sum()/60 for ds in metrics.dataset])
 ax.bar(np.arange(2),vals,bottom=bottom,label=label,color=color,width=.6);bottom+=vals
ax.set_ylim(0,max(bottom)*1.5)
ax.set_xticks([0,1],[display[x].replace(' (','\n(') for x in metrics.dataset]);ax.set_ylabel('Core pipeline time (min)');ax.set_title('a  End-to-end metacell construction',loc='left',fontweight='bold');ax.legend(frameon=False,fontsize=8)
for pos,cols,title in [(gs[0,1],['main_CPU_peak_GiB','tree_PSS_peak_GiB'],'b  Peak host memory'),(gs[1,1],['PyTorch_reserved_peak_GiB','process_GPU_peak_GiB'],'c  Peak GPU memory')]:
 a=fig.add_subplot(pos)
 for j,(col,color) in enumerate(zip(cols,['#b65d71','#8da0cb'])):
  a.bar(np.arange(2)+(j-.5)*.3,metrics[col],width=.3,color=color,label=col.replace('_peak_GiB','').replace('_',' '))
 a.set_ylim(0,float(metrics[cols].to_numpy().max())*1.35)
 a.set_xticks([0,1],['Ma','GSE164378']);a.set_ylabel('GiB');a.set_title(title,loc='left',fontweight='bold');a.legend(frameon=False,fontsize=8)
fig.tight_layout();save(fig,'Figure_Scalability_Profile')
metrics.to_latex(OUT/'Large_Data_Resources_and_Quality.tex',index=False,float_format='%.4f',longtable=True,escape=True)
complete_stability=len(frozen)==2
if complete_stability:
 F=pd.concat(frozen,ignore_index=True);F.to_csv(OUT/'Frozen_Inference_Stability_All_Runs.csv',index=False)
 P=F[F.order_seed!='canonical'];summary=P.groupby(['dataset','batch_size']).agg(ARI_mean=('ARI_assignment','mean'),ARI_min=('ARI_assignment','min'),ARI_max=('ARI_assignment','max'),ARI_sd=('ARI_assignment','std'),changed_mean=('changed_fraction','mean'),balanced_purity_mean=('balanced_purity','mean'),balanced_purity_min=('balanced_purity','min'),balanced_purity_max=('balanced_purity','max'),K_mean=('K_actual','mean')).reset_index()
 summary.to_csv(OUT/'Frozen_Inference_Stability_Summary.csv',index=False)
 summary.to_latex(OUT/'Frozen_Inference_Stability_Summary.tex',index=False,float_format='%.4f',longtable=True,escape=True)
 frozen_compact=[]
 for (ds,b),g in P.groupby(['dataset','batch_size'],sort=False):
  frozen_compact.append({'Dataset':display[ds].split(' (')[0],'Batch':b,'ARI mean (range)':f"{g.ARI_assignment.mean():.3f} ({g.ARI_assignment.min():.3f}–{g.ARI_assignment.max():.3f})",'Purity mean (range)':f"{g.balanced_purity.mean():.3f} ({g.balanced_purity.min():.3f}–{g.balanced_purity.max():.3f})",'Changed (%)':f"{100*g.changed_fraction.mean():.1f}",'Nonempty K':f"{g.K_actual.min()}–{g.K_actual.max()}"})
 pd.DataFrame(frozen_compact).to_latex(OUT/'Table_Frozen_Stability_Compact.tex',index=False,escape=True)

 fig,axs=plt.subplots(2,2,figsize=(10.5,6.6),sharex=True)
 for i,ds in enumerate(metrics.dataset):
  d=P[P.dataset==ds]
  for j,col in enumerate(['ARI_assignment','balanced_purity']):
   a=axs[j,i];groups=[d[d.batch_size==b][col].to_numpy() for b in [512,1024,2048,4096]]
   a.boxplot(groups,positions=np.arange(4),widths=.5,showfliers=False,patch_artist=True,boxprops={'facecolor':'#dcc1c8'},medianprops={'color':'#963b53'})
   for k,g in enumerate(groups):a.scatter(k+np.linspace(-.10,.10,len(g)),g,s=10,color='#963b53',zorder=3)
   canonical=F[(F.dataset==ds)&(F.order_seed=='canonical')].set_index('batch_size')
   a.plot(range(4),[canonical.loc[b,col] for b in [512,1024,2048,4096]],'s--',color='#466887',ms=4,label='Canonical order')
   a.set_ylim(0,1.03);a.set_ylabel('Assignment ARI' if j==0 else 'Balanced cell-type purity');a.set_title(f'{chr(97+j*2+i)}  {display[ds]}',loc='left',fontweight='bold')
   a.set_xticks(range(4),['512','1024','2048','4096'])
   if j==1:a.set_xlabel('Inference batch size')
   if i==0 and j==0:a.legend(frameon=False,fontsize=8)
 fig.tight_layout();save(fig,'Figure_Frozen_Inference_Stability')
train_file=OUT/'Training_Batch_Quality_Fixed_Evaluation.csv';comp_file=OUT/'Training_Batch_Assignment_Stability.csv'
complete_training=train_file.exists() and comp_file.exists()
if complete_training:
 T=pd.read_csv(train_file);C=pd.read_csv(comp_file)
 assert len(T)==18 and len(C)==12
 T.groupby(['dataset','train_batch_size']).agg(balanced_purity_mean=('balanced_purity','mean'),balanced_purity_sd=('balanced_purity','std'),K_mean=('K_actual','mean')).reset_index().to_csv(OUT/'Training_Batch_Quality_Summary.csv',index=False)
 fig,axs=plt.subplots(2,2,figsize=(10,6.8))
 for i,ds in enumerate(T.dataset.unique()):
  a=axs[0,i];d=C[C.dataset==ds]
  for seed,col in zip([1,2,3],['#963b53','#466887','#719d72']):
   ss=d[d.seed==seed].sort_values('train_batch_size');a.plot(np.arange(2),ss.ARI_assignment,'o-',color=col,label=f'Seed {seed}',ms=4)
  a.set_xticks(range(2),['256','1024']);a.set_ylim(min(-.02,float(d.ARI_assignment.min())-.02),1.03);a.set_xlabel('Training batch size (reference: 512)');a.set_ylabel('Assignment ARI');a.set_title(f'{chr(97+i)}  {display[ds]}',loc='left',fontweight='bold');a.legend(frameon=False,fontsize=8)
  a=axs[1,i];d=T[T.dataset==ds]
  for seed,col in zip([1,2,3],['#963b53','#466887','#719d72']):
   ss=d[d.seed==seed].sort_values('train_batch_size');a.plot(np.arange(3),ss.balanced_purity,'o-',color=col,label=f'Seed {seed}',ms=4)
  a.set_xticks(range(3),['256','512','1024']);a.set_ylim(0,1.03);a.set_xlabel('Training batch size');a.set_ylabel('Balanced cell-type purity');a.set_title(f'{chr(99+i)}  {display[ds]}',loc='left',fontweight='bold');a.legend(frameon=False,fontsize=8)
 fig.tight_layout();save(fig,'Figure_Training_Batch_Sensitivity')
 training_compact=[]
 for (ds,b),g in T.groupby(['dataset','train_batch_size'],sort=False):
  cg=C[(C.dataset==ds)&(C.train_batch_size==b)]
  ar='Reference' if b==512 else f"{cg.ARI_assignment.mean():.3f} ({cg.ARI_assignment.min():.3f}–{cg.ARI_assignment.max():.3f})"
  training_compact.append({'Dataset':display[ds].split(' (')[0],'Train batch':b,'Purity mean (SD)':f"{g.balanced_purity.mean():.3f} ({g.balanced_purity.std():.3f})",'ARI mean (range)':ar,'Nonempty K':f"{g.K_actual.min()}–{g.K_actual.max()}"})
 pd.DataFrame(training_compact).to_latex(OUT/'Table_Training_Stability_Compact.tex',index=False,escape=True)

# Compact source tables for direct supplementary inclusion.
resource_cols=['dataset','cells','K_requested','K_actual','profiled_process_wall_min','main_CPU_peak_GiB','tree_PSS_peak_GiB','PyTorch_allocated_peak_GiB','PyTorch_reserved_peak_GiB','process_GPU_peak_GiB']
labels_units=[('cells','Cells','count'),('K_requested','Requested metacells','count'),('K_actual','Nonempty metacells','count'),('profiled_process_wall_min','Profiled wall time','min'),('main_CPU_peak_GiB','Main-process CPU RSS','GiB'),('tree_PSS_peak_GiB','Process-tree CPU PSS','GiB'),('PyTorch_allocated_peak_GiB','PyTorch GPU allocated','GiB'),('PyTorch_reserved_peak_GiB','PyTorch GPU reserved','GiB'),('process_GPU_peak_GiB','Process GPU memory','GiB')]
compact=[]
for key,label,unit in labels_units:
 compact.append({'Measure':label,'Unit':unit,'Ma RNA+ATAC':str(int(rows[0][key])) if unit=='count' else f"{rows[0][key]:.2f}",'GSE164378 RNA+ADT':str(int(rows[1][key])) if unit=='count' else f"{rows[1][key]:.2f}"})
for key,label,unit in [('original_features','Original features (RNA / partner)','count'),('selected_features','Selected features (RNA / partner)','count'),('raw_dtype','Original matrix dtype',''),('batch_sizes','Batch sizes (train / infer)','cells')]:
 compact.append({'Measure':label,'Unit':unit,'Ma RNA+ATAC':rows[0][key],'GSE164378 RNA+ADT':rows[1][key]})
pd.DataFrame(compact).to_latex(OUT/'Table_Resources_Compact.tex',index=False,escape=True)
if complete_stability and complete_training:
 pending=OUT/'EXPERIMENTS_IN_PROGRESS.json'
 if pending.exists():pending.unlink()
 (OUT/'ALL_EXPERIMENTS_COMPLETE.json').write_text(json.dumps({'status':'complete','large_runs':2,'training_sensitivity_runs':18,'frozen_inference_tests':len(F),'artifact_status':'Numerical tables and figures generated.'},indent=2))
else:
 complete=OUT/'ALL_EXPERIMENTS_COMPLETE.json'
 if complete.exists():complete.unlink()
 (OUT/'EXPERIMENTS_IN_PROGRESS.json').write_text(json.dumps({'resource_runs_complete':2,'frozen_stability_complete':complete_stability,'training_sensitivity_complete':complete_training,'artifact_status':'Available numerical tables and figures generated; remaining analyses are incomplete.'},indent=2))
print('DELIVERABLES_UPDATED','all_complete',complete_stability and complete_training)
