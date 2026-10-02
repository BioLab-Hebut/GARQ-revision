#!/usr/bin/env python
"""Create tables, editable figures and a numeric reply only from completed server runs."""
import sys,json,hashlib,zipfile
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
 parts=[]
 for r in rows:
  parts.append(f"{display[r['dataset']]} ({r['cells']:,} cells; $K={r['K_requested']}$) completed in {r['profiled_process_wall_min']:.1f} min, with peak main-process CPU RSS of {r['main_CPU_peak_GiB']:.1f} GiB, process-tree PSS of {r['tree_PSS_peak_GiB']:.1f} GiB, and PyTorch allocated/reserved GPU memory of {r['PyTorch_allocated_peak_GiB']:.2f}/{r['PyTorch_reserved_peak_GiB']:.2f} GiB")
 frozen_text=[]
 for ds in metrics.dataset:
  d=P[(P.dataset==ds)&(P.batch_size==2048)]
  frozen_text.append(f"{display[ds]}: assignment ARI {d.ARI_assignment.mean():.3f} (range {d.ARI_assignment.min():.3f}--{d.ARI_assignment.max():.3f})")
 training_text=[]
 for ds in T.dataset.unique():
  d=T[T.dataset==ds].groupby('train_batch_size').balanced_purity.mean()
  a=C[C.dataset==ds].ARI_assignment
  training_text.append(f"{display[ds]}: mean balanced cell-type purity {d.min():.3f}--{d.max():.3f} across training batch sizes; assignment ARI {a.min():.3f}--{a.max():.3f} relative to the same-seed $B=512$ model")
 reply=r"""\noindent\textbf{Response.}
We thank the Reviewer for identifying this discrepancy. We audited the manuscript implementation against the public repository (commit \texttt{5da45adcd62f}); the model, training and preprocessing files are identical. The additional command-line aliases and evaluation-cache naming differences are documented in the reproducibility materials.

We agree that the original description of attention across cells was inaccurate. Each cell enters the encoder as a separate sequence of length one, with tensor shape $B\times1\times d_e$, so the encoder does not construct an $N\times N$ cell-attention matrix. Intercellular information is incorporated through graph-aware quantization. We have corrected the encoder equations and specified that the KNN graph is rebuilt within each training or inference batch. Equation~(9) now uses batch-local indices and the implemented symmetrized adjacency and residual coefficient. These corrections describe the implementation used for the experiments; they do not introduce a different model.

Sparse matrices are converted to dense arrays on the CPU before feature selection, whereas GPU computation uses mini-batches. We now explicitly distinguish host-memory requirements from GPU working memory. We repeated the two large-data analyses using the confirmed metacell targets and original code defaults ($300$ maximum epochs, seed~1, training batch size~512 and inference batch size~2048), including preprocessing, initialization, training, inference, aggregation and output saving. """
 reply+='; '.join(parts)+'. '
 reply+=r"""These measurements include the saved checkpoint and separately timed validation overhead; optional post-hoc visualization and pairwise-correlation plotting are excluded. The stage-wise measurements, hardware details, actual metacell counts and separate CPU/GPU memory definitions are provided in Supplementary Fig.~\ref{fig:scalability_profile} and Table~\ref{tab:scalability_resources}. The server has two Intel Xeon Platinum 8336C CPUs (64 logical CPUs), 503.5 GiB host RAM, and RTX 4090 GPUs; each resource-profile run used one GPU with Python 3.11.6, PyTorch 2.1.1 (CUDA 12.1) and Scanpy 1.9.6. The results document feasibility on a 24-GB GPU while also making the substantial CPU preprocessing requirement explicit.

We additionally tested the frozen graph-aware quantizer at inference batch sizes 512, 1024, 2048 and 4096 using the original cell order and ten predefined permutations, restoring assignments to the same cell identities before comparison. At the original inference batch size, """
 reply+='; '.join(frozen_text)+'. '
 reply+=r"""We also repeated training on D5 and D11 at batch sizes 256, 512 and 1024 with three random seeds, evaluating each trained model at a common inference batch size of 2048. """
 reply+='; '.join(training_text)+'. '
 reply+=r"""All settings and per-run results are reported in Supplementary Figs.~\ref{fig:frozen_stability} and \ref{fig:training_batch_sensitivity}. These results show appreciable sensitivity of individual metacell memberships to both inference batch composition and the training batch setting; cell-type purity also varies, particularly across training settings. We therefore distinguish cell-type preservation from membership stability and explicitly state that the current implementation is batch-sensitive.

The revised computational analysis states the actual dense batch-graph costs, including $O(B^2d)$ cell similarities and $O(B^2K)$ graph--score multiplication, rather than global cell attention. The biological interpretation continues to rest on GARQ's multimodal reconstruction, shared anchors and graph-guided aggregation, while the scalability claim is now limited to practical mini-batch GPU execution under the documented host-memory requirements. We provide the exact code snapshot, configurations, checkpoints, monitoring and analysis scripts, and complete result tables in the reproducibility package.

\medskip
\noindent\textbf{Changes in the revised manuscript.}
We corrected the encoder and quantization descriptions in Methods (original pp.~4--5, Eqs.~2--4 and 9), expanded preprocessing and implementation details (original p.~3, lines~96--107), and qualified the runtime/memory statements in Results (original p.~12, lines~338--339, and p.~15, lines~393--395). We added the computational profile and batch-sensitivity analyses to the Supplementary Information.
"""
 (OUT/'Reply_Letter.tex').write_text(reply)
 (OUT/'Revised_Results_Numerical.tex').write_text(r"The additional implementation audit and large-data profiling confirmed that GARQ performs metacell construction with mini-batch GPU computation, without an all-cell attention matrix. "+'; '.join(parts)+r". Dense preprocessing dominated host-memory requirements. Batch-sensitivity analyses quantify changes in metacell membership separately from cell-type purity (Supplementary Figs.~\ref{fig:frozen_stability} and \ref{fig:training_batch_sensitivity}); the current batch-local graph implementation is not claimed to be invariant to batch composition.")
 import subprocess
 for _ in range(2):
  done_compile=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-no-shell-escape','Reply_and_Supplement_Standalone.tex'],cwd=OUT,capture_output=True,text=True)
  (OUT/'latex_compile.log').write_text(done_compile.stdout+'\n'+done_compile.stderr)
  if done_compile.returncode!=0:raise RuntimeError('Reply PDF compilation failed; see latex_compile.log')
 for _ in range(2):
  method_compile=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-no-shell-escape','Methods_Replacements_Standalone.tex'],cwd=OUT,capture_output=True,text=True)
  (OUT/'methods_latex_compile.log').write_text(method_compile.stdout+'\n'+method_compile.stderr)
  if method_compile.returncode!=0:raise RuntimeError('Methods PDF compilation failed; see methods_latex_compile.log')
 qa=OUT/'qa';qa.mkdir(exist_ok=True)
 subprocess.run(['pdftoppm','-scale-to','1400','-png',str(OUT/'Methods_Replacements_Standalone.pdf'),str(qa/'methods')],check=True,capture_output=True)
 subprocess.run(['pdftoppm','-scale-to','1400','-png',str(OUT/'Reply_and_Supplement_Standalone.pdf'),str(qa/'reply')],check=True,capture_output=True)
 (OUT/'ALL_EXPERIMENTS_COMPLETE.json').write_text(json.dumps({'status':'complete','large_runs':2,'training_sensitivity_runs':18,'frozen_inference_tests':len(F),'review_status':'Numerical outputs generated from completed runs. Inspect figures and wording before manuscript submission.'},indent=2))
else:
 (OUT/'EXPERIMENTS_IN_PROGRESS.json').write_text(json.dumps({'resource_runs_complete':2,'frozen_stability_complete':complete_stability,'training_sensitivity_complete':complete_training,'reply_status':'Final numeric reply is generated only after all prespecified experiments complete.'},indent=2))
readme=(ROOT/'README_ZH.md').read_text()
if complete_stability and complete_training:
 readme=readme.replace('正式实验正在运行。请先查看 pipeline_state.json 和 queue_state.json；不要将“已启动”解释为“已完成”。','全部预设实验及数值图表已完成；请查看 ALL_EXPERIMENTS_COMPLETE.json 和质量检查记录。')
 readme=readme.replace('在所有预设实验完成前，不生成包含最终数值结论的 Reply_Letter.tex。','Reply_Letter.tex 已根据全部完成的预设实验生成。')
(OUT/'README_ZH.md').write_text(readme)
(ROOT/'audit/final_code_configuration_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['scripts','source_server','configs'] for p in (ROOT/folder).glob('*') if p.is_file()},indent=2))
files=[p for p in OUT.iterdir() if p.is_file() and p.name!='SHA256.json']
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(OUT/'SHA256.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(ROOT/'GARQ_Scalability_Revision_Server_Package.zip','w',zipfile.ZIP_DEFLATED) as z:
 for folder in ['deliverables','scripts','configs','audit','source_server']:
  for p in (ROOT/folder).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and 'smoke_inputs' not in p.parts and 'qa' not in p.parts:z.write(p,str(p.relative_to(ROOT)))
 for name in names+[json.loads((ROOT/'configs'/c).read_text())['run_id'] for c in json.loads((ROOT/'configs/training_sensitivity_queue.json').read_text())]:
  p=ROOT/'runs'/name
  if not p.exists():continue
  for f in p.rglob('*'):
   if f.is_file() and f.suffix in ['.json','.csv','.log','.pt','.npy'] and not f.name=='console.log':z.write(f,str(f.relative_to(ROOT)))
print('DELIVERABLES_UPDATED','all_complete',complete_stability and complete_training)
