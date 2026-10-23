from pathlib import Path
import pandas as pd,numpy as np,shutil,json,hashlib,datetime
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from release_paths import output_root, env_path
O=output_root();(O/'A').mkdir(exist_ok=True);(O/'figures').mkdir(exist_ok=True)
S=env_path('GARQ_R79_RESULTS')
plt.rcParams.update({'font.family':'Arial','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
paths=['tables/E1A_v2','tables/E1B_v2','tables/frozen_rechecked','tables/recheck_summaries']
records=[]
for p in paths:
 for f in (S/p).glob('*'):
  if f.is_file():
   dst=O/'A'/p;dst.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst/f.name)
   records.append({'source':str(f),'relative_output':str((dst/f.name).relative_to(O)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'status':'REUSED_VERIFIED'})
pd.DataFrame(records).to_csv(O/'A/REUSED_EVIDENCE_MAP.tsv',sep='\t',index=False)
q=pd.read_csv(S/'tables/E1A_v2/per_run_quality.tsv',sep='\t')
p=pd.read_csv(S/'tables/E1A_v2/pairwise_stability.tsv',sep='\t')
assert len(q)==36 and len(p)==36
fig,axs=plt.subplots(1,4,figsize=(13.2,3.3),layout='constrained')
for ax,(df,col,title) in zip(axs,[(q,'balanced_accuracy','a  Balanced accuracy'),(q,'macro_F1','b  Macro-F1'),(p,'ARI_assignment_shared_assigned','c  Assignment ARI'),(p,'dominant_label_agreement_full_cohort','d  Inherited-label agreement')]):
 for i in range(1,13):
  vals=df.loc[df.dataset==f'D{i}',col].to_numpy()
  ax.scatter(np.full(len(vals),i)+np.linspace(-.16,.16,len(vals)),vals,s=16,color='#316B83' if i<=6 else '#A66C54',alpha=.75)
  ax.plot([i-.22,i+.22],[vals.mean()]*2,color='black',lw=1.5)
 ax.set(title=title,xticks=range(1,13),xticklabels=[f'D{i}' for i in range(1,13)])
 ax.tick_params(axis='x',rotation=60);ax.set_ylim(0,1);ax.set_ylabel('Score')
fig.legend(handles=[Line2D([],[],marker='o',linestyle='none',color='#316B83',label='RNA + ADT (D1–D6)'),Line2D([],[],marker='o',linestyle='none',color='#A66C54',label='RNA + ATAC (D7–D12)'),Line2D([],[],color='black',label='Mean')],loc='lower center',bbox_to_anchor=(.5,-.075),ncol=3,frameon=False)
fig.savefig(O/'figures/A1_training_results.pdf',bbox_inches='tight');fig.savefig(O/'figures/A1_training_results.png',dpi=180,bbox_inches='tight');plt.close(fig)
t=pd.read_csv(S/'tables/frozen_rechecked/frozen_transitions_all_conditions.tsv',sep='\t')
t=t[pd.to_numeric(t.order_seed,errors='coerce').notna() & (pd.to_numeric(t.order_seed,errors='coerce')>0)].copy()
t['label_changed']=1-t.dominant_label_agreement
for col in ['correct_baseline_wrong_test','wrong_baseline_correct_test']:t[col+'_fraction']=t[col]/t.n_cells
fig,axs=plt.subplots(2,3,figsize=(10.8,5.8),layout='constrained')
for row,ds in enumerate(['D13','D16']):
 d=t[t.dataset==ds];batches=sorted(d.inference_batch.unique())
 for ax,col,title in zip(axs[row],['anchor_changed_fraction','label_changed','correct_baseline_wrong_test_fraction'],['Anchor changes','Inherited-label changes','Correctness transitions']):
  for j,b in enumerate(batches):
   v=d[d.inference_batch==b]
   cols=[col] if title!='Correctness transitions' else ['correct_baseline_wrong_test_fraction','wrong_baseline_correct_test_fraction']
   for k,c in enumerate(cols):
    x=j+(.12 if k else -.12) if len(cols)>1 else j
    a=100*v[c].to_numpy();ax.scatter(np.full(len(a),x)+np.linspace(-.055,.055,len(a)),a,s=12,color=['#A66C54','#316B83'][k],alpha=.6)
    ax.plot([x-.08,x+.08],[a.mean()]*2,color='black',lw=1.4,label=('Correct to wrong' if k==0 else 'Wrong to correct') if j==0 and len(cols)>1 else None)
  ax.set(title=f'{ds}: {title}',xticks=range(len(batches)),xticklabels=batches,xlabel='Inference batch',ylabel='Cells (%)')
  if title=='Correctness transitions':ax.legend(handles=[Line2D([],[],marker='o',linestyle='none',color='#A66C54',label='Correct to wrong'),Line2D([],[],marker='o',linestyle='none',color='#316B83',label='Wrong to correct')],frameon=False,fontsize=7)
fig.savefig(O/'figures/A2_frozen_inference.pdf');fig.savefig(O/'figures/A2_frozen_inference.png',dpi=180);plt.close(fig)
t.to_csv(O/'A/frozen_plot_data.tsv',sep='\t',index=False)
(O/'A/STATUS.json').write_text(json.dumps({'status':'NEW_PRESENTATION_FROM_VERIFIED_TABLES','new_evaluation':False,'new_fits':0,'runs':36,'pairs':36,'figures':2},indent=2))
print('A_COMPLETE_REUSED_ONLY')
