"""Re-evaluate every archived frozen-inference assignment; no model inference or training."""
from pathlib import Path
import io,json,csv,hashlib,zipfile,itertools,sys
import numpy as np,pandas as pd
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,f1_score,balanced_accuracy_score,confusion_matrix
from release_paths import output_root, env_path
ARCHIVE=env_path('GARQ_FROZEN_ARCHIVE')
OUT=output_root()/'R79/frozen_rechecked';OUT.mkdir(parents=True,exist_ok=True)
probe=json.loads(json.loads(env_path('GARQ_FROZEN_IDENTITY_PROBE').read_text())['stdout'])
checks=[];sources=[];qualities=[];transitions=[];populations=[];confusions=[]
def ck(name,b):
 checks.append({'check':name,'pass':bool(b)})
 if not b:raise AssertionError(name)
def digest(b):return hashlib.sha256(b).hexdigest()
def diglist(x):return digest(('\n'.join(x)+'\n').encode())
def near(a,b):return np.isclose(float(a),float(b),rtol=1e-10,atol=1e-12)
def read(z,p):
 b=z.read(p);sources.append({'archive':str(ARCHIVE),'entry':p,'bytes':len(b),'sha256':digest(b)});return b
def quality(ids,ann):
 labels=ann.astype(str).to_numpy();pred=np.empty(len(ids),object);groups=[];ties=0
 order=np.argsort(ids,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(ids[order]))+1,len(ids)]
 for a,b in zip(starts[:-1],starts[1:]):
  ix=order[a:b];counts=ann.iloc[ix].value_counts();major=str(counts.idxmax());pred[ix]=major;groups.append(float(np.mean(labels[ix]==major)));ties+=int((counts==counts.max()).sum()>1)
 classes=list(dict.fromkeys(labels));cm=confusion_matrix(labels,pred,labels=classes);den=cm.sum(1)+cm.sum(0);fs=np.divide(2*np.diag(cm),den,out=np.zeros(len(classes),float),where=den!=0)
 ba=np.mean(np.diag(cm)/cm.sum(1));mf=float(fs.mean())
 ck('independent F1',near(mf,f1_score(labels,pred,labels=classes,average='macro',zero_division=0)))
 ck('independent BA',near(ba,balanced_accuracy_score(labels,pred)))
 return pred,dict(balanced_accuracy=float(ba),macro_F1=mf,cell_weighted_correctness=float(np.mean(labels==pred)),mean_metacell_purity=float(np.mean(groups)),actual_K=len(groups),tie_groups=ties),classes,cm,fs
with zipfile.ZipFile(ARCHIVE) as z:
 for src in ['scripts/frozen_stability.py','scripts/analysis_metrics.py','source_server/model.py']:
  b=read(z,src);dest=OUT/'source'/src;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
 for row in probe['rows']:
  run=row['run'];d=row['dataset'];base='runs/'+run+'/'
  df=pd.read_csv(io.BytesIO(read(z,base+'cell_index.csv')),dtype={'cell_id':str,'cell_type':str});n=len(df);labels=df.cell_type.to_numpy();ids=df.cell_id.tolist()
  for inp in row['inputs']:
   ck(d+' raw ordered cells',n==inp['n'] and diglist(ids)==inp['cell_id_sha256'] and df.cell_id.is_unique)
   ck(d+' raw ordered labels',diglist(df.cell_type.tolist())==inp['labels_sha256'])
  meta=row['inputs'][0];ann=pd.Series(pd.Categorical(labels,categories=meta['categories'],ordered=meta['ordered'])) if meta['categories'] is not None else pd.Series(labels)
  cfg=json.loads(read(z,base+'effective_config.json'));done=json.loads(read(z,base+'frozen_stability/completed.json'));repro=json.loads(read(z,base+'frozen_stability/baseline_reproduction.json'))
  ck(d+' frozen completion',done['status']=='complete' and done['tests']==44 and done['anchors_unchanged'] and done['baseline_mismatch']==0 and repro['changed_fraction_vs_saved_inference']==0)
  ck(d+' paper dataset identity',cfg['paper_table1_dataset_id']==d)
  for fname in ['effective_config.json','frozen_stability/completed.json','frozen_stability/baseline_reproduction.json','encoder_independence.json']:
   dst=OUT/'audit'/run/fname;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(read(z,base+fname))
  for fn in ['checkpoint.pt','embeddings.npy']:
   h=hashlib.sha256()
   with z.open(base+fn) as f:
    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
   sources.append({'archive':str(ARCHIVE),'entry':base+fn,'bytes':z.getinfo(base+fn).file_size,'sha256':h.hexdigest(),'not_extracted':True})
  baseline=np.load(io.BytesIO(read(z,base+'assignments.npy')),allow_pickle=False)
  ck(d+' baseline shape/range',baseline.shape==(n,) and (baseline>=0).all() and (baseline<cfg['K']).all())
  bp,bq,cls,bcm,bf=quality(baseline,ann)
  old=pd.read_csv(io.BytesIO(read(z,base+'frozen_stability/stability.csv')),dtype={'order_seed':str});oldpop=pd.read_csv(io.BytesIO(read(z,base+'frozen_stability/cell_type_stability.csv')),dtype={'order_seed':str,'cell_type':str})
  expected={(B,str(s) if s is not None else 'canonical') for B in [512,1024,2048,4096] for s in [None]+list(range(1701,1711))}
  ck(d+' all fixed conditions',set(zip(old.batch_size,old.order_seed))==expected and len(old)==44)
  for B,tag in sorted(expected):
   arr=np.load(io.BytesIO(read(z,base+f'frozen_stability/assignments_B{B}_{tag}.npy')),allow_pickle=False)
   ck(d+f' B{B} {tag} shape/range',arr.shape==(n,) and (arr>=0).all() and (arr<cfg['K']).all())
   if B==2048 and tag=='canonical':ck(d+' canonical exact baseline',np.array_equal(arr,baseline))
   pred,q,classes,cm,fs=quality(arr,ann);o=old[(old.batch_size==B)&(old.order_seed==tag)].iloc[0]
   prefix=dict(dataset=d,run_id=run,inference_batch=B,order_seed=tag,n_cells=n,requested_K=cfg['K'])
   ari=float(adjusted_rand_score(baseline,arr));nmi=float(normalized_mutual_info_score(baseline,arr));chg=arr!=baseline;lc=pred!=bp;ca=bp==labels;cb=pred==labels
   for key,val in [('ARI_assignment',ari),('NMI_assignment',nmi),('changed_fraction',float(chg.mean())),('balanced_purity',q['balanced_accuracy']),('cell_weighted_purity',q['cell_weighted_correctness']),('mean_metacell_purity',q['mean_metacell_purity']),('K_actual',q['actual_K'])]:ck(d+f' B{B} {tag} historical '+key,near(o[key],val))
   qualities.append({**prefix,**q,'ARI_vs_canonical_B2048':ari,'NMI_vs_canonical_B2048':nmi})
   def trans(mask):
    return dict(n_cells=int(mask.sum()),anchor_changed_fraction=float(chg[mask].mean()),dominant_label_agreement=float((~lc)[mask].mean()),anchor_changed_label_unchanged_fraction=float((chg&~lc)[mask].mean()),correct_both=int((ca&cb&mask).sum()),correct_baseline_wrong_test=int((ca&~cb&mask).sum()),wrong_baseline_correct_test=int((~ca&cb&mask).sum()),wrong_both=int((~ca&~cb&mask).sum()))
   transitions.append({**prefix,**trans(np.ones(n,bool))})
   for j,label in enumerate(classes):
    mask=labels==label;rec=float(np.mean(pred[mask]==label));pr=oldpop[(oldpop.batch_size==B)&(oldpop.order_seed==tag)&(oldpop.cell_type==label)]
    ck(d+f' B{B} {tag} population '+label,len(pr)==1 and near(pr.iloc[0].majority_recall,rec) and near(pr.iloc[0].changed_fraction,chg[mask].mean()) and int(pr.iloc[0].n_cells)==int(mask.sum()))
    populations.append({**prefix,'cell_type':label,'original_fraction':float(mask.mean()),'rare_le_1pct':int(mask.sum())*100<=n,'recall':rec,'F1':float(fs[j]),**trans(mask)})
   for a,b in zip(*np.nonzero(cm)):
    confusions.append({**prefix,'true_label':classes[a],'inherited_label':classes[b],'n_cells':int(cm[a,b])})
  print('VERIFIED',d,44,flush=True)
for name,rows in [('frozen_quality_all_conditions',qualities),('frozen_transitions_all_conditions',transitions),('frozen_population_all_conditions',populations),('frozen_confusion_all_conditions',confusions)]:pd.DataFrame(rows).to_csv(OUT/(name+'.tsv'),sep='\t',index=False)
# Equal test weighting within each dataset and batch; canonical kept separate.
p=pd.DataFrame(transitions);q=pd.DataFrame(qualities);m=p.merge(q[['dataset','inference_batch','order_seed','balanced_accuracy','macro_F1','ARI_vs_canonical_B2048']],on=['dataset','inference_batch','order_seed'])
num=['anchor_changed_fraction','dominant_label_agreement','anchor_changed_label_unchanged_fraction','balanced_accuracy','macro_F1','ARI_vs_canonical_B2048']
m[m.order_seed!='canonical'].groupby(['dataset','inference_batch'])[num].agg(['mean','std','min','max']).to_csv(OUT/'frozen_permutation_summary.tsv',sep='\t')
report={'status':'PASS','datasets':2,'conditions':len(qualities),'population_rows':len(populations),'checks':len(checks),'failed':sum(not x['pass'] for x in checks),'new_training':0,'new_inference':0,'new_assignments':0,'sources':sources,'tests':checks,'versions':{'python':sys.version,'pandas':pd.__version__,'numpy':np.__version__},'frozen_state_evidence':'Original source asserts anchors unchanged; original completion and zero-mismatch records checked. Existing checkpoint/embedding bytes hashed, not re-run. Canonical saved array exactly matches baseline.'}
(OUT/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:report[k] for k in ['status','datasets','conditions','population_rows','checks','failed']}))
