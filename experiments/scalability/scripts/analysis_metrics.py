"""Use the published eval_utils.py majority-label rule, including categorical ties."""
from pathlib import Path
import numpy as np,pandas as pd,h5py
from sklearn.metrics import balanced_accuracy_score
def load_annotations(run, label_key='celltype'):
 files=list((Path(run)/'save').glob('*_ids.h5ad'))
 if len(files)!=1:raise RuntimeError('Expected one saved assignment H5AD in '+str(run))
 def decode(values):return [v.decode('utf-8') if isinstance(v,bytes) else str(v) for v in values]
 with h5py.File(files[0]) as f:
  g=f['obs'][label_key]
  if isinstance(g,h5py.Group) and 'categories' in g:
   cats=decode(g['categories'][:]);codes=g['codes'][:]
   annotations=pd.Series(pd.Categorical.from_codes(codes,categories=cats,ordered=bool(g.attrs.get('ordered',False))))
  else:annotations=pd.Series(decode(g[:]))
 if annotations.isna().any():raise RuntimeError('Missing cell-type annotations')
 return annotations
def majority_statistics(ids,annotations):
 labels=annotations.astype(str).to_numpy();pred=np.empty(len(ids),object);sizes=[];purity=[];ties=0
 for group in np.unique(ids):
  mask=ids==group
  counts=annotations[mask].value_counts()
  majority=str(counts.idxmax())  # Exact original eval_utils.py rule.
  pred[mask]=majority;sizes.append(int(mask.sum()))
  purity.append(float(np.mean(labels[mask]==majority)))
  ties+=int(np.sum(counts.to_numpy()==counts.max())>1)
 sizes=np.asarray(sizes)
 result={'balanced_purity':float(balanced_accuracy_score(labels,pred)),'cell_weighted_purity':float(np.mean(labels==pred)),'mean_metacell_purity':float(np.mean(purity)),'K_actual':len(sizes),'size_min':int(sizes.min()),'size_median':float(np.median(sizes)),'size_mean':float(sizes.mean()),'size_max':int(sizes.max())}
 return result,pred,{'metacell':np.unique(ids),'size':sizes,'purity':purity},ties
def refresh_native_quality(run):
 import json
 run=Path(run)
 if not (run/'completed.json').exists():return
 cfg=json.loads((run/'effective_config.json').read_text())
 ann=load_annotations(run,cfg['label_key']);ids=np.load(run/'assignments.npy')
 current=json.loads((run/'quality.json').read_text())
 if not (run/'quality_before_tie_rule_alignment.json').exists():
  (run/'quality_before_tie_rule_alignment.json').write_text(json.dumps(current,indent=2))
 quality,pred,sizes,ties=majority_statistics(ids,ann);current.update(quality)
 (run/'quality.json').write_text(json.dumps(current,indent=2))
 done=json.loads((run/'completed.json').read_text());done['quality']=current
 (run/'completed.json').write_text(json.dumps(done,indent=2))
 pd.DataFrame(sizes).to_csv(run/'metacell_sizes.csv',index=False)
 labels=ann.astype(str).to_numpy()
 pd.DataFrame([{'cell_type':v,'n_cells':int(np.sum(labels==v)),'majority_recall':float(np.mean(pred[labels==v]==v))} for v in np.unique(labels)]).to_csv(run/'cell_type_quality.csv',index=False)
 (run/'quality_definition.json').write_text(json.dumps({'majority_rule':'annotations[mask].value_counts().idxmax(), preserving saved categorical annotation order; matches source_server/eval_utils.py','tied_majority_metacells':ties,'model_and_assignments_changed':False},indent=2))
