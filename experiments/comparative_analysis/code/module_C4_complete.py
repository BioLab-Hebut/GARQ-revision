from pathlib import Path
import json,hashlib,pandas as pd,numpy as np,datetime
from release_paths import output_root, load_config
R=output_root();O=R/'C/equal_link_budget';O.mkdir(parents=True,exist_ok=True)
src=load_config('C4_inputs.py')['raw_link_files']
tables={}
for x in src:
 p=Path(x['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
 f=pd.read_csv(p);print(x['arm'],list(f.columns),len(f));tables[x['arm']]=f
(O/'INPUT_MANIFEST_RECHECK.json').write_text(json.dumps({'sources':src,'L':[1000,5000,10000],'rule':'descending abs(Pearson r), then peak/gene lexical; reference11885; no result-dependent ranking','locked_at':datetime.datetime.now().astimezone().isoformat()},indent=2))

reference=tables['SingleCell'].set_index('pair_id')
assert reference.index.is_unique and len(reference)==11885
summ=[];retained=[]
for arm,f in tables.items():
 if arm=='SingleCell':continue
 assert f.pair_id.is_unique and np.isfinite(f.score).all()
 ranked=f.assign(abs_score=f.score.abs()).sort_values(['abs_score','peak','gene'],ascending=[False,True,True],kind='mergesort')
 for L in [1000,5000,10000]:
  if len(ranked)<L:
   summ.append(dict(arm=arm,L=L,status='insufficient_candidates'));continue
  top=ranked.head(L);hit=top[top.pair_id.isin(reference.index)]
  same=np.sign(hit.score.to_numpy())==np.sign(reference.loc[hit.pair_id,'score'].to_numpy())
  summ.append(dict(arm=arm,L=L,retained_candidates=len(f),shared_reference_links=len(hit),overlap_per_L=len(hit)/L,reference_coverage=len(hit)/11885,direction_agreement=float(same.mean()) if len(hit) else np.nan,status='defined'))
 top=ranked.head(10000).copy();top['arm']=arm;top['rank']=np.arange(1,len(top)+1);top['reference_hit']=top.pair_id.isin(reference.index)
 retained.append(top[['arm','rank','peak','gene','pair_id','score','reference_hit']])
pd.DataFrame(summ).to_csv(O/'all_methods_all_budgets.tsv',sep='\t',index=False)
pd.concat(retained).to_csv(O/'all_top10000_candidates.tsv.gz',sep='\t',index=False,compression='gzip')
(O/'STATUS.json').write_text(json.dumps({'status':'COMPLETE','new_LinkPeaks_fits':0,'comparisons':15,'reference_links':11885,'scope':'conditional agreement among saved threshold-retained candidates, not regulatory precision or full eligible-pair PR'},indent=2))
