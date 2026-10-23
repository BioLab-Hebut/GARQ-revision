"""Export complete global links and plot summaries, retaining dataset/arm/batch."""
import pathlib,json,shutil
import pandas as pd
import numpy as np
r=pathlib.Path('/workspace/garq/vscode/GARQ20260905/reviewer2_comment4_20260918')
D=r/'package/data';E=r/'package/evidence';all_files=sorted((r/'peak_gene').glob('*/*/DONE'))
assert len(all_files)==11, f'Expected 11 completed global controls, found {len(all_files)}'
summaries=[];frames=[];bins=[];overlaps=[];links={}
for done in all_files:
 p=done.parent;s=json.loads((p/'summary.json').read_text());batch=s['batch'];arm=s['arm'];d=pd.read_csv(p/'links.csv.gz')
 assert len(d)==s['links'] and d.pair_id.is_unique
 assert np.isfinite(d[['score','zscore','pvalue','distance']]).all().all()
 assert (d.pvalue<.05).all() and (abs(d.score)>.05).all()
 assert ((d.peak_center-d.gene_tss).abs()==d.distance).all()
 s['peak_midpoints_beyond_500kb']=int((d.distance>500000).sum())
 s['retained_fraction_of_candidate_pairs']=len(d)/s['candidate_cis_pairs']
 for k in ['source_paths']:s.pop(k,None)
 summaries.append(s);links[(batch,arm)]=set(d.pair_id)
 od=D/'global_links'/batch/arm;od.mkdir(parents=True,exist_ok=True)
 shutil.copy2(p/'links.csv.gz',od/'links.csv.gz');shutil.copy2(p/'summary.json',od/'summary.json')
 frame=d[['score','distance']].copy();frame['batch']=batch;frame['arm']=arm;frames.append(frame)
 db=pd.cut(d.distance,bins=[-1,10000,50000,100000,200000,np.inf],labels=['0-10','10-50','50-100','100-200','200-500'],right=True)
 for b,n in db.value_counts(sort=False).items():bins.append(dict(batch=batch,arm=arm,distance_bin=b,count=int(n),fraction=float(n/len(d))))
sc=links[('single_cell','SingleCell')]
for (batch,arm),v in links.items():
 if batch=='single_cell':continue
 overlaps.append(dict(batch=batch,arm=arm,single_total=len(sc),metacell_total=len(v),overlap=len(v&sc),single_only=len(sc-v),metacell_only=len(v-sc),jaccard=len(v&sc)/len(v|sc)))
pd.DataFrame(summaries).to_csv(D/'global_peak_gene_summary.csv',index=False)
pd.concat(frames,ignore_index=True).to_csv(D/'global_peak_gene_plot_data.csv.gz',index=False)
pd.DataFrame(bins).to_csv(D/'global_peak_gene_distance_bins.csv',index=False)
pd.DataFrame(overlaps).to_csv(D/'global_peak_gene_overlap.csv',index=False)
for name in ['fast_sampler_validation.txt','fast_LinkPeaks_validation.json','original_objects.json']:
 shutil.copy2(r/'audit'/name,E/name)
(E/'global_peak_gene_verification.json').write_text(json.dumps(dict(completed_controls=len(all_files),all_links_unique=True,all_cutoffs_verified=True,all_distances_verified=True,matching_batch='matched_seed1',note='The final distance bin includes peak-midpoint boundary overhang from overlap with the 500 kb TSS window.'),indent=2))
print(pd.DataFrame(summaries)[['batch','arm','links','strong_positive_links','genes','candidate_cis_pairs','peak_midpoints_beyond_500kb']].to_string(index=False))
