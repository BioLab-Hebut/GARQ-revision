"""R80 C: frozen memberships, prespecified full gene-antibody panel."""
from pathlib import Path
import json,hashlib,time,re,os
import numpy as np,pandas as pd,anndata as ad
from scipy import sparse,stats
from release_paths import output_root, env_path
R=output_root();O=R/'C'
OLD=env_path('GARQ_D18_ANALYSIS')
RAW=env_path('GARQ_D18_RAW')
COMBOS=['RNA','ATAC','ADT','RNA_ATAC','RNA_ADT','ATAC_ADT','RNA_ATAC_ADT']
ARMS=['GARQ_'+c for c in COMBOS]+['MetaQ','SEACells','MetaCellV2','SuperCell']
ALIAS={'CD10':'MME','CD11b':'ITGAM','CD11c':'ITGAX','CD123':'IL3RA','CD127':'IL7R','CD14':'CD14','CD141':'THBD','CD172a':'SIRPA','CD185':'CXCR5','CD19':'CD19','CD192':'CCR2','CD197':'CCR7','CD21':'CR2','CD24':'CD24','CD25':'IL2RA','CD269':'TNFRSF17','CD27':'CD27','CD278':'ICOS','CD279':'PDCD1','CD304':'NRP1','CD319':'SLAMF7','CD38':'CD38','CD39':'ENTPD1','CD4':'CD4','CD40':'CD40','CD56':'NCAM1','CD66b':'CEACAM8','CD71':'TFRC','CD80':'CD80','CD86':'CD86','CD8a':'CD8A','CD95':'FAS','KLRG1':'KLRG1','IgD':'IGHD','IgM':'IGHM'}
EXCLUDE={'CD16':'FCGR3A/FCGR3B specificity not resolved by saved antibody name','CD3':'multisubunit CD3 complex; clone specificity absent','CD45RA':'PTPRC isoform; gene-level counts do not distinguish epitope','CD45RO':'PTPRC isoform; gene-level counts do not distinguish epitope','FceRI':'multisubunit receptor; clone specificity absent','HLA.DR':'HLA-DR alpha/beta family; no unique gene mapping','IgG1.K.Isotype.Control':'isotype control','TCR.Va24.Ja18':'rearranged variable/joining epitope, not one gene','TCR.Va7.2':'variable segment epitope; clone/feature mapping absent','TCR.a.b':'multisubunit rearranged receptor','TCR.g.d':'multisubunit rearranged receptor'}
def js(p,x):p.write_text(json.dumps(x,indent=2,default=str))
def canon(x):return [re.sub(r'\.(\d+)$',r'-\1',str(v)) for v in x]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm_groups(v):
 invalid={'-1','-1.0','nan','None','outliers','__missing__','<NA>'}
 names=sorted(set(map(str,v))-invalid);lu={v:i for i,v in enumerate(names)}
 return np.array([lu.get(str(x),-1) for x in v]),names
def read_members(p,ids,labels,full):
 f=pd.read_csv(p,dtype={'cell_id':str});assert not f.cell_id.duplicated().any()
 gcol='group' if 'group' in f else 'metacell_id'
 lab='celltype' if 'celltype' in f else 'cell_type'
 assert set(f.cell_id)<=set(ids)
 if full:assert set(f.cell_id)==set(ids) or len(f)==25507
 idx=pd.Index(ids).get_indexer(f.cell_id);assert (idx>=0).all()
 assert np.array_equal(f[lab].astype(str).values,labels[idx])
 v=np.full(len(ids),'-1',object);v[idx]=f[gcol].astype(str)
 return norm_groups(v)[0]
def sample(g,seed):
 rng=np.random.default_rng(seed);eligible=[k for k in np.unique(g) if k>=0 and (g==k).sum()>=20]
 if len(eligible)<369:return None
 chosen=sorted(rng.choice(eligible,369,replace=False).tolist());v=np.full(len(g),-1,int)
 for j,k in enumerate(chosen):v[rng.choice(np.flatnonzero(g==k),20,replace=False)]=j
 return v
def corr(x,y,kind):
 if not np.isfinite(x).all() or not np.isfinite(y).all():return np.nan,'nonfinite'
 if len(x)<3 or np.ptp(x)<1e-12 or np.ptp(y)<1e-12:return np.nan,'constant_or_insufficient'
 return float(stats.spearmanr(x,y).statistic if kind=='spearman' else stats.pearsonr(x,y).statistic),'defined'
def main():
 start=time.time();O.mkdir(exist_ok=True)
 a=ad.read_h5ad(RAW/'GSE158013_rna.h5ad',backed='r');b=ad.read_h5ad(RAW/'GSE158013_adt.h5ad',backed='r')
 ids=canon(a.obs_names);labels=a.obs.celltype.astype(str).to_numpy();assert len(ids)==25517 and len(set(ids))==len(ids)
 bi=pd.Index(canon(b.obs_names)).get_indexer(ids);assert (bi>=0).all() and len(set(bi))==len(ids)
 names=list(b.var_names.astype(str));assert len(names)==46 and set(names)==set(ALIAS)|set(EXCLUDE)
 rows=[]
 for j,p in enumerate(names):
  gene=ALIAS.get(p,'');where=np.flatnonzero(a.var_names==gene)
  rows.append({'antibody':p,'adt_column':j,'gene':gene,'rna_column':int(where[0]) if len(where)==1 else -1,'mapped':bool(gene and len(where)==1),'reason':'unique_gene_target' if gene and len(where)==1 else EXCLUDE.get(p,'gene_missing_or_duplicated'),'biological_adt':p!='IgG1.K.Isotype.Control','alias_source':'ImmPort analyte ontology and HGNC canonical symbol; archived antibody name only'})
 mapping=pd.DataFrame(rows);mapping.to_csv(O/'panel_mapping.tsv',sep='\t',index=False)
 mapped=mapping[mapping.mapped];gix=mapped.rna_column.to_numpy(int);pidx=mapped.adt_column.to_numpy(int)
 assert {'CD8A','NCAM1'}<=set(mapped.gene)
 js(O/'MAPPING_LOCKED.json',{'mapping_sha256':sha(O/'panel_mapping.tsv'),'locked_before_correlations':True,'filter':'unique mapped RNA target; no score/ranking filter; globally absent or constant marked NA','biological_adt_panel':mapping.loc[mapping.biological_adt,'antibody'].tolist(),'source_url':'https://docs.dev.immport.org/datasubmission/templatedocumentation/lktables/lk_analyte/','n_mapped':len(mapped)})
 # Chunk raw RNA once: full-library totals, mapped columns only.
 X=np.empty((len(ids),len(gix)),np.float64);tot=np.empty(len(ids))
 for s in range(0,len(ids),1024):
  e=min(s+1024,len(ids));block=a.X[s:e,:]
  v=block.data if sparse.issparse(block) else np.asarray(block)
  assert np.isfinite(v).all() and v.min()>=0 and np.max(np.abs(v-np.round(v)))<1e-5,'RNA X not raw counts'
  tot[s:e]=np.asarray(block.sum(1)).ravel()
  sub=block[:,gix];X[s:e]=sub.toarray() if sparse.issparse(sub) else sub
 Y=np.asarray(b.X[:,:])[bi].astype(float)
 assert np.isfinite(Y).all() and Y.min()>=0 and np.max(np.abs(Y-np.round(Y)))<1e-5,'ADT X not raw counts'
 a.file.close();b.file.close()
 groups={};source=[]
 for arm in ARMS:
  p=env_path('GARQ_D18_METAQ_ASSIGNMENTS') if arm=='MetaQ' else OLD/'full'/arm/'memberships.csv.gz'
  if not p.exists():source.append({'arm':arm,'status':'MISSING','path':str(p)});continue
  groups[arm]=read_members(p,ids,labels,True)
  source.append({'arm':arm,'status':'AVAILABLE','path':str(p),'sha256':sha(p),'actual_K':len(np.unique(groups[arm][groups[arm]>=0])),'eligible_groups':sum((groups[arm]==k).sum()>=20 for k in np.unique(groups[arm]) if k>=0)})
 pd.DataFrame(source).to_csv(O/'input_availability.tsv',sep='\t',index=False)
 assert groups['GARQ_RNA_ATAC_ADT'].max()+1==617,'Must use historical downstream 617 grouping, not structural 608'
 common=np.logical_and.reduce([g>=0 for g in groups.values()])
 np.savez_compressed(O/'common_cell_ids.npz',cell_ids=np.array(ids)[common])
 designs=[]
 for arm,g in groups.items():
  v=g.copy();v[~common]=-1;designs.append((arm,'common_cell',0,norm_groups(v)[0],'fixed common native intersection'))
  for seed in [1,2,3]:
   p=OLD/f'matched_seed{seed}'/arm/'memberships.csv.gz'
   if p.exists():
    v=read_members(p,ids,labels,False);used=np.unique(v[v>=0])
    assert len(used)==369 and all((v==k).sum()==20 for k in used)
    assert all(len(np.unique(g[v==k]))==1 and g[v==k][0]>=0 for k in used)
    provenance=str(p)
   else:v=sample(g,seed);provenance='original sample_equal algorithm; no saved matched file'
   if v is None:continue
   designs.append((arm,'matched_369x20',seed,v,provenance))
 js(O/'COHORT_LOCKED.json',{'common_N':int(common.sum()),'all_input_N':len(ids),'designs':[{'arm':a,'design':d,'seed':s,'source':p,'n_cells':int((v>=0).sum()),'actual_K':len(np.unique(v[v>=0]))} for a,d,s,v,p in designs],'locked_before_correlations':True})
 metrics=[];qualities=[];allprofiles=[];population=[];types=np.unique(labels)
 for arm,design,seed,g,provenance in designs:
  valid=g>=0;k=int(g.max()+1);sizes=np.bincount(g[valid],minlength=k)
  M=sparse.csr_matrix((1/sizes[g[valid]],(g[valid],np.flatnonzero(valid))),shape=(k,len(ids)))
  xr=np.log1p((M@X)/np.maximum((M@tot)[:,None],1e-20)*1e4)
  yl=np.log1p(M@Y);yr=yl-yl[:,mapping.biological_adt.to_numpy()].mean(1,keepdims=True)
  counts=np.stack([np.bincount(g[valid],weights=(labels[valid]==t),minlength=k) for t in types],axis=1)
  dom=types[counts.argmax(1)];pred=np.array(['__UNASSIGNED__']*len(ids),object);pred[valid]=dom[g[valid]]
  recalls=[np.mean(pred[(labels==t)&valid]==t) if ((labels==t)&valid).any() else np.nan for t in types]
  qualities.append({'arm':arm,'design':design,'seed':seed,'actual_K':k,'n_assigned':int(valid.sum()),'cell_coverage':valid.mean(),'balanced_accuracy_conditional':np.nanmean(recalls),'balanced_accuracy_full':np.mean([np.mean(pred[labels==t]==t) for t in types]),'mean_purity':np.mean(counts.max(1)/sizes),'median_size':np.median(sizes),'p95_size':np.quantile(sizes,.95),'max_size':sizes.max()})
  for t,rec in zip(types,recalls):population.append({'arm':arm,'design':design,'seed':seed,'celltype':t,'n_original':int((labels==t).sum()),'n_retained':int(((labels==t)&valid).sum()),'recall_conditional':rec})
  for j,rr in enumerate(mapped.itertuples()):
   xx=xr[:,j];yy=yr[:,rr.adt_column];rx=xx.copy();ry=yy.copy()
   for ct in np.unique(dom):
    z=dom==ct;rx[z]-=rx[z].mean();ry[z]-=ry[z].mean()
   pr,ps=corr(xx,yy,'pearson');sr,ss=corr(xx,yy,'spearman');ar,ass=corr(rx,ry,'pearson')
   metrics.append({'arm':arm,'design':design,'seed':seed,'gene':rr.gene,'antibody':rr.antibody,'pearson':pr,'spearman':sr,'annotation_adjusted_pearson':ar,'pearson_status':ps,'spearman_status':ss,'adjusted_status':ass,'n_groups':k,'panel_denominator':len(mapped)})
  f=O/'profiles'/f'{arm}__{design}__s{seed}.npz';f.parent.mkdir(exist_ok=True)
  np.savez_compressed(f,rna=xr,adt=yr[:,pidx],dominant=dom,sizes=sizes,genes=mapped.gene.to_numpy(str),antibodies=mapped.antibody.to_numpy(str),metacell=g,cell_ids=np.asarray(ids))
  print('C',arm,design,seed,k,flush=True)
 m=pd.DataFrame(metrics);m.to_csv(O/'all_pair_metrics.tsv',sep='\t',index=False)
 pd.DataFrame(qualities).to_csv(O/'quality_coverage.tsv',sep='\t',index=False);pd.DataFrame(population).to_csv(O/'all_populations.tsv',sep='\t',index=False)
 summaries=[];deltas=[]
 for (arm,design,seed),v in m.groupby(['arm','design','seed']):
  for metric in ['pearson','spearman','annotation_adjusted_pearson']:
   y=v[metric].dropna();gy=v.groupby('gene')[metric].mean().dropna()
   summaries.append({'arm':arm,'design':design,'seed':seed,'metric':metric,'valid_pairs':len(y),'panel_denominator':len(mapped),'median':y.median(),'q25':y.quantile(.25),'q75':y.quantile(.75),'gene_equal_mean':gy.mean(),'gene_equal_median':gy.median()})
 for (design,seed),v in m.groupby(['design','seed']):
  q=v[v.arm=='GARQ_RNA_ATAC_ADT']
  for arm in v.arm.unique():
   if arm=='GARQ_RNA_ATAC_ADT':continue
   z=q.merge(v[v.arm==arm],on=['gene','antibody'],suffixes=('_GARQ','_other'))
   for metric in ['pearson','spearman','annotation_adjusted_pearson']:
    mask=z[metric+'_GARQ'].notna()&z[metric+'_other'].notna()
    for rr in z.loc[mask].itertuples():
     d=getattr(rr,metric+'_GARQ')-getattr(rr,metric+'_other')
     deltas.append({'design':design,'seed':seed,'other_arm':arm,'gene':rr.gene,'antibody':rr.antibody,'metric':metric,'GARQ_minus_other':d,'shared_valid_pairs':int(mask.sum()),'panel_denominator':len(mapped)})
 pd.DataFrame(summaries).to_csv(O/'panel_summaries.tsv',sep='\t',index=False)
 pd.DataFrame(deltas).to_csv(O/'all_paired_differences.tsv',sep='\t',index=False)
 js(O/'STATUS.json',{'status':'COMPLETE','arms':len(groups),'design_conditions':len(designs),'mapped_pairs':len(mapped),'metric_rows':len(m),'common_cells':int(common.sum()),'seconds':time.time()-start,'new_fits':0,'normalization':'RNA raw mean/full-library mean total *10000 log1p; ADT log1p(raw mean) centered over all45 biological antibodies','seeds_type':'technical matching resampling'})
if __name__=='__main__':main()
