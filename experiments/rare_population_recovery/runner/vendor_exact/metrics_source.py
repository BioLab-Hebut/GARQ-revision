"""Annotation-only evaluation; no metric is used for fitting or run selection."""
import numpy as np
import pandas as pd

def evaluate(labels, assignments, requested_k=None):
    labels=np.asarray(labels,dtype=str); assignments=np.asarray(assignments)
    valid=np.asarray([str(v) not in ['-1','nan','None','unassigned'] for v in assignments])
    types,li=np.unique(labels,return_inverse=True)
    clusters,ci=np.unique(assignments[valid],return_inverse=True)
    counts=np.zeros((len(clusters),len(types)),dtype=np.int64)
    np.add.at(counts,(ci,li[valid]),1)
    sizes=counts.sum(1); totals=np.bincount(li,minlength=len(types))
    if not len(sizes):raise ValueError('No assigned cells')
    fractions=counts/sizes[:,None]
    plurality=counts.argmax(1) # ties resolved by alphabetical label order
    ba=float(np.mean(np.bincount(plurality,weights=counts[np.arange(len(sizes)),plurality],minlength=len(types))/totals))
    pop=[]
    for i,label in enumerate(types):
        dominated=fractions[:,i]>=.5
        tp=int(counts[dominated,i].sum());pred=int(sizes[dominated].sum());n=int(totals[i])
        precision=tp/pred if pred else 0.;recall=tp/n
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.
        pop.append({'celltype':label,'n_celltype':n,'abundance':n/len(labels),'precision':precision,'recall':recall,'F1':f1,'pooled_purity':tp/pred if pred else np.nan,'mean_metacell_purity':float(fractions[dominated,i].mean()) if dominated.any() else np.nan,'target_dominated_metacells':int(dominated.sum()),'median_metacell_size':float(np.median(sizes[dominated])) if dominated.any() else np.nan,'mean_metacell_size':float(sizes[dominated].mean()) if dominated.any() else np.nan,'min_metacell_size':int(sizes[dominated].min()) if dominated.any() else np.nan,'max_metacell_size':int(sizes[dominated].max()) if dominated.any() else np.nan,'true_positive_cells':tp,'predicted_positive_cells':pred})
    summary={'n_cells':len(labels),'n_assigned':int(valid.sum()),'coverage':float(valid.mean()),'outlier_fraction':float(1-valid.mean()),'requested_K':requested_k,'realized_K':len(sizes),'empty_anchors':int(requested_k-len(sizes)) if requested_k is not None else None,'compression_N_over_K':len(labels)/len(sizes),'mean_assigned_size':float(sizes.mean()),'size_min':int(sizes.min()),'size_p05':float(np.quantile(sizes,.05)),'size_p25':float(np.quantile(sizes,.25)),'size_median':float(np.median(sizes)),'size_p75':float(np.quantile(sizes,.75)),'size_p95':float(np.quantile(sizes,.95)),'size_max':int(sizes.max()),'singleton_metacell_fraction':float(np.mean(sizes==1)),'cells_in_metacells_smaller_than_5_fraction':float(sizes[sizes<5].sum()/len(labels)),'balanced_accuracy_plurality':ba,'mean_metacell_purity':float(fractions.max(1).mean()),'cell_weighted_purity':float(counts.max(1).sum()/len(labels)),'population_macro_precision':float(np.mean([x['precision'] for x in pop])),'population_macro_recall':float(np.mean([x['recall'] for x in pop])),'population_macro_F1':float(np.mean([x['F1'] for x in pop]))}
    return summary,pop,pd.DataFrame({'metacell':clusters,'size':sizes,'majority_celltype':types[plurality],'purity':fractions.max(1)})

def test_metrics():
    # Mixed/tied clusters, a missed class, and an unassigned cell have known outcomes.
    s,p,m=evaluate(['A','A','B','B','C','D'],[0,0,0,1,1,-1],3)
    q={x['celltype']:x for x in p}
    assert abs(q['A']['precision']-2/3)<1e-12 and q['A']['recall']==1
    assert abs(q['A']['F1']-.8)<1e-12 and q['B']['F1']==.5 and q['D']['F1']==0
    assert abs(q['C']['F1']-2/3)<1e-12
    assert abs(s['balanced_accuracy_plurality']-.375)<1e-12
    assert s['realized_K']==2 and s['empty_anchors']==1 and s['n_assigned']==5
    assert s['size_median']==2.5 and abs(s['compression_N_over_K']-3)<1e-12
    assert np.isnan(q['D']['pooled_purity'])
    return {'toy_counts':'passed','target_fraction_threshold_inclusive':0.5,'threshold_provenance':'matches the supplied analysis source (fraction >= purity_threshold)','ties':'alphabetical for plurality BA; a 50:50 metacell qualifies for each of the two independent population recovery calculations','zero_predicted_positive':'precision and F1 are zero; purity and target-metacell sizes are undefined'}

if __name__=='__main__':print(test_metrics())
