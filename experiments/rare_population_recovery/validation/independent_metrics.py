"""Independent count-based evaluator. Does not import the experiment evaluator."""
from collections import Counter, defaultdict
import math
from statistics import mean

MISSING_ASSIGNMENTS = {'-1','nan','None','unassigned'}

def quantile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered)-1)*q
    left, right = math.floor(position), math.ceil(position)
    return float(ordered[left] + (ordered[right]-ordered[left])*(position-left))

def evaluate(labels, assignments, requested_k):
    truth = list(map(str, labels))
    membership = list(map(str, assignments))
    if len(truth) != len(membership) or not truth:
        raise ValueError('Invalid cohort lengths')
    totals = Counter(truth)
    groups = defaultdict(Counter)
    for label, group in zip(truth, membership):
        if group not in MISSING_ASSIGNMENTS:
            groups[group][label] += 1
    if not groups:
        raise ValueError('A successful fit cannot have no assigned cells')
    sizes = {group:sum(counts.values()) for group, counts in groups.items()}
    correct_plurality = Counter()
    group_rows = {}
    for group, counts in groups.items():
        chosen = sorted(counts, key=lambda label:(-counts[label],label))[0]
        correct_plurality[chosen] += counts[chosen]
        group_rows[group] = {'size':sizes[group], 'majority_celltype':chosen,
                             'purity':counts[chosen]/sizes[group]}
    populations = {}
    for label in sorted(totals):
        # Integer inequality gives exactly the inclusive >= 0.5 rule.
        selected = [group for group, counts in groups.items() if 2*counts[label] >= sizes[group]]
        tp = sum(groups[group][label] for group in selected)
        predicted = sum(sizes[group] for group in selected)
        true_n = totals[label]
        precision = tp/predicted if predicted else 0.0
        recall = tp/true_n
        f1 = 2*tp/(predicted+true_n) if predicted else 0.0
        selected_sizes = [sizes[group] for group in selected]
        populations[label] = {'celltype':label, 'n_celltype':true_n, 'abundance':true_n/len(truth),
            'precision':precision,'recall':recall,'F1':f1,
            'pooled_purity':tp/predicted if predicted else None,
            'mean_metacell_purity':mean(groups[group][label]/sizes[group] for group in selected) if selected else None,
            'target_dominated_metacells':len(selected),
            'median_metacell_size':quantile(selected_sizes,.5),
            'mean_metacell_size':mean(selected_sizes) if selected else None,
            'min_metacell_size':min(selected_sizes) if selected else None,
            'max_metacell_size':max(selected_sizes) if selected else None,
            'true_positive_cells':tp,'predicted_positive_cells':predicted}
    size_values = list(sizes.values())
    assigned = sum(size_values)
    summary = {'n_cells':len(truth),'n_assigned':assigned,'coverage':assigned/len(truth),
        'outlier_fraction':1-assigned/len(truth),'requested_K':requested_k,'realized_K':len(groups),
        'empty_anchors':requested_k-len(groups),'compression_N_over_K':len(truth)/len(groups),
        'mean_assigned_size':mean(size_values),'size_min':min(size_values),'size_max':max(size_values),
        'singleton_metacell_fraction':sum(n==1 for n in size_values)/len(size_values),
        'cells_in_metacells_smaller_than_5_fraction':sum(n for n in size_values if n<5)/len(truth),
        'balanced_accuracy_plurality':mean(correct_plurality[label]/totals[label] for label in totals),
        'mean_metacell_purity':mean(row['purity'] for row in group_rows.values()),
        'cell_weighted_purity':sum(max(counts.values()) for counts in groups.values())/len(truth),
        'population_macro_precision':mean(row['precision'] for row in populations.values()),
        'population_macro_recall':mean(row['recall'] for row in populations.values()),
        'population_macro_F1':mean(row['F1'] for row in populations.values())}
    for name, q in [('size_p05',.05),('size_p25',.25),('size_median',.5),('size_p75',.75),('size_p95',.95)]:
        summary[name] = quantile(size_values,q)
    return summary,populations,group_rows

def test_independent_metrics():
    summary,pop,groups=evaluate(['A','A','B','B','C','D'],[0,0,0,1,1,-1],3)
    assert abs(pop['A']['F1']-.8)<1e-12
    assert pop['B']['F1']==.5
    assert abs(pop['C']['F1']-2/3)<1e-12
    assert pop['D']['F1']==0 and pop['D']['pooled_purity'] is None
    assert summary['balanced_accuracy_plurality']==.375
    assert summary['n_assigned']==5 and summary['realized_K']==2 and summary['empty_anchors']==1
    assert summary['size_median']==2.5
    # A 50:50 group qualifies for both independent population recovery tests.
    summary,pop,groups=evaluate(['B','A'],[0,0],1)
    assert groups['0']['majority_celltype']=='A'
    assert pop['A']['recall']==1 and pop['B']['recall']==1
    assert pop['A']['precision']==.5 and pop['B']['precision']==.5
    # All classes stay in macro scores even if every member is unassigned.
    summary,pop,groups=evaluate(['A','B','C','D','E'],[0,'-1','nan','None','unassigned'],2)
    assert summary['coverage']==.2 and summary['population_macro_F1']==.2
    assert all(pop[x]['F1']==0 for x in ['B','C','D','E'])
    assert math.isclose(quantile([1,3,8],.95),7.5,abs_tol=1e-12)
    return {'status':'PASS','counting_method':'independent Python Counter and integer threshold',
            'cases':['mixed groups','alphabetical plurality ties','inclusive 50% threshold',
                     'unassigned cells in denominator','zero-recovery retained','undefined purity','linear quantile']}

if __name__=='__main__':
    print(test_independent_metrics())
