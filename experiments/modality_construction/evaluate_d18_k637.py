"""Re-evaluate seven EXISTING D18 K637 runs in saved clean references.

No training. Source H5AD, prepared files and reference caches are read-only.
Example (in the existing GARQ Python environment):
  python evaluate_d18_k637.py --output /path/to/new/D18_K637_evaluation

Requires existing reference caches, numpy, pandas and anndata. Barcode suffix
normalization is explicit: terminal '.6' and '-6' denote the same paired cell.
Historical assignment obs_names are positional 0..N-1; raw input row order and
all per-cell annotations are checked before mapping them to reference barcodes.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from release_paths import load_json, env_path, PACKAGE_ROOT

EXPECTED = {
    "RNA": (635, "8fc917a4bb5e8e424574a49d3a938f16590010649b2a1440a05c1f59de1fe142"),
    "ATAC": (634, "c587222dd9cec85b8b35fab57f81d025b15677c0eb542a9f4fc9f2d7396e4296"),
    "ADT": (605, "2f07ab93c0a950d756d87ae1db6a3b1b4c2b0e4aa2291380cb43da2808a2e66f"),
    "RNA_ATAC": (625, "1d1bec57a153a5d9696a47455759cbb03db0be3b02a93d50e02d715664e18066"),
    "RNA_ADT": (632, "ec26ce4ab0441f8807e03adaa1f74e74dd33788fd92bb8fdc26f5d7f14c814f7"),
    "ATAC_ADT": (597, "17668603f04283408977a294204377ec4fb5de9d503c486a5118f6d23080b05d"),
    "RNA_ATAC_ADT": (617, "5a3b03ce5b4a6cd34b6f05b34926c125fdc55ddc7dc5c6fdd492ba212c81e3d7"),
}
N_CELLS, REQUESTED_K, EVAL_K = 25517, 637, 15
INPUT_RECORDS = {}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    digest = h.hexdigest()
    INPUT_RECORDS[str(path)] = {"bytes": path.stat().st_size, "sha256": digest}
    return digest


def read_json(path):
    sha_file(path)
    return load_json(path)


def ids_hash(ids):
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()


def canonical(ids):
    values = np.asarray([re.sub(r"\.(\d+)$", r"-\1", str(x)) for x in ids], dtype=str)
    require(pd.Index(values).is_unique, "Duplicate barcodes after terminal separator normalization")
    return values


def align(reference, observed):
    ref, obs = pd.Index(reference), pd.Index(observed)
    require(ref.is_unique and obs.is_unique, "Non-unique cells")
    order = obs.get_indexer(ref)
    require(len(ref)==len(obs) and (order>=0).all(), "Cell sets do not match")
    return order


def finite_mean(values):
    values = np.asarray(values)
    return float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else np.nan


def annotation_metrics(ids, labels):
    """Independent majority-label evaluation; lexical tie-break matches old evaluator."""
    labels = np.asarray(labels, dtype=str)
    table = pd.crosstab(pd.Series(ids, name="anchor"), pd.Series(labels, name="type"))
    table = table.reindex(sorted(table.columns), axis=1)
    size = table.sum(axis=1)
    winner = table.idxmax(axis=1)
    purity = table.max(axis=1)/size
    tied = table.eq(table.max(axis=1), axis=0).sum(axis=1)>1
    prediction = pd.Series(ids).map(winner).to_numpy()
    per_type = []
    for label in table.columns:
        true, pred = labels==label, prediction==label
        tp, fp, fn = int(np.sum(true & pred)), int(np.sum(~true & pred)), int(np.sum(true & ~pred))
        precision = tp/(tp+fp) if tp+fp else 0.0
        recall = tp/(tp+fn)
        f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
        dominated = winner.eq(label)
        retained = dominated & purity.ge(.7) & ~tied
        per_type.append({"celltype": label, "n_cells": int(true.sum()), "frequency": float(true.mean()),
            "rare_le_1pct": bool(true.mean()<=.01), "tp":tp, "fp":fp, "fn":fn,
            "precision":precision, "recall":recall, "f1":f1,
            "retained_recall_purity_ge_0_7":float(table.loc[retained,label].sum()/true.sum()),
            "dominant_anchor_count":int(dominated.sum()),
            "dominant_anchor_mean_purity":float(purity[dominated].mean()) if dominated.any() else np.nan,
            "dominant_anchor_median_size":float(size[dominated].median()) if dominated.any() else np.nan})
    per_type = pd.DataFrame(per_type)
    # A second BA calculation from independently selected true-label subsets.
    ba_direct = np.mean([np.mean(prediction[labels==label]==label) for label in np.unique(labels)])
    require(np.isclose(ba_direct, per_type.recall.mean(), atol=1e-14, rtol=0), "Independent BA checks disagree")
    anchors = pd.DataFrame({"anchor":table.index, "size":size.to_numpy(),
        "majority_label":winner.to_numpy(), "purity":purity.to_numpy(), "majority_tie":tied.to_numpy()})
    return per_type, anchors


def neighborhood(ids, knn):
    _, inverse, sizes = np.unique(ids, return_inverse=True, return_counts=True)
    retention = np.mean(ids[knn]==ids[:,None], axis=1)
    chance = (sizes[inverse]-1)/(len(ids)-1)
    adjusted = np.divide(retention-chance,1-chance,out=np.full(len(ids),np.nan),where=chance<1)
    return retention,chance,adjusted


def within_cosine(z, ids):
    z = np.asarray(z,dtype=np.float64)
    norm = np.linalg.norm(z,axis=1)
    valid = norm>1e-12
    z = np.divide(z,norm[:,None],out=np.zeros_like(z),where=valid[:,None])
    _, inv = np.unique(ids,return_inverse=True)
    sums = np.zeros((int(inv.max())+1,z.shape[1]))
    np.add.at(sums,inv,z)
    counts = np.bincount(inv,weights=valid.astype(float))
    denominator = counts[inv]-valid
    numerator = np.einsum("ij,ij->i",z,sums[inv])-valid
    return np.clip(np.divide(numerator,denominator,out=np.full(len(ids),np.nan),
                            where=valid & (denominator>0)),-1,1)


def joint_embedding_scale(embedding, raw_info):
    """Single saved clean run; cell dispersion is not between-seed uncertainty."""
    z=np.asarray(embedding,dtype=np.float64)
    require(z.shape==(N_CELLS,96) and np.isfinite(z).all(), "Expected finite 96D concatenated RNA/ATAC/ADT embedding")
    blocks=np.split(z,3,axis=1)
    norm2=np.stack([np.sum(block**2,axis=1) for block in blocks],axis=1)
    norm2_sum=norm2.sum(axis=1,keepdims=True)
    require((norm2_sum>0).all(), "Zero joint embedding cannot define a squared-norm share")
    shares=norm2/norm2_sum
    coordinate_variances=[np.var(block,axis=0,ddof=1) for block in blocks]
    traces=np.asarray([v.sum() for v in coordinate_variances])
    require(traces.sum()>0,"Zero coordinate variance")
    rows,coordinates=[],[]
    for index,mod in enumerate(("RNA","ATAC","ADT")):
        norms=np.sqrt(norm2[:,index])
        rows.append({"dataset":"D18","seed":1,"condition":"clean",
            "series":"historical_K637_batch512","modality_combination":"RNA+ATAC+ADT",
            "data_type":mod,"n_cells":N_CELLS,"original_features":int(raw_info[mod]["n_features"]),
            "requested_k":REQUESTED_K,"realized_k":EXPECTED["RNA_ATAC_ADT"][0],"block_dim":32,
            "l2_norm_mean":float(norms.mean()),"l2_norm_median":float(np.median(norms)),
            "l2_norm_p95":float(np.quantile(norms,.95)),"l2_norm_sd_across_cells":float(norms.std(ddof=1)),
            "mean_squared_norm_share":float(shares[:,index].mean()),
            "variance_trace":float(traces[index]),"variance_per_dim_mean":float(coordinate_variances[index].mean()),
            "variance_share":float(traces[index]/traces.sum()),
            "pairwise_squared_distance_energy":float(2*traces[index]),
            "variance_ddof":1,"normalization_before_concat":False,"modality_weight":1.0,
            "n_independent_runs":1})
        for dimension,value in enumerate(coordinate_variances[index],start=1):
            coordinates.append({"dataset":"D18","seed":1,"data_type":mod,
                "block_coordinate":dimension,"joint_coordinate":index*32+dimension,
                "coordinate_sample_variance":float(value),"variance_ddof":1})
    require(np.isclose(sum(r["mean_squared_norm_share"] for r in rows),1), "Norm shares do not sum to one")
    require(np.isclose(sum(r["variance_share"] for r in rows),1), "Variance shares do not sum to one")
    return rows,coordinates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=env_path('GARQ_HISTORICAL_OUTPUT',PACKAGE_ROOT/'external_original_outputs'))
    parser.add_argument("--reference-root", type=Path,
        default=env_path('GARQ_REFERENCE_ROOT',PACKAGE_ROOT/'external_references'))
    parser.add_argument("--output", type=Path, required=True, help="New empty output directory")
    args = parser.parse_args()
    base, root, out = args.base.resolve(), args.reference_root.resolve(), args.output.resolve()
    require(not out.exists() or not any(out.iterdir()), "Output must be a new or empty directory; existing results will not be overwritten")
    for source_dir in (base/"save",root/"prepared",root/"references",root/"runs"):
        require(out!=source_dir and source_dir not in out.parents, "Output must not be inside old source/reference/run directories")
    prepared, ref_dir = root/"prepared/D18",root/"references/D18"
    manifest_path, settings_path = prepared/"manifest.json",ref_dir/"settings.json"
    require(manifest_path.is_file() and settings_path.is_file(), "Saved clean reference setup is missing; do not silently build a different reference")
    manifest, settings = read_json(manifest_path),read_json(settings_path)
    require(settings["prepared_manifest_sha256"]==sha_file(manifest_path), "Reference/input manifest mismatch")
    require(int(settings["evaluation_k"])==EVAL_K, "Expected the same 15NN reference")
    cells_path = prepared/"cells.csv"
    sha_file(cells_path)
    cells = pd.read_csv(cells_path,dtype=str)
    require(len(cells)==N_CELLS and cells.cell_id.is_unique, "Expected 25,517 unique D18 cells")
    require(not cells.celltype.isna().any(), "Missing reference annotations")
    cell_ids, labels = cells.cell_id.to_numpy(),cells.celltype.to_numpy()
    require(np.array_equal(canonical(cell_ids),cell_ids), "Reference barcodes must already be canonical")
    require(ids_hash(cell_ids)==manifest["cell_ids_sha256"], "Reference cells hash mismatch")
    refs = {}
    for mod in ("RNA","ATAC","ADT"):
        path=ref_dir/(mod+".npz")
        require(path.is_file(), "Missing saved reference "+str(path))
        sha_file(path)
        with np.load(path,allow_pickle=False) as cache:
            require(str(cache["cell_ids_sha256"])==ids_hash(cell_ids), "Reference embedding row hash mismatch")
            z,knn=cache["embedding"],cache["neighbors"]
        require(z.shape[0]==N_CELLS and np.isfinite(z).all(), "Invalid embedding")
        require(knn.shape==(N_CELLS,EVAL_K) and np.issubdtype(knn.dtype,np.integer), "Invalid neighbor shape/type")
        require(knn.min()>=0 and knn.max()<N_CELLS and not (knn==np.arange(N_CELLS)[:,None]).any(), "Invalid or self neighbors")
        refs[mod]=(z,knn)

    raw_orders,raw_label_orders,raw_info={},{},{}
    for mod in ("RNA","ATAC","ADT"):
        raw_path=Path(manifest["inputs"][mod]["path"])
        require(sha_file(raw_path)==manifest["inputs"][mod]["sha256"], "Raw input file changed: "+str(raw_path))
        a=ad.read_h5ad(raw_path,backed="r")
        try:
            require(a.n_obs==N_CELLS and "celltype" in a.obs, "Unexpected raw input shape/labels")
            require(not a.obs.celltype.isna().any(), "Missing raw annotations")
            original=np.asarray(a.obs_names,dtype=str)
            normalized=canonical(original)
            raw_labels=np.asarray(a.obs.celltype.astype(str),dtype=str)
            order=align(cell_ids,normalized)
            require(np.array_equal(raw_labels[order],labels), "Raw versus fixed-reference annotation mismatch: "+mod)
            raw_orders[mod],raw_label_orders[mod]=normalized,raw_labels
            raw_info[mod]={"path":str(raw_path),"n_cells":a.n_obs,"n_features":a.n_vars,
                "canonical_row_ids_sha256":ids_hash(normalized),
                "terminal_separator_changes":int(np.sum(original!=normalized)),
                "reordering_to_reference_required":bool(np.any(order!=np.arange(N_CELLS)))}
        finally:
            a.file.close()
    # Historical loader had no barcode realignment. Each multi-input saved run is
    # valid only if its original input modalities were already in paired row order.
    for mod in ("ATAC","ADT"):
        require(np.array_equal(raw_orders["RNA"],raw_orders[mod]), "Original modality rows were unpaired; cannot validate historical concatenation: "+mod)

    rna_knn,atac_knn=refs["RNA"][1],refs["ATAC"][1]
    overlap=np.asarray([len(set(x)&set(y)) for x,y in zip(rna_knn,atac_knn)])
    jaccard=overlap/(2*EVAL_K-overlap)
    cutoff=float(np.quantile(jaccard,.25))
    low=jaccard<=cutoff
    rare=cells.celltype.map(cells.celltype.value_counts(normalize=True)).to_numpy()<=.01
    populations=[("all_cells",np.ones(N_CELLS,dtype=bool)),("low_RNA_ATAC_concordance",low),("rare_annotations_le_1pct",rare)]
    out.mkdir(parents=True,exist_ok=True)
    (out/"anchors").mkdir()
    (out/"assignments").mkdir()
    rows,views,types,runs=[],[],[],[]
    scale_rows,coordinate_rows=[],[]
    for combo,(expected_k,expected_hash) in EXPECTED.items():
        path=base/"save"/("GSE158013637_"+combo+"_637metacell_k5_ids.h5ad")
        require(sha_file(path)==expected_hash, "Historical saved assignment hash mismatch: "+combo)
        a=ad.read_h5ad(path,backed="r")
        try:
            require(a.n_obs==N_CELLS and {"metacell","celltype"}<=set(a.obs), "Incomplete saved assignments")
            require(np.array_equal(np.asarray(a.obs_names,dtype=str),np.arange(N_CELLS).astype(str)), "Saved row names are not expected positional IDs")
            first_mod=combo.split("_")[0]
            saved_labels=np.asarray(a.obs.celltype.astype(str),dtype=str)
            require(np.array_equal(saved_labels,raw_label_orders[first_mod]), "Saved labels disagree with original input row order: "+combo)
            ids=pd.to_numeric(a.obs.metacell,errors="raise").to_numpy()
            if combo=="RNA_ATAC_ADT":
                embedded=a.X
                if hasattr(embedded,"to_memory"):
                    embedded=embedded.to_memory()
                if hasattr(embedded,"toarray"):
                    embedded=embedded.toarray()
                scale_rows,coordinate_rows=joint_embedding_scale(np.asarray(embedded),raw_info)
        finally:
            a.file.close()
        order=align(cell_ids,raw_orders[first_mod])
        ids=ids[order]
        require(np.isfinite(ids).all() and np.all(ids==np.rint(ids)) and ids.min()>=0 and ids.max()<REQUESTED_K, "Invalid metacell IDs")
        ids=ids.astype(np.int64)
        unique,sizes=np.unique(ids,return_counts=True)
        require(len(unique)==expected_k, "Unexpected actual K: "+combo)
        meta={"dataset":"D18","construction":combo.replace("_","+"),"condition":"clean","seed":1,
              "series":"historical_K637_batch512","requested_k":REQUESTED_K,"realized_k":len(unique),
              "within_5pct_of_target":abs(len(unique)-REQUESTED_K)/REQUESTED_K<=.05}
        pt,anchors=annotation_metrics(ids,labels)
        row={**meta,"n_cells":N_CELLS,"target_k":REQUESTED_K,"empty_anchors":REQUESTED_K-len(unique),
             "cells_per_nonempty_anchor":N_CELLS/len(unique),"size_min":int(sizes.min()),
             "size_median":float(np.median(sizes)),"size_p95":float(np.quantile(sizes,.95)),
             "size_max":int(sizes.max()),"singleton_anchor_fraction":float(np.mean(sizes==1)),
             "balanced_accuracy":float(pt.recall.mean()),"macro_f1":float(pt.f1.mean()),
             "rare_macro_f1":float(pt.loc[pt.rare_le_1pct,"f1"].mean()),"source_file":str(path)}
        rows.append(row)
        for mod,(z,knn) in refs.items():
            retention,chance,adjusted=neighborhood(ids,knn)
            cosine=within_cosine(z,ids)
            for population,mask in populations:
                require(mask.any(),"Empty evaluation population")
                views.append({**meta,"evaluation_modality":mod,"population":population,
                    "n_evaluated":int(mask.sum()),"neighbor_retention":float(retention[mask].mean()),
                    "chance_retention":float(chance[mask].mean()),"size_adjusted_retention":finite_mean(adjusted[mask]),
                    "within_anchor_cosine":finite_mean(cosine[mask]),"n_homogeneity_defined":int(np.isfinite(cosine[mask]).sum())})
            bytype=pd.DataFrame({"celltype":labels,mod+"_neighbor_retention":retention,
                mod+"_size_adjusted_retention":adjusted,mod+"_within_anchor_cosine":cosine}).groupby("celltype").mean()
            pt=pt.merge(bytype,on="celltype",validate="one_to_one")
            byanchor=pd.DataFrame({"anchor":ids,mod+"_neighbor_retention":retention,
                mod+"_within_anchor_cosine":cosine}).groupby("anchor").mean()
            anchors=anchors.merge(byanchor,on="anchor",validate="one_to_one")
        for key,value in meta.items():
            pt[key]=value
        types.append(pt)
        anchors.to_csv(out/"anchors"/("D18_"+combo+"_K637.csv.gz"),index=False)
        pd.DataFrame({"cell_id":cell_ids,"celltype":labels,"metacell":ids}).to_csv(
            out/"assignments"/("D18_"+combo+"_K637.csv.gz"),index=False)
        runs.append({**meta,"source_file":str(path),"source_sha256":expected_hash,
            "row_mapping":"saved positional IDs -> verified raw first-modality row barcodes -> fixed-reference order",
            "majority_tie_anchors":int(anchors.majority_tie.sum()),
            "cells_in_majority_tie_anchors":int(anchors.loc[anchors.majority_tie,"size"].sum())})
        print(combo,"actual K",len(unique),"BA",row["balanced_accuracy"],flush=True)
    pd.DataFrame(rows).to_csv(out/"D18_resolution_and_accuracy.csv",index=False)
    pd.DataFrame(views).to_csv(out/"D18_fixed_reference_metrics.csv",index=False)
    pd.concat(types,ignore_index=True).to_csv(out/"D18_celltype_metrics.csv",index=False)
    require(len(scale_rows)==3 and len(coordinate_rows)==96, "Clean joint scale diagnostic was not generated")
    pd.DataFrame(scale_rows).to_csv(out/"D18_joint_embedding_diagnostics.csv",index=False)
    pd.DataFrame(coordinate_rows).to_csv(out/"D18_joint_coordinate_variances.csv",index=False)
    report={"training_performed":False,"source_files_modified":False,"n_reused_runs":len(runs),
        "requested_K":REQUESTED_K,"n_cells":N_CELLS,"evaluation_k":EVAL_K,
        "seed_annotation":"Historical runs identified as seed 1 in the prior run audit; no new seeds run.",
        "raw_input_alignment_audit":raw_info,"reference_settings":settings,
        "reference_cell_ids_sha256":ids_hash(cell_ids),"reference_root":str(root),
        "low_concordance_definition":{"jaccard_cutoff":cutoff,"count":int(low.sum()),
            "rule":"lower quartile including all cutoff ties"},
        "runs":runs,"input_files":INPUT_RECORDS,"joint_embedding_scale":{
            "source_run":"GSE158013637_RNA_ATAC_ADT","n_independent_runs":1,
            "source_embedding":"X in saved joint assignment H5AD; RNA,ATAC,ADT blocks of 32 dimensions",
            "mean_squared_norm_share":"Per-cell block squared L2 norm divided by that cell's total joint squared norm, then averaged over cells",
            "variance_share":"Sum of sample variances (ddof=1) of a modality's 32 coordinates divided by the sum across all 96 coordinates",
            "uncertainty":"No between-seed SD or replicate error bars are available; l2_norm_sd_across_cells describes cell dispersion only",
            "interpretation":"Numerical scale diagnostics, not causal attribution or equal biological influence"},
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "notes":["All modality constructions evaluated against the same saved clean references as the original R2 evaluation.",
                 "Actual K differs by construction and must be reported; requested K637 does not imply matched actual K.",
                 "Majority-label BA is mean class recall; F1 and high-purity recall are separate descriptive annotation metrics.",
                 "Canonical barcode normalization replaces only a final dot before numeric sample suffix with a hyphen; no cells are dropped.",
                 "Positional source IDs are mapped through raw input order only after all saved per-cell annotations and paired raw row IDs agree."]}
    (out/"D18_K637_evaluation_manifest.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print("DONE: seven existing assignments re-evaluated; no training.",flush=True)


if __name__=="__main__":
    main()
