import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, normalize


# ==============================================================================
# 1. 参数
# ==============================================================================

N_HVG = 3000
N_PCA_RNA = 50
N_LSI_ATAC = 50
N_PCA_ADT = 20
RANDOM_STATE = 0

OUTPUT_ROOT = Path("/path/to/project_home/results/xiaorong/fixed_representation")


DATASETS_RNA_ADT = [
    {
        "dataset": "GSE163120",
        "pair": "RNA_ADT",
        "rna_h5ad": "/path/to/project_home/data/RNA+ADT/GSE163120/GSE163120_rna.h5ad",
        "adt_h5ad": "/path/to/project_home/data/RNA+ADT/GSE163120/GSE163120_adt.h5ad",
        "output_h5ad": str(OUTPUT_ROOT / "RNA_ADT/GSE163120/GSE163120_fixed_embedding.h5ad"),
    },
    {
        "dataset": "12_GSE156478_control",
        "pair": "RNA_ADT",
        "rna_h5ad": "/path/to/project_home/data/RNA+ADT/12_GSE156478_control/12_GSE156478_control_rna.h5ad",
        "adt_h5ad": "/path/to/project_home/data/RNA+ADT/12_GSE156478_control/12_GSE156478_control_adt.h5ad",
        "output_h5ad": str(OUTPUT_ROOT / "RNA_ADT/12_GSE156478_control/12_GSE156478_control_fixed_embedding.h5ad"),
    },
    {
        "dataset": "D1",
        "pair": "RNA_ADT",
        "rna_h5ad": "/path/to/project_home/data/RNA+ADT/D1/D1_rna.h5ad",
        "adt_h5ad": "/path/to/project_home/data/RNA+ADT/D1/D1_adt.h5ad",
        "output_h5ad": str(OUTPUT_ROOT / "RNA_ADT/D1/D1_fixed_embedding.h5ad"),
    },
    {
        "dataset": "D5",
        "pair": "RNA_ADT",
        "rna_h5ad": "/path/to/project_home/data/RNA+ADT/D5/D5_rna.h5ad",
        "adt_h5ad": "/path/to/project_home/data/RNA+ADT/D5/D5_adt.h5ad",
        "output_h5ad": str(OUTPUT_ROOT / "RNA_ADT/D5/D5_fixed_embedding.h5ad"),
    },
    {
        "dataset": "D8",
        "pair": "RNA_ADT",
        "rna_h5ad": "/path/to/project_home/data/RNA+ADT/D8/D8_rna.h5ad",
        "adt_h5ad": "/path/to/project_home/data/RNA+ADT/D8/D8_adt.h5ad",
        "output_h5ad": str(OUTPUT_ROOT / "RNA_ADT/D8/D8_fixed_embedding.h5ad"),
    },
    {
        "dataset": "D9",
        "pair": "RNA_ADT",
        "rna_h5ad": "/path/to/project_home/data/RNA+ADT/D9/D9_rna.h5ad",
        "adt_h5ad": "/path/to/project_home/data/RNA+ADT/D9/D9_adt.h5ad",
        "output_h5ad": str(OUTPUT_ROOT / "RNA_ADT/D9/D9_fixed_embedding.h5ad"),
    },
]


DATASETS_RNA_ATAC = [
    {
        "dataset": "GSE194122",
        "pair": "RNA_ATAC",
        "rna_h5ad": "/path/to/project_home/data/RNA+ATAC/GSE194122/GSE194122_rna.h5ad",
        "atac_h5ad": "/path/to/project_home/data/RNA+ATAC/GSE194122/GSE194122_atac.h5ad",
        "output_h5ad": str(OUTPUT_ROOT / "RNA_ATAC/GSE194122/GSE194122_fixed_embedding.h5ad"),
    },
    {
        "dataset": "8_pbmc_unsorted_10k",
        "pair": "RNA_ATAC",
        "rna_h5ad": "/path/to/project_home/data/RNA+ATAC/8_pbmc_unsorted_10k/8_pbmc_unsorted_10k_rna.h5ad",
        "atac_h5ad": "/path/to/project_home/data/RNA+ATAC/8_pbmc_unsorted_10k/8_pbmc_unsorted_10k_atac.h5ad",
        "output_h5ad": str(OUTPUT_ROOT / "RNA_ATAC/8_pbmc_unsorted_10k/8_pbmc_unsorted_10k_fixed_embedding.h5ad"),
    },
    {
        "dataset": "shareseq_skin",
        "pair": "RNA_ATAC",
        "rna_h5ad": "/path/to/project_home/data/RNA+ATAC/shareseq_skin/shareseq_skin_rna.h5ad",
        "atac_h5ad": "/path/to/project_home/data/RNA+ATAC/shareseq_skin/shareseq_skin_atac.h5ad",
        "output_h5ad": str(OUTPUT_ROOT / "RNA_ATAC/shareseq_skin/shareseq_skin_fixed_embedding.h5ad"),
    },
    {
        "dataset": "bmmc",
        "pair": "RNA_ATAC",
        "rna_h5ad": "/path/to/project_home/data/RNA+ATAC/bmmc/bmmc_rna.h5ad",
        "atac_h5ad": "/path/to/project_home/data/RNA+ATAC/bmmc/bmmc_atac.h5ad",
        "output_h5ad": str(OUTPUT_ROOT / "RNA_ATAC/bmmc/bmmc_fixed_embedding.h5ad"),
    },
    {
        "dataset": "pbmc10k",
        "pair": "RNA_ATAC",
        "rna_h5ad": "/path/to/project_home/data/RNA+ATAC/pbmc10k/pbmc10k_rna.h5ad",
        "atac_h5ad": "/path/to/project_home/data/RNA+ATAC/pbmc10k/pbmc10k_atac.h5ad",
        "output_h5ad": str(OUTPUT_ROOT / "RNA_ATAC/pbmc10k/pbmc10k_fixed_embedding.h5ad"),
    },
    {
        "dataset": "chen",
        "pair": "RNA_ATAC",
        "rna_h5ad": "/path/to/project_home/data/RNA+ATAC/chen/chen_rna.h5ad",
        "atac_h5ad": "/path/to/project_home/data/RNA+ATAC/chen/chen_atac.h5ad",
        "output_h5ad": str(OUTPUT_ROOT / "RNA_ATAC/chen/chen_fixed_embedding.h5ad"),
    },
]


# ==============================================================================
# 2. 工具函数
# ==============================================================================

def to_dense_float32(X):
    if sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def get_celltype_obs(rna):
    for col in ["celltype", "cell_type", "CellType", "annotation", "cluster"]:
        if col in rna.obs.columns:
            return rna.obs[col].astype(str).values
    return np.array(["unknown"] * rna.n_obs)


def align_modalities(adatas):
    common = set(adatas[0].obs_names.astype(str))

    for x in adatas[1:]:
        common = common.intersection(set(x.obs_names.astype(str)))

    common = list(common)

    if len(common) == 0:
        raise ValueError("不同模态之间没有共同 obs_names，请检查 cell barcode 是否一致。")

    first_order = [x for x in adatas[0].obs_names.astype(str) if x in common]

    aligned = []
    for x in adatas:
        x.obs_names = x.obs_names.astype(str)
        aligned.append(x[first_order].copy())

    return aligned


def standardize_and_l2(X):
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    X = normalize(X, norm="l2", axis=1)
    return X.astype(np.float32)


# ==============================================================================
# 3. RNA: normalize_total + log1p + HVG + PCA
# ==============================================================================

def compute_rna_pca(rna, n_pca=N_PCA_RNA, n_hvg=N_HVG):
    rna = rna.copy()

    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)

    n_top = min(n_hvg, rna.n_vars)
    sc.pp.highly_variable_genes(rna, n_top_genes=n_top, flavor="seurat")

    if "highly_variable" in rna.var.columns and rna.var["highly_variable"].sum() > 0:
        rna = rna[:, rna.var["highly_variable"]].copy()

    sc.pp.scale(rna, max_value=10)

    n_comps = min(n_pca, rna.n_obs - 1, rna.n_vars - 1)
    if n_comps < 2:
        raise ValueError("RNA PCA 维度太小，请检查数据。")

    sc.tl.pca(rna, n_comps=n_comps, svd_solver="arpack", random_state=RANDOM_STATE)

    X = rna.obsm["X_pca"].astype(np.float32)

    if X.shape[1] < n_pca:
        pad = np.zeros((X.shape[0], n_pca - X.shape[1]), dtype=np.float32)
        X = np.hstack([X, pad])

    return X[:, :n_pca]


# ==============================================================================
# 4. ATAC: TF-IDF + LSI
# ==============================================================================

def compute_atac_lsi(atac, n_lsi=N_LSI_ATAC):
    atac = atac.copy()

    X = atac.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)

    X = X.astype(np.float32)

    cell_sums = np.asarray(X.sum(axis=1)).ravel()
    cell_sums[cell_sums == 0] = 1.0
    tf = X.multiply(1.0 / cell_sums[:, None])

    peak_counts = np.asarray((X > 0).sum(axis=0)).ravel()
    idf = np.log(1.0 + X.shape[0] / (1.0 + peak_counts))

    tfidf = tf.multiply(idf)

    n_comps = min(n_lsi + 1, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    if n_comps < 2:
        raise ValueError("ATAC LSI 维度太小，请检查数据。")

    svd = TruncatedSVD(n_components=n_comps, random_state=RANDOM_STATE)
    X_lsi = svd.fit_transform(tfidf)

    if X_lsi.shape[1] > 1:
        X_lsi = X_lsi[:, 1:]

    if X_lsi.shape[1] < n_lsi:
        pad = np.zeros((X_lsi.shape[0], n_lsi - X_lsi.shape[1]), dtype=np.float32)
        X_lsi = np.hstack([X_lsi, pad])

    return X_lsi[:, :n_lsi].astype(np.float32)


# ==============================================================================
# 5. ADT: CLR + PCA
# ==============================================================================

def compute_adt_pca(adt, n_pca=N_PCA_ADT):
    adt = adt.copy()

    X = to_dense_float32(adt.X)

    X_log = np.log1p(X)
    X_clr = X_log - X_log.mean(axis=1, keepdims=True)

    adt_clr = ad.AnnData(
        X=X_clr,
        obs=adt.obs.copy(),
        var=adt.var.copy(),
    )

    sc.pp.scale(adt_clr, max_value=10)

    n_comps = min(n_pca, adt_clr.n_obs - 1, adt_clr.n_vars - 1)
    if n_comps < 1:
        raise ValueError("ADT PCA 维度太小，请检查数据。")

    sc.tl.pca(adt_clr, n_comps=n_comps, svd_solver="arpack", random_state=RANDOM_STATE)

    X = adt_clr.obsm["X_pca"].astype(np.float32)

    if X.shape[1] < n_pca:
        pad = np.zeros((X.shape[0], n_pca - X.shape[1]), dtype=np.float32)
        X = np.hstack([X, pad])

    return X[:, :n_pca]


# ==============================================================================
# 6. 保存 fixed embedding
# ==============================================================================

def save_fixed_embedding(dataset, pair, output_h5ad, obs_names, celltype, fixed_blocks):
    block_names = list(fixed_blocks.keys())

    normalized_blocks = {}
    for name, X in fixed_blocks.items():
        normalized_blocks[name] = standardize_and_l2(X)

    X_fixed = np.hstack([normalized_blocks[name] for name in block_names]).astype(np.float32)
    X_fixed = normalize(X_fixed, norm="l2", axis=1).astype(np.float32)

    obs = pd.DataFrame(index=obs_names)
    obs["celltype"] = celltype
    obs["dataset"] = dataset
    obs["modality_pair"] = pair

    var = pd.DataFrame(index=[f"fixed_dim_{i}" for i in range(X_fixed.shape[1])])
    var["feature_type"] = "fixed_embedding"

    out = ad.AnnData(X=X_fixed, obs=obs, var=var)

    for name, X in normalized_blocks.items():
        out.obsm[name] = X

    out.obsm["X_fixed"] = X_fixed

    out.uns["fixed_representation"] = {
        "dataset": dataset,
        "pair": pair,
        "blocks": block_names,
        "rna": "normalize_total + log1p + HVG + PCA50",
        "atac": "TF-IDF + LSI50",
        "adt": "CLR + PCA20",
        "block_processing": "standardize each block + per-cell L2 normalization + equal-weight concatenation",
        "random_state": RANDOM_STATE,
    }

    output_h5ad = Path(output_h5ad)
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)

    out.write_h5ad(output_h5ad)

    print("Saved:", output_h5ad)
    print(out)


# ==============================================================================
# 7. RNA + ADT
# ==============================================================================

def process_rna_adt(item):
    print("=" * 100)
    print("Processing RNA+ADT:", item["dataset"])

    rna = sc.read_h5ad(item["rna_h5ad"])
    adt = sc.read_h5ad(item["adt_h5ad"])

    rna, adt = align_modalities([rna, adt])

    X_rna = compute_rna_pca(rna)
    X_adt = compute_adt_pca(adt)

    save_fixed_embedding(
        dataset=item["dataset"],
        pair=item["pair"],
        output_h5ad=item["output_h5ad"],
        obs_names=rna.obs_names.astype(str),
        celltype=get_celltype_obs(rna),
        fixed_blocks={
            "X_rna_pca": X_rna,
            "X_adt_pca": X_adt,
        },
    )


# ==============================================================================
# 8. RNA + ATAC
# ==============================================================================

def process_rna_atac(item):
    print("=" * 100)
    print("Processing RNA+ATAC:", item["dataset"])

    rna = sc.read_h5ad(item["rna_h5ad"])
    atac = sc.read_h5ad(item["atac_h5ad"])

    rna, atac = align_modalities([rna, atac])

    X_rna = compute_rna_pca(rna)
    X_atac = compute_atac_lsi(atac)

    save_fixed_embedding(
        dataset=item["dataset"],
        pair=item["pair"],
        output_h5ad=item["output_h5ad"],
        obs_names=rna.obs_names.astype(str),
        celltype=get_celltype_obs(rna),
        fixed_blocks={
            "X_rna_pca": X_rna,
            "X_atac_lsi": X_atac,
        },
    )


# ==============================================================================
# 9. main
# ==============================================================================

def main():
    for item in DATASETS_RNA_ADT:
        try:
            process_rna_adt(item)
        except Exception as e:
            print("FAILED RNA+ADT:", item["dataset"], e)

    for item in DATASETS_RNA_ATAC:
        try:
            process_rna_atac(item)
        except Exception as e:
            print("FAILED RNA+ATAC:", item["dataset"], e)


if __name__ == "__main__":
    main()