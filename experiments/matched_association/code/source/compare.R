setwd("/workspace/garq/Rworkspace/MetaQ/20260831/lasso/")

suppressPackageStartupMessages({
  library(Seurat)
  library(zellkonverter)
  library(SingleCellExperiment)
  library(Matrix)
  library(tidyverse)
  library(ComplexHeatmap)
  library(circlize)
  library(glmnet)
  library(grid)
})

# ==============================================================================
# 1. 输入路径
# ==============================================================================

algorithm_inputs <- list(
  Supercell = list(
    rna_h5ad = "/workspace/garq/results/5_supercell/three/GSE158013/GSE158013_rna_metacells.h5ad"
  ),
  SEACell = list(
    rna_h5ad = "/workspace/garq/results/3_seacell/three/GSE158013/GSE158013_rna_metacells.h5ad"
  ),
  MetaCellV2 = list(
    rna_h5ad = "/workspace/garq/results/4_mc2/three/GSE158013/GSE158013_rna_metacells.h5ad"
  )
)

# 这两个文件来自你原始 metacell 图
REFERENCE_COEF_CSV <- "/workspace/garq/Rworkspace/MetaQ/three/compare/15_seacell/celltype_hvg_lasso_coefficients.csv"
REFERENCE_MATRIX_RDS <- "/workspace/garq/Rworkspace/MetaQ/three/compare/15_seacell/celltype_hvg_lasso_matrix.rds"

base_outdir <- "lasso_heatmap_three_algorithms_aligned_to_original_metacell"
dir.create(base_outdir, showWarnings = FALSE, recursive = TRUE)

# ==============================================================================
# 2. 从原 metacell 图读取固定 gene / TF 顺序
# ==============================================================================

sig_coefs_ref <- read.csv(REFERENCE_COEF_CSV, stringsAsFactors = FALSE)
mc_lasso_matrix <- readRDS(REFERENCE_MATRIX_RDS)

max_val_mc <- max(abs(mc_lasso_matrix), na.rm = TRUE)
if (max_val_mc < 0.01) {
  max_val_mc <- 0.5
}

col_fun_mc <- colorRamp2(
  c(-max_val_mc, -max_val_mc / 2, 0, max_val_mc / 2, max_val_mc),
  c("#2166AC", "#92C5DE", "#F7F7F7", "#F4A582", "#B2182B")
)

ht_mc_temp <- Heatmap(
  mc_lasso_matrix,
  col = col_fun_mc,
  cluster_rows = TRUE,
  cluster_columns = TRUE,
  show_heatmap_legend = FALSE
)

ht_mc_draw <- draw(ht_mc_temp)

mc_genes_ordered <- rownames(mc_lasso_matrix)[row_order(ht_mc_draw)]
mc_tfs_ordered <- colnames(mc_lasso_matrix)[column_order(ht_mc_draw)]

cat("Reference ordered genes:", length(mc_genes_ordered), "\n")
cat("Reference ordered TFs:", length(mc_tfs_ordered), "\n")
print(mc_genes_ordered)
print(mc_tfs_ordered)

write.csv(
  data.frame(gene = mc_genes_ordered),
  file.path(base_outdir, "ordered_genes_from_original_metacell.csv"),
  row.names = FALSE
)

write.csv(
  data.frame(tf = mc_tfs_ordered),
  file.path(base_outdir, "ordered_tfs_from_original_metacell.csv"),
  row.names = FALSE
)

# ==============================================================================
# 3. h5ad -> Seurat
# ==============================================================================

safe_method_name <- function(x) {
  x <- gsub("[^A-Za-z0-9_\\-]+", "_", x)
  x <- gsub("_+", "_", x)
  x
}

read_h5ad_as_seurat <- function(h5ad_path, method_name) {
  cat("\nReading h5ad:", h5ad_path, "\n")
  
  sce <- readH5AD(h5ad_path)
  
  if (!"X" %in% assayNames(sce)) {
    stop(method_name, ": h5ad 中没有 X assay")
  }
  
  mat <- assay(sce, "X")
  meta <- as.data.frame(colData(sce))
  
  if (is.null(rownames(mat))) {
    if ("name" %in% colnames(as.data.frame(rowData(sce)))) {
      rownames(mat) <- rowData(sce)$name
    } else {
      rownames(mat) <- paste0("gene_", seq_len(nrow(mat)))
    }
  }
  
  if (is.null(colnames(mat))) {
    colnames(mat) <- colnames(sce)
  }
  
  rownames(mat) <- make.unique(as.character(rownames(mat)))
  colnames(mat) <- make.unique(as.character(colnames(mat)))
  
  if (!"celltype" %in% colnames(meta)) {
    meta$celltype <- "Unknown"
  }
  
  rownames(meta) <- colnames(mat)
  
  seu <- CreateSeuratObject(
    counts = mat,
    assay = "RNA",
    meta.data = meta,
    project = method_name
  )
  
  seu$celltype <- as.character(seu$celltype)
  
  seu <- NormalizeData(
    seu,
    assay = "RNA",
    normalization.method = "LogNormalize",
    scale.factor = 10000,
    verbose = FALSE
  )
  
  cat("Created Seurat object:", nrow(seu), "genes ×", ncol(seu), "metacells\n")
  print(table(seu$celltype))
  
  return(seu)
}

get_expr_matrix <- function(seurat_obj) {
  expr_matrix <- tryCatch(
    GetAssayData(seurat_obj, assay = "RNA", layer = "data"),
    error = function(e) GetAssayData(seurat_obj, assay = "RNA", slot = "data")
  )
  
  if (inherits(expr_matrix, "sparseMatrix")) {
    expr_matrix <- as.matrix(expr_matrix)
  }
  
  return(expr_matrix)
}

# ==============================================================================
# 4. 固定 gene × TF 的 LASSO
# ==============================================================================

run_lasso_for_gene <- function(gene, tfs, expr_mat) {
  avail_tfs <- intersect(tfs, rownames(expr_mat))
  
  if (!(gene %in% rownames(expr_mat)) || length(avail_tfs) == 0) {
    return(data.frame(
      gene = gene,
      tf = tfs,
      coefficient = 0,
      lambda = NA,
      stringsAsFactors = FALSE
    ))
  }
  
  y <- expr_mat[gene, ]
  
  if (sd(y) < 1e-6) {
    return(data.frame(
      gene = gene,
      tf = tfs,
      coefficient = 0,
      lambda = NA,
      stringsAsFactors = FALSE
    ))
  }
  
  X <- t(expr_mat[avail_tfs, , drop = FALSE])
  tf_sds <- apply(X, 2, sd)
  X_valid <- X[, tf_sds > 1e-6, drop = FALSE]
  
  if (ncol(X_valid) == 0) {
    return(data.frame(
      gene = gene,
      tf = tfs,
      coefficient = 0,
      lambda = NA,
      stringsAsFactors = FALSE
    ))
  }
  
  X_valid <- scale(X_valid)
  y <- scale(y)
  
  tryCatch({
    cv_fit <- cv.glmnet(
      X_valid,
      as.vector(y),
      alpha = 1,
      nfolds = 5,
      standardize = FALSE,
      parallel = FALSE
    )
    
    coefs <- as.vector(coef(cv_fit, s = cv_fit$lambda.1se))[-1]
    names(coefs) <- colnames(X_valid)
    
    all_coefs <- rep(0, length(tfs))
    names(all_coefs) <- tfs
    all_coefs[names(coefs)] <- coefs
    
    data.frame(
      gene = gene,
      tf = names(all_coefs),
      coefficient = all_coefs,
      lambda = cv_fit$lambda.1se,
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    data.frame(
      gene = gene,
      tf = tfs,
      coefficient = 0,
      lambda = NA,
      stringsAsFactors = FALSE
    )
  })
}

run_lasso_for_fixed_matrix <- function(expr_matrix, genes_order, tfs_order) {
  results <- list()
  
  pb <- txtProgressBar(min = 0, max = length(genes_order), style = 3)
  
  for (i in seq_along(genes_order)) {
    gene <- genes_order[i]
    
    res <- run_lasso_for_gene(
      gene = gene,
      tfs = tfs_order,
      expr_mat = expr_matrix
    )
    
    results[[gene]] <- res
    setTxtProgressBar(pb, i)
  }
  
  close(pb)
  
  dplyr::bind_rows(results)
}

make_fixed_lasso_matrix <- function(lasso_results, genes_order, tfs_order) {
  mat <- lasso_results %>%
    dplyr::select(gene, tf, coefficient) %>%
    tidyr::pivot_wider(
      names_from = tf,
      values_from = coefficient,
      values_fill = 0
    ) %>%
    tibble::column_to_rownames("gene") %>%
    as.matrix()
  
  missing_genes <- setdiff(genes_order, rownames(mat))
  if (length(missing_genes) > 0) {
    mat <- rbind(
      mat,
      matrix(
        0,
        nrow = length(missing_genes),
        ncol = ncol(mat),
        dimnames = list(missing_genes, colnames(mat))
      )
    )
  }
  
  missing_tfs <- setdiff(tfs_order, colnames(mat))
  if (length(missing_tfs) > 0) {
    mat <- cbind(
      mat,
      matrix(
        0,
        nrow = nrow(mat),
        ncol = length(missing_tfs),
        dimnames = list(rownames(mat), missing_tfs)
      )
    )
  }
  
  mat <- mat[genes_order, tfs_order, drop = FALSE]
  
  return(mat)
}

# ==============================================================================
# 5. 计算三个算法的固定矩阵
# ==============================================================================

all_matrices <- list()
all_summaries <- list()

for (method_name in names(algorithm_inputs)) {
  cat("\n", strrep("=", 80), "\n", sep = "")
  cat("Running fixed-order LASSO for: ", method_name, "\n", sep = "")
  cat(strrep("=", 80), "\n", sep = "")
  
  method_outdir <- file.path(base_outdir, safe_method_name(method_name))
  dir.create(method_outdir, showWarnings = FALSE, recursive = TRUE)
  
  seu <- read_h5ad_as_seurat(
    h5ad_path = algorithm_inputs[[method_name]]$rna_h5ad,
    method_name = method_name
  )
  
  saveRDS(seu, file.path(method_outdir, "seurat_mc_processed_celltype.rds"))
  
  expr_matrix <- get_expr_matrix(seu)
  
  missing_genes <- setdiff(mc_genes_ordered, rownames(expr_matrix))
  missing_tfs <- setdiff(mc_tfs_ordered, rownames(expr_matrix))
  
  if (length(missing_genes) > 0) {
    cat("Missing genes in", method_name, ":\n")
    print(missing_genes)
  }
  
  if (length(missing_tfs) > 0) {
    cat("Missing TFs in", method_name, ":\n")
    print(missing_tfs)
  }
  
  lasso_results <- run_lasso_for_fixed_matrix(
    expr_matrix = expr_matrix,
    genes_order = mc_genes_ordered,
    tfs_order = mc_tfs_ordered
  )
  
  write.csv(
    lasso_results,
    file.path(method_outdir, "fixed_metacell_order_lasso_coefficients.csv"),
    row.names = FALSE
  )
  
  mat <- make_fixed_lasso_matrix(
    lasso_results = lasso_results,
    genes_order = mc_genes_ordered,
    tfs_order = mc_tfs_ordered
  )
  
  all_matrices[[method_name]] <- mat
  
  saveRDS(
    mat,
    file.path(method_outdir, "fixed_metacell_order_lasso_matrix.rds")
  )
  
  all_summaries[[method_name]] <- data.frame(
    method = method_name,
    n_metacells = ncol(seu),
    n_genes_total = nrow(seu),
    n_fixed_genes = length(mc_genes_ordered),
    n_fixed_tfs = length(mc_tfs_ordered),
    n_missing_genes = length(missing_genes),
    n_missing_tfs = length(missing_tfs),
    n_nonzero = sum(abs(mat) > 1e-6),
    stringsAsFactors = FALSE
  )
}

summary_df <- dplyr::bind_rows(all_summaries)

write.csv(
  summary_df,
  file.path(base_outdir, "three_algorithms_fixed_metacell_order_summary.csv"),
  row.names = FALSE
)

# ==============================================================================
# 6. 画三个算法并排热图
# ==============================================================================

plot_three_algorithm_heatmap <- function(matrices, genes_order, tfs_order) {
  all_values <- unlist(lapply(matrices, as.vector))
  max_val <- max(abs(all_values), na.rm = TRUE)
  
  if (max_val < 0.01) {
    max_val <- 0.5
  }
  
  max_val <- max(1, max_val)
  
  col_fun <- colorRamp2(
    c(-max_val, -max_val / 2, 0, max_val / 2, max_val),
    c("#2166AC", "#92C5DE", "#F7F7F7", "#F4A582", "#B2182B")
  )
  
  heatmaps <- list()
  method_names <- names(matrices)
  
  for (i in seq_along(method_names)) {
    method_name <- method_names[i]
    mat <- matrices[[method_name]]
    
    top_ha <- HeatmapAnnotation(
      `N targets` = anno_barplot(
        colSums(abs(mat) > 1e-6),
        gp = gpar(fill = "#4393C3"),
        height = unit(1.4, "cm")
      ),
      annotation_name_side = "left",
      annotation_name_gp = gpar(fontsize = 8)
    )
    
    if (i == 1) {
      heatmaps[[i]] <- Heatmap(
        mat,
        name = "LASSO\nbeta",
        col = col_fun,
        cluster_rows = FALSE,
        cluster_columns = FALSE,
        show_row_names = TRUE,
        show_column_names = TRUE,
        row_names_side = "left",
        row_names_gp = gpar(fontsize = 8),
        column_names_gp = gpar(fontsize = 9),
        column_names_rot = 45,
        top_annotation = top_ha,
        row_title = "Target genes",
        column_title = method_name,
        width = unit(max(4, length(tfs_order) * 0.7), "cm"),
        height = unit(max(10, length(genes_order) * 0.38), "cm"),
        border = TRUE,
        heatmap_legend_param = list(
          title = "LASSO beta",
          title_gp = gpar(fontsize = 10, fontface = "bold"),
          labels_gp = gpar(fontsize = 8),
          legend_height = unit(4, "cm")
        )
      )
    } else {
      heatmaps[[i]] <- Heatmap(
        mat,
        name = paste0("LASSO\nbeta\n", method_name),
        col = col_fun,
        cluster_rows = FALSE,
        cluster_columns = FALSE,
        show_row_names = FALSE,
        show_column_names = TRUE,
        column_names_gp = gpar(fontsize = 9),
        column_names_rot = 45,
        top_annotation = top_ha,
        column_title = method_name,
        width = unit(max(4, length(tfs_order) * 0.7), "cm"),
        height = unit(max(10, length(genes_order) * 0.38), "cm"),
        border = TRUE,
        show_heatmap_legend = FALSE
      )
    }
  }
  
  ht_list <- heatmaps[[1]]
  if (length(heatmaps) > 1) {
    for (i in 2:length(heatmaps)) {
      ht_list <- ht_list + heatmaps[[i]]
    }
  }
  
  pdf_width <- max(16, length(matrices) * max(4, length(tfs_order) * 0.7) + 7)
  pdf_height <- max(13, length(genes_order) * 0.38 + 4)
  
  pdf(
    file.path(base_outdir, "three_algorithms_aligned_to_original_metacell_heatmap.pdf"),
    width = pdf_width,
    height = pdf_height
  )
  
  draw(
    ht_list,
    column_title = paste0(
      "TF-Gene regulatory programs across algorithms\n",
      "Same TF-gene set and clustered order as original metacell heatmap"
    ),
    column_title_gp = gpar(fontsize = 15, fontface = "bold"),
    heatmap_legend_side = "right",
    padding = unit(c(2, 2, 10, 2), "mm")
  )
  
  dev.off()
  
  cat(
    "Saved:",
    file.path(base_outdir, "three_algorithms_aligned_to_original_metacell_heatmap.pdf"),
    "\n"
  )
}

plot_three_algorithm_heatmap(
  matrices = all_matrices,
  genes_order = mc_genes_ordered,
  tfs_order = mc_tfs_ordered
)

cat("\nDone.\n")
cat("Output dir:", normalizePath(base_outdir), "\n")
print(summary_df)

