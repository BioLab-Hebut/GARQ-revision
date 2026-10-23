library(Seurat)
library(tidyverse)
library(ComplexHeatmap)
library(circlize)
library(glmnet)
library(ggplot2)

# ============================================================================
# 基于细胞类型高变基因的LASSO回归分析
# ============================================================================

cat("=== 基于细胞类型高变基因的LASSO分析 ===\n\n")

# ============================================================================
# Step 1: 加载数据
# ============================================================================

cat("Step 1: 加载数据...\n")

seurat_mc <- readRDS("intermediate_data/seurat_mc_processed_celltype.rds")

cat("Metacell对象加载成功\n")
cat("总metacell数:", ncol(seurat_mc), "\n")
cat("总基因数:", nrow(seurat_mc), "\n")

# 检查细胞类型信息
if (!"celltype" %in% colnames(seurat_mc@meta.data)) {
  stop("❌ 错误：Seurat对象中没有 'celltype' 列！请检查元数据。")
}

cat("可用的细胞类型:\n")
print(table(seurat_mc$celltype))

# ============================================================================
# Step 2: 指定要分析的细胞类型（★ 配置项）
# ============================================================================

cat("\nStep 2: 指定要分析的细胞类型...\n")

# ★ 你指定的6个细胞类型
target_celltypes <- c(
  
  'T.CD4.Naive',
  'T.CD4.Memory',
  'T.CD8.Naive',
  'T.CD8.Effector'
)

cat("指定的细胞类型:\n")
for (ct in target_celltypes) {
  cat("  •", ct, "\n")
}

# 检查这些细胞类型是否存在
available_celltypes <- intersect(target_celltypes, unique(seurat_mc$celltype))

cat("\n在数据中可用的细胞类型:\n")
for (ct in available_celltypes) {
  n_cells <- sum(seurat_mc$celltype == ct)
  cat(sprintf("  • %s: %d个metacells\n", ct, n_cells))
}

missing_celltypes <- setdiff(target_celltypes, available_celltypes)
if (length(missing_celltypes) > 0) {
  cat("\n⚠️  缺失的细胞类型:\n")
  for (ct in missing_celltypes) {
    cat("  •", ct, "\n")
  }
}

if (length(available_celltypes) == 0) {
  stop("❌ 错误：没有可用的细胞类型！请检查celltype名称。")
}

# ============================================================================
# Step 3: 提取每个细胞类型的高变基因
# ============================================================================

cat("\nStep 3: 计算每个细胞类型的高变基因...\n")

# 配置参数
n_hvg_per_celltype <- 15  # ★ 每个细胞类型取top N高变基因（可调整）

# 提取表达矩阵
expr_matrix <- GetAssayData(seurat_mc, assay = "RNA", slot = "data")
if (inherits(expr_matrix, "sparseMatrix")) {
  expr_matrix <- as.matrix(expr_matrix)
}

# 对每个细胞类型找高变基因
hvg_list <- list()

for (celltype in available_celltypes) {
  
  cat("\n处理", celltype, "...\n")
  
  # 提取该细胞类型的cells
  cells_in_celltype <- colnames(seurat_mc)[seurat_mc$celltype == celltype]
  
  if (length(cells_in_celltype) < 3) {
    cat("  ⚠️  细胞数太少，跳过\n")
    next
  }
  
  # 提取该细胞类型的表达矩阵
  expr_subset <- expr_matrix[, cells_in_celltype, drop = FALSE]
  
  # 计算基因的变异系数（CV）或方差
  gene_vars <- apply(expr_subset, 1, var)
  gene_means <- rowMeans(expr_subset)
  
  # 只考虑有表达的基因（mean > 0.1）
  expressed_genes <- names(gene_means)[gene_means > 0.1]
  gene_vars <- gene_vars[expressed_genes]
  
  # 选择top N高变基因
  if (length(gene_vars) > 0) {
    top_hvgs <- names(sort(gene_vars, decreasing = TRUE)[1:min(n_hvg_per_celltype, length(gene_vars))])
    hvg_list[[celltype]] <- top_hvgs
    
    cat("  • Top", length(top_hvgs), "高变基因:", 
        paste(head(top_hvgs, 5), collapse = ", "), "...\n")
  } else {
    cat("  ⚠️  没有符合条件的基因\n")
  }
}

# 合并所有高变基因（去重）
all_hvgs <- unique(unlist(hvg_list))

cat("\n总结:\n")
cat("  • 分析的细胞类型:", length(hvg_list), "\n")
cat("  • 每类型高变基因数:", n_hvg_per_celltype, "\n")
cat("  • 总高变基因数（去重后）:", length(all_hvgs), "\n")

# 保存高变基因列表
hvg_summary <- data.frame(
  celltype = rep(names(hvg_list), sapply(hvg_list, length)),
  gene = unlist(hvg_list),
  stringsAsFactors = FALSE
)
write.csv(hvg_summary, "15_seacell/celltype_hvg_list.csv", row.names = FALSE)

cat("\n✓ 高变基因列表已保存\n")

# ============================================================================
# Step 4: 选择候选TF（修正版）
# ============================================================================

cat("\nStep 4: 选择候选TF...\n")



cat("⚠️  未找到tf_target_matrix_mc.rds，使用已知TF列表...\n")

# ★★★ 策略B：使用已知的免疫相关TF列表 ★★★
known_tfs <- c(
  "TCF7",
  "RUNX3",
  "STAT3",
  "TBX21",
  "RORC",
  "GATA3",
  "IRF4",
  "BATF",
  "ETS1",
  "BACH2"
)

cat("  • 已知TF列表:", length(known_tfs), "个\n")

# 筛选在表达矩阵中存在的
candidate_tfs <- intersect(known_tfs, rownames(expr_matrix))
cat("  • 在表达矩阵中可用:", length(candidate_tfs), "个\n")


# 移除与高变基因重叠的（避免基因调控自己）
candidate_tfs <- setdiff(candidate_tfs, all_hvgs)

# ★★★ 重要：过滤掉非TF基因（黑名单）★★★
non_tf_blacklist <- c(
  # 免疫球蛋白基因（不是TF！）
  "IGHM", "IGHD", "IGHA1", "IGHA2", "IGHG1", "IGHG2", "IGHG3", "IGHG4", "IGHE",
  "IGLC1", "IGLC2", "IGLC3", "IGLC4", "IGLC5", "IGLC6", "IGLC7",
  "IGKC", "IGKV", "IGLV", "kappa",  # 统称
  
  # T细胞受体基因（不是TF！）
  "TRAC", "TRBC1", "TRBC2", "TRDC", "TRGC1", "TRGC2",
  "TRAV", "TRBV", "TRDV", "TRGV",
  
  # 管家基因（不是TF！）
  "ACTB", "GAPDH", "B2M", "PPIA", "HPRT1", "TBP",
  "RPL", "RPS",  # 核糖体蛋白（前缀）
  
  # 细胞表面标志（不是TF！）
  "CD79A", "CD79B", "MS4A1", "CD19", "CD3D", "CD3E", "CD3G",
  "CD4", "CD8A", "CD8B", "PTPRC", "CD74",
  
  # HLA基因（不是TF！）
  "HLA-A", "HLA-B", "HLA-C", "HLA-DRA", "HLA-DRB1", "HLA-DQA1", "HLA-DQB1",
  
  # 线粒体基因（不是TF！）
  "MT-", "MTRNR"  # 前缀
)

# 过滤掉黑名单基因
before_filter <- length(candidate_tfs)

# 精确匹配
candidate_tfs <- setdiff(candidate_tfs, non_tf_blacklist)

# 前缀匹配（如RPL, RPS, MT-等）
prefix_patterns <- c("^RPL", "^RPS", "^MT-", "^MTRNR", "^IGH", "^IGL", "^IGK", "^TRA", "^TRB", "^TRD", "^TRG")
for (pattern in prefix_patterns) {
  candidate_tfs <- candidate_tfs[!grepl(pattern, candidate_tfs)]
}

after_filter <- length(candidate_tfs)

cat("\n✓ 黑名单过滤:\n")
cat("  • 过滤前:", before_filter, "个候选TF\n")
cat("  • 过滤后:", after_filter, "个候选TF\n")
if (before_filter > after_filter) {
  cat("  • 移除了:", before_filter - after_filter, "个非TF基因\n")
}

cat("\n✓ 最终候选TF数:", length(candidate_tfs), "\n")
cat("  候选TF示例:", paste(head(candidate_tfs, 10), collapse = ", "), "...\n")

if (length(candidate_tfs) < 5) {
  stop("❌ 错误：候选TF太少（<5个）！请检查数据或TF列表。")
}

# ============================================================================
# Step 5: 运行LASSO回归
# ============================================================================

cat("\nStep 5: 对每个高变基因运行LASSO回归...\n")

run_lasso_for_gene <- function(gene, tfs, expr_mat) {
  
  # 提取基因表达
  y <- expr_mat[gene, ]
  
  # 检查变异性
  if (sd(y) < 1e-6) {
    return(NULL)
  }
  
  # 提取TF表达矩阵
  available_tfs <- intersect(tfs, rownames(expr_mat))
  if (length(available_tfs) < 3) {
    return(NULL)
  }
  
  X <- t(expr_mat[available_tfs, , drop = FALSE])
  
  # 移除无变异的TF
  tf_sds <- apply(X, 2, sd)
  valid_tf_idx <- tf_sds > 1e-6
  X <- X[, valid_tf_idx, drop = FALSE]
  
  if (ncol(X) < 3) {
    return(NULL)
  }
  
  # 标准化
  X <- scale(X)
  y <- scale(y)
  
  # LASSO回归
  tryCatch({
    cv_fit <- cv.glmnet(
      x = X,
      y = as.vector(y),
      alpha = 1,
      nfolds = 5,
      standardize = FALSE,
      parallel = FALSE
    )
    
    best_lambda <- cv_fit$lambda.1se
    
    coefs <- coef(cv_fit, s = best_lambda)
    coefs <- as.vector(coefs)[-1]
    names(coefs) <- colnames(X)
    
    result <- data.frame(
      gene = gene,
      tf = names(coefs),
      coefficient = coefs,
      lambda = best_lambda,
      stringsAsFactors = FALSE
    )
    
    return(result)
    
  }, error = function(e) {
    return(NULL)
  })
}

# 检查高变基因是否在表达矩阵中
available_hvgs <- intersect(all_hvgs, rownames(expr_matrix))
cat("可分析的高变基因:", length(available_hvgs), "/", length(all_hvgs), "\n")

# 运行LASSO
lasso_results <- list()

pb <- txtProgressBar(min = 0, max = length(available_hvgs), style = 3)

for (i in seq_along(available_hvgs)) {
  gene <- available_hvgs[i]
  result <- run_lasso_for_gene(gene, candidate_tfs, expr_matrix)
  
  if (!is.null(result)) {
    lasso_results[[gene]] <- result
  }
  
  setTxtProgressBar(pb, i)
}

close(pb)

# 合并结果
lasso_results <- bind_rows(lasso_results)

cat("\n\n✓ LASSO回归完成！\n")
cat("  • 总TF-gene对:", nrow(lasso_results), "\n")

# ============================================================================
# Step 6: 筛选显著系数
# ============================================================================

cat("\nStep 6: 筛选显著系数...\n")

# 系数统计
cat("\nLASSO系数统计:\n")
print(summary(lasso_results$coefficient))

# 筛选阈值
coef_threshold <- 0.1
sig_coefs <- lasso_results %>%
  filter(abs(coefficient) > coef_threshold)

cat("\n显著系数 (|β| >", coef_threshold, "):", nrow(sig_coefs), "\n")

if (nrow(sig_coefs) == 0) {
  cat("⚠️  没有显著系数，降低阈值到 0.05\n")
  coef_threshold <- 0.05
  sig_coefs <- lasso_results %>%
    filter(abs(coefficient) > coef_threshold)
  cat("显著系数 (|β| >", coef_threshold, "):", nrow(sig_coefs), "\n")
}

if (nrow(sig_coefs) == 0) {
  stop("❌ 没有显著的调控关系，请检查数据或降低阈值。")
}

# TF统计
tf_stats <- sig_coefs %>%
  group_by(tf) %>%
  summarise(
    n_targets = n(),
    n_activated = sum(coefficient > coef_threshold),
    n_repressed = sum(coefficient < -coef_threshold),
    mean_abs_coef = mean(abs(coefficient)),
    .groups = "drop"
  ) %>%
  arrange(desc(n_targets))

cat("\n=== TF调控统计 (Top 10) ===\n")
print(head(tf_stats, 10))

# 基因统计（包含细胞类型信息）
gene_stats <- sig_coefs %>%
  left_join(hvg_summary, by = "gene") %>%
  group_by(gene) %>%
  summarise(
    celltypes = paste(unique(celltype), collapse = ", "),
    n_regulators = n(),
    mean_abs_coef = mean(abs(coefficient)),
    top_tf = tf[which.max(abs(coefficient))],
    .groups = "drop"
  ) %>%
  arrange(desc(mean_abs_coef))

cat("\n=== 基因调控统计 (Top 10) ===\n")
print(head(gene_stats, 10))

# 保存结果
write.csv(sig_coefs, "15_seacell/celltype_hvg_lasso_coefficients.csv", row.names = FALSE)
write.csv(tf_stats, "15_seacell/celltype_hvg_tf_stats.csv", row.names = FALSE)
write.csv(gene_stats, "15_seacell/celltype_hvg_gene_stats.csv", row.names = FALSE)

# ============================================================================
# Step 7: 构建热图矩阵
# ============================================================================

cat("\nStep 7: 构建热图矩阵...\n")

# 转换为矩阵
#lasso_matrix <- sig_coefs %>%
#  pivot_wider(
#    names_from = tf,
#    values_from = coefficient,
#    values_fill = 0
#  ) %>%
#  column_to_rownames("gene") %>%
#  as.matrix()

#cat("热图矩阵:", dim(lasso_matrix), "\n")




lasso_matrix <- sig_coefs %>%
  select(gene, tf, coefficient) %>%   # 👈 关键
  pivot_wider(
    names_from  = tf,
    values_from = coefficient,
    values_fill = 0
  ) %>%
  column_to_rownames("gene") %>%
  as.matrix()

cat("热图矩阵:", dim(lasso_matrix), "\n")





# 限制大小（如果太大）
max_genes_display <- 50
max_tfs_display <- 15

if (nrow(lasso_matrix) > max_genes_display) {
  gene_importance <- apply(abs(lasso_matrix), 1, max)
  top_genes <- names(sort(gene_importance, decreasing = TRUE)[1:max_genes_display])
  lasso_matrix <- lasso_matrix[top_genes, , drop = FALSE]
  cat("限制到前", max_genes_display, "个基因\n")
}

if (ncol(lasso_matrix) > max_tfs_display) {
  tf_importance <- apply(abs(lasso_matrix), 2, sum)
  top_tfs <- names(sort(tf_importance, decreasing = TRUE)[1:max_tfs_display])
  lasso_matrix <- lasso_matrix[, top_tfs, drop = FALSE]
  cat("限制到前", max_tfs_display, "个TF\n")
}

saveRDS(lasso_matrix, "15_seacell/celltype_hvg_lasso_matrix.rds")

# ============================================================================
# Step 8: 绘制热图
# ============================================================================

cat("\nStep 8: 绘制热图...\n")

# 配色
max_val <- max(abs(lasso_matrix))
col_fun <- colorRamp2(
  c(-max_val, -max_val/2, 0, max_val/2, max_val),
  c("#2166AC", "#92C5DE", "#F7F7F7", "#F4A582", "#B2182B")
)

# 添加基因的细胞类型注释
gene_celltype_anno <- gene_stats %>%
  filter(gene %in% rownames(lasso_matrix)) %>%
  select(gene, celltypes) %>%
  mutate(
    n_celltypes = str_count(celltypes, ",") + 1
  )

# 匹配行名顺序
gene_celltype_anno <- gene_celltype_anno[match(rownames(lasso_matrix), gene_celltype_anno$gene), ]

# 为细胞类型分配颜色
all_celltypes_in_data <- unique(unlist(strsplit(gene_celltype_anno$celltypes, ", ")))
celltype_colors <- setNames(
  rainbow(length(all_celltypes_in_data)),
  all_celltypes_in_data
)

# 行注释（左侧）
row_ha_left <- rowAnnotation(
  `Cell types` = anno_text(
    gene_celltype_anno$celltypes,
    gp = gpar(fontsize = 6),
    location = 0,
    just = "left"
  ),
  annotation_name_gp = gpar(fontsize = 9),
  show_annotation_name = TRUE
)

# 行注释（右侧）
row_ha_right <- rowAnnotation(
  `N TFs` = anno_barplot(
    rowSums(abs(lasso_matrix) > coef_threshold),
    gp = gpar(fill = "#D6604D"),
    width = unit(2, "cm")
  ),
  annotation_name_gp = gpar(fontsize = 9)
)

# 列注释
column_ha <- HeatmapAnnotation(
  `N targets` = anno_barplot(
    colSums(abs(lasso_matrix) > coef_threshold),
    gp = gpar(fill = "#4393C3"),
    height = unit(2, "cm")
  ),
  annotation_name_side = "left",
  annotation_name_gp = gpar(fontsize = 9)
)

# 绘制热图
ht <- Heatmap(
  lasso_matrix,
  name = "LASSO\nCoefficient",
  col = col_fun,
  
  cluster_rows = TRUE,
  cluster_columns = TRUE,
  show_row_names = TRUE,
  show_column_names = TRUE,
  row_names_gp = gpar(fontsize = 8),
  column_names_gp = gpar(fontsize = 10),
  column_names_rot = 45,
  
  left_annotation = row_ha_left,
  right_annotation = row_ha_right,
  top_annotation = column_ha,
  
  row_title = paste0("High Variable Genes (n=", nrow(lasso_matrix), ")"),
  column_title = paste0("Transcription Factors (n=", ncol(lasso_matrix), ")"),
  
  width = unit(max(8, ncol(lasso_matrix) * 1.2), "cm"),
  height = unit(max(12, nrow(lasso_matrix) * 0.4), "cm"),
  border = TRUE,
  
  heatmap_legend_param = list(
    title = "LASSO β",
    title_gp = gpar(fontsize = 11, fontface = "bold"),
    labels_gp = gpar(fontsize = 9),
    legend_height = unit(4, "cm")
  )
)

# 保存PDF
pdf("15_seacell/celltype_hvg_lasso_heatmap.pdf", 
    width = max(12, ncol(lasso_matrix) * 1 + 6), 
    height = max(14, nrow(lasso_matrix) * 0.35 + 4))
draw(
  ht,
  column_title = paste0("TF-Gene Network (Top HVGs from ", 
                        length(available_celltypes), " Cell Types)"),
  column_title_gp = gpar(fontsize = 16, fontface = "bold"),
  heatmap_legend_side = "right",
  padding = unit(c(2, 2, 10, 2), "mm")
)
dev.off()

# 保存PNG
png("15_seacell/celltype_hvg_lasso_heatmap.png", 
    width = max(12, ncol(lasso_matrix) * 1 + 6), 
    height = max(14, nrow(lasso_matrix) * 0.35 + 4), 
    units = "in", res = 300)
draw(
  ht,
  column_title = paste0("TF-Gene Network (Top HVGs from ", 
                        length(available_celltypes), " Cell Types)"),
  column_title_gp = gpar(fontsize = 16, fontface = "bold"),
  heatmap_legend_side = "right",
  padding = unit(c(2, 2, 10, 2), "mm")
)
dev.off()

cat("✓ 热图已保存\n")

# ============================================================================
# Step 9: 按细胞类型分析
# ============================================================================

cat("\nStep 9: 按细胞类型统计分析...\n")

celltype_analysis <- sig_coefs %>%
  left_join(hvg_summary, by = "gene") %>%
  group_by(celltype) %>%
  summarise(
    n_genes = n_distinct(gene),
    n_tfs = n_distinct(tf),
    n_regulations = n(),
    mean_abs_coef = mean(abs(coefficient)),
    top_gene = gene[which.max(abs(coefficient))],
    top_tf = tf[which.max(abs(coefficient))],
    .groups = "drop"
  ) %>%
  arrange(desc(n_regulations))

cat("\n=== 按细胞类型统计 ===\n")
print(celltype_analysis)

write.csv(celltype_analysis, "15_seacell/celltype_regulation_summary.csv", row.names = FALSE)

# 绘制细胞类型比较图
if (nrow(celltype_analysis) > 0) {
  
  p_celltype <- ggplot(celltype_analysis, 
                       aes(x = reorder(celltype, n_regulations), 
                           y = n_regulations)) +
    geom_bar(stat = "identity", fill = "#4393C3") +
    geom_text(aes(label = n_regulations), hjust = -0.2, size = 4) +
    coord_flip() +
    labs(
      title = "Number of Significant TF-Gene Regulations by Cell Type",
      x = "Cell Type",
      y = "Number of Regulations"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      axis.text = element_text(size = 10),
      axis.title = element_text(size = 11, face = "bold")
    )
  
  ggsave("15_seacell/celltype_regulation_count.pdf", p_celltype, 
         width = 8, height = 6)
  
  cat("✓ 细胞类型比较图已保存\n")
}

# ============================================================================
# Step 10: 总结报告
# ============================================================================

cat("\n", rep("=", 80), "\n")
cat("基于细胞类型高变基因的LASSO分析完成！\n")
cat(rep("=", 80), "\n\n")

cat("📊 分析总结:\n")
cat("  • 指定的细胞类型:", length(target_celltypes), "\n")
cat("  • 实际分析的细胞类型:", length(available_celltypes), "\n")
cat("  • 每类型高变基因数:", n_hvg_per_celltype, "\n")
cat("  • 总高变基因数（去重）:", length(all_hvgs), "\n")
cat("  • 候选TF数:", length(candidate_tfs), "\n")
cat("  • 显著调控关系 (|β| >", coef_threshold, "):", nrow(sig_coefs), "\n\n")

cat("🧬 细胞类型列表:\n")
for (ct in available_celltypes) {
  n_genes <- sum(gene_stats$celltypes == ct | grepl(ct, gene_stats$celltypes))
  cat(sprintf("  • %s: %d个高变基因\n", ct, n_genes))
}

if (nrow(tf_stats) > 0) {
  cat("\n🎯 Top 5 最活跃的TF:\n")
  for (i in 1:min(5, nrow(tf_stats))) {
    cat(sprintf("  %d. %s: 调控%d个基因 (激活:%d, 抑制:%d)\n",
                i,
                tf_stats$tf[i],
                tf_stats$n_targets[i],
                tf_stats$n_activated[i],
                tf_stats$n_repressed[i]))
  }
}

cat("\n📁 生成的文件:\n")
cat("  1. celltype_hvg_lasso_heatmap.pdf           - LASSO系数热图（主图）\n")
cat("  2. celltype_hvg_list.csv                    - 每个细胞类型的高变基因\n")
cat("  3. celltype_hvg_lasso_coefficients.csv      - LASSO系数详细表\n")
cat("  4. celltype_hvg_tf_stats.csv                - TF统计\n")
cat("  5. celltype_hvg_gene_stats.csv              - 基因统计（含细胞类型）\n")
cat("  6. celltype_regulation_summary.csv          - 按细胞类型统计\n")
cat("  7. celltype_regulation_count.pdf            - 细胞类型比较图\n")
cat("  8. celltype_hvg_lasso_matrix.rds            - LASSO系数矩阵\n\n")

cat("💡 热图解读:\n")
cat("  • 红色 = TF激活基因\n")
cat("  • 蓝色 = TF抑制基因\n")
cat("  • 左侧文字 = 基因来自哪些细胞类型\n")
cat("  • 右侧柱状图 = 每个基因被多少TF调控\n")
cat("  • 顶部柱状图 = 每个TF调控多少基因\n\n")

cat("⚙️ 调整建议:\n")
cat("  • 修改每类型基因数: n_hvg_per_celltype = ", n_hvg_per_celltype, 
    " → 改为20或50\n")
cat("  • 修改候选TF数: candidate_tfs前30改为50\n")
cat("  • 修改阈值: coef_threshold = ", coef_threshold, 
    " → 改为0.05或0.2\n\n")

cat(rep("=", 80), "\n")
cat("完成时间:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat(rep("=", 80), "\n")















library(tidyverse)
library(ggplot2)
library(scales)

# ============================================================================
# TCF7 调控网络可视化
# ============================================================================

cat("=== 绘制 TCF7 调控图谱 ===\n\n")

# ============================================================================
# Step 1: 加载LASSO结果
# ============================================================================

cat("Step 1: 加载数据...\n")

# 从最新的LASSO分析结果中读取
sig_coefs <- read.csv("15_seacell/celltype_hvg_lasso_coefficients.csv")

cat("总显著系数:", nrow(sig_coefs), "\n")
cat("涉及的TF:", length(unique(sig_coefs$tf)), "个\n")
cat("涉及的基因:", length(unique(sig_coefs$gene)), "个\n")

# 检查TCF7是否在结果中
if (!"TCF7" %in% sig_coefs$tf) {
  cat("\n⚠️  TCF7不在显著系数中！\n")
  cat("可能原因：\n")
  cat("  1. TCF7被过滤掉了\n")
  cat("  2. TCF7没有显著调控任何高变基因\n")
  cat("  3. LASSO将TCF7的系数压缩为0\n\n")
  
  cat("可用的TF列表:\n")
  print(sort(unique(sig_coefs$tf)))
  
  stop("请选择一个可用的TF重新运行")
}

# ============================================================================
# Step 2: 提取TCF7的调控关系
# ============================================================================

cat("\nStep 2: 提取 TCF7 的调控关系...\n")

# 筛选TCF7
tcf7_data <- sig_coefs %>%
  filter(tf == "TCF7")

cat("TCF7 调控的基因数:", nrow(tcf7_data), "\n")

if (nrow(tcf7_data) == 0) {
  stop("❌ TCF7没有显著调控任何基因")
}

# 统计
n_activated <- sum(tcf7_data$coefficient > 0)
n_repressed <- sum(tcf7_data$coefficient < 0)

cat("  • 激活:", n_activated, "个基因\n")
cat("  • 抑制:", n_repressed, "个基因\n")
cat("  • 平均|系数|:", round(mean(abs(tcf7_data$coefficient)), 3), "\n")
cat("  • 最大系数:", round(max(tcf7_data$coefficient), 3), "\n")
cat("  • 最小系数:", round(min(tcf7_data$coefficient), 3), "\n")

# ============================================================================
# Step 3: 添加细胞类型信息（可选）
# ============================================================================

cat("\nStep 3: 添加细胞类型注释...\n")

# 如果有基因统计文件（包含细胞类型信息）
if (file.exists("15_seacell/celltype_hvg_gene_stats.csv")) {
  gene_stats <- read.csv("15_seacell/celltype_hvg_gene_stats.csv")
  
  tcf7_data <- tcf7_data %>%
    left_join(gene_stats %>% select(gene, celltypes), by = "gene")
  
  cat("✓ 已添加细胞类型信息\n")
} else {
  tcf7_data$celltypes <- "Unknown"
  cat("⚠️  未找到细胞类型信息文件\n")
}

# ============================================================================
# Step 4: 数据准备
# ============================================================================

cat("\nStep 4: 准备绘图数据...\n")

# 按系数绝对值排序（最强的调控在上）
tcf7_plot_data <- tcf7_data %>%
  arrange(coefficient) %>%  # 从负到正
  mutate(
    gene = factor(gene, levels = gene),  # 保持排序
    effect = ifelse(coefficient > 0, "Activation", "Repression"),
    abs_coef = abs(coefficient),
    # 简化细胞类型显示
    celltype_simple = case_when(
      grepl("B.Naive", celltypes) ~ "B.Naive",
      grepl("B.Activated", celltypes) ~ "B.Activated",
      grepl("T.CD4.Naive", celltypes) ~ "T.CD4.Naive",
      grepl("T.CD4.Memory", celltypes) ~ "T.CD4.Memory",
      grepl("T.CD8.Naive", celltypes) ~ "T.CD8.Naive",
      grepl("T.CD8.Effector", celltypes) ~ "T.CD8.Effector",
      TRUE ~ "Multiple"
    )
  )

# 标记Top基因（用于高亮）
threshold_top <- quantile(abs(tcf7_plot_data$coefficient), 0.7)
tcf7_plot_data <- tcf7_plot_data %>%
  mutate(is_top = abs(coefficient) >= threshold_top)

cat("✓ 数据准备完成\n")

# ============================================================================
# Step 5: 绘制主图 - 基础柱状图
# ============================================================================

cat("\nStep 5: 绘制 TCF7 调控图谱...\n")

# 配色方案
color_activation <- "#B2182B"  # 红色（激活）
color_repression <- "#2166AC"  # 蓝色（抑制）

# 主图
p_main <- ggplot(tcf7_plot_data, aes(x = gene, y = coefficient, fill = effect)) +
  geom_bar(stat = "identity", width = 0.7, alpha = 0.9) +
  coord_flip() +
  scale_fill_manual(
    values = c("Activation" = color_activation, "Repression" = color_repression),
    name = "Regulation Type"
  ) +
  labs(
    title = "TCF7 Regulatory Network",
    subtitle = paste0("Regulates ", nrow(tcf7_plot_data), " high variable genes (", 
                      n_activated, " activated, ", n_repressed, " repressed)"),
    x = "Target Gene",
    y = "LASSO Coefficient (β)",
    caption = "LASSO regression coefficients from cell type-specific HVGs"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    # 标题
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray40"),
    plot.caption = element_text(size = 9, color = "gray50", hjust = 1),
    
    # 坐标轴
    axis.title = element_text(size = 12, face = "bold"),
    axis.text.y = element_text(size = 9, color = "black"),
    axis.text.x = element_text(size = 10),
    
    # 图例
    legend.position = "bottom",
    legend.title = element_text(size = 11, face = "bold"),
    legend.text = element_text(size = 10),
    
    # 网格
    panel.grid.major.y = element_line(color = "grey90", size = 0.5),
    panel.grid.major.x = element_line(color = "grey90", size = 0.5),
    panel.grid.minor = element_blank(),
    
    # 背景
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA)
  ) +
  geom_hline(yintercept = 0, linetype = "solid", color = "black", size = 0.5) +
  # 添加系数标签（可选，如果基因不多）
  {if(nrow(tcf7_plot_data) <= 30) 
    geom_text(aes(label = sprintf("%.2f", coefficient), 
                  x = gene, 
                  y = ifelse(coefficient > 0, coefficient + 0.02, coefficient - 0.02)),
              hjust = ifelse(tcf7_plot_data$coefficient > 0, -0.1, 1.1),
              size = 3, color = "black")
  }

# 保存主图
ggsave(
  filename = "15_seacell/TCF7_regulation_barplot.pdf",
  plot = p_main,
  width = 10,
  height = max(6, nrow(tcf7_plot_data) * 0.25),
  limitsize = FALSE
)

ggsave(
  filename = "15_seacell/TCF7_regulation_barplot.png",
  plot = p_main,
  width = 10,
  height = max(6, nrow(tcf7_plot_data) * 0.25),
  dpi = 300,
  limitsize = FALSE
)

cat("✓ 主图已保存: TCF7_regulation_barplot.pdf/png\n")

