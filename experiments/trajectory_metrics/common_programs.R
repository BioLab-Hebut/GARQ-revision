args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop('Usage: common_programs.R <analysis_root>')
root <- normalizePath(args[[1]], mustWork = TRUE)
dest <- file.path(root, 'common_raw_programs')
success_path <- file.path(dest, 'common_programs_SUCCESS.txt')
if (file.exists(success_path)) stop('A completed common_raw_programs run already exists')

suppressPackageStartupMessages({
  library(Seurat)
  library(UCell)
  library(anndata)
  library(jsonlite)
})
if (!startsWith(as.character(packageVersion('UCell')), '2.8.')) {
  stop('The fixed analysis requires UCell 2.8.x')
}
cfg <- fromJSON(file.path(root, 'analysis_plan.json'), simplifyVector = FALSE)
amendment <- fromJSON(file.path(root, 'analysis_amendment_before_results.json'),
                      simplifyVector = FALSE)
if (!isTRUE(amendment$before_any_metric_results) ||
    !isTRUE(amendment$no_outcome_selection)) {
  stop('The common raw-input amendment must be fixed before metric results')
}

check_ids <- function(x, expected_length, what) {
  x <- as.character(x)
  if (length(x) != expected_length || anyNA(x) || any(!nzchar(x)) ||
      anyDuplicated(x)) stop('Invalid or non-unique ', what)
  x
}
canon <- function(x) {
  x <- as.character(x)
  x[x == 'Tumor (Neuronal-like)'] <- 'Tumor Cells (Dedifferentiated)'
  x
}
genes <- check_ids(unlist(cfg$gene_panel, use.names = FALSE), 15L, 'fixed gene panel')
signatures <- lapply(cfg$signatures, function(x) as.character(unlist(x, use.names = FALSE)))
if (length(signatures) != 3L || is.null(names(signatures)) ||
    anyDuplicated(names(signatures)) || any(!nzchar(names(signatures)))) {
  stop('Expected three uniquely named fixed signatures')
}
for (key in names(signatures)) {
  if (!length(signatures[[key]]) || anyNA(signatures[[key]]) ||
      anyDuplicated(signatures[[key]]) || any(!nzchar(signatures[[key]]))) {
    stop('Invalid fixed signature: ', key)
  }
}
states <- check_ids(unlist(cfg$states, use.names = FALSE), 3L, 'canonical states')
if (!identical(states, c('Tumor Epithelial', 'Tumor Cells (Dedifferentiated)',
                         'Stressed Tumor (p53+)'))) stop('Canonical state order differs')
method_names <- names(cfg$methods)
if (!identical(method_names, c('GARQ', 'MetaQ', 'SEACells', 'MetaCellV2', 'SuperCell'))) {
  stop('Expected the five fixed benchmark methods in their original order')
}
paths <- c(SingleCell = file.path(root, 'single_three_states.h5ad'),
           setNames(file.path(root, paste0('common_raw_', method_names, '.h5ad')),
                    method_names))
if (any(!file.exists(paths))) stop('Common raw RNA input missing: ',
                                  paste(paths[!file.exists(paths)], collapse = ', '))
dir.create(dest, recursive = TRUE, showWarnings = FALSE)
set.seed(42)
background <- NULL
manifest <- list(
  input_convention = amendment$molecular_input,
  normalization = list(method = 'LogNormalize', scale_factor = 10000),
  UCell = list(maxRank = 1500, ncores = 1,
               signatures = signatures),
  fixed_gene_panel = genes,
  canonical_states = states,
  pseudotime_policy = 'Use the existing single-cell reference and five archived metacell pseudotimes without refitting.',
  inputs = list()
)

for (name in names(paths)) {
  cat('START', name, format(Sys.time()), '\n')
  flush.console()
  a <- read_h5ad(paths[[name]])
  counts <- t(a$X)
  gene_ids <- check_ids(a$var_names, nrow(counts), paste0(name, ' gene IDs'))
  cell_ids <- check_ids(a$obs_names, ncol(counts), paste0(name, ' cell IDs'))
  if (is.null(background)) {
    background <- gene_ids
    writeLines(background, file.path(dest, 'gene_background.txt'))
  } else if (!identical(gene_ids, background)) {
    stop('RNA gene background or its order differs in ', name)
  }
  if (length(background) < 1500L) stop('RNA background is smaller than fixed UCell maxRank')
  values <- if (inherits(counts, 'sparseMatrix')) counts@x else counts
  if (any(!is.finite(values)) || any(values < 0)) {
    stop('Common raw RNA input contains nonfinite or negative values in ', name)
  }
  meta <- as.data.frame(a$obs)
  meta_ids <- check_ids(rownames(meta), ncol(counts), paste0(name, ' metadata cell IDs'))
  if (!identical(cell_ids, meta_ids)) stop('Expression and metadata cell IDs differ in ', name)
  if (!'celltype' %in% names(meta)) stop('celltype metadata is missing in ', name)
  meta$celltype <- canon(meta$celltype)
  if (anyNA(meta$celltype) || any(!nzchar(meta$celltype))) stop('Invalid celltype metadata in ', name)
  dimnames(counts) <- list(gene_ids, cell_ids)
  missing <- setdiff(c(genes, unique(unlist(signatures, use.names = FALSE))), gene_ids)
  if (length(missing)) stop('Fixed feature missing in ', name, ': ', paste(missing, collapse = ', '))
  obj <- CreateSeuratObject(counts = counts, meta.data = meta,
                            min.cells = 0, min.features = 0)
  if (!identical(rownames(obj), gene_ids) || !identical(colnames(obj), cell_ids)) {
    stop('Seurat altered an input gene or cell ID in ', name)
  }
  keep <- cell_ids[meta$celltype %in% states]
  if (!length(keep)) stop('No target tumor states in ', name)
  obj <- subset(obj, cells = keep)
  if (!identical(colnames(obj), keep)) stop('Target cell ordering changed in ', name)
  if (!setequal(unique(as.character(obj$celltype)), states)) {
    stop('A canonical tumor state is absent in ', name)
  }
  input_n <- ncol(counts)
  rm(a, counts, meta, values)
  gc()

  obj <- NormalizeData(obj, normalization.method = 'LogNormalize',
                       scale.factor = 10000, verbose = FALSE)
  mat <- GetAssayData(obj, assay = 'RNA', layer = 'data')
  if (!identical(rownames(mat), background) ||
      !identical(colnames(mat), keep)) stop('Normalized RNA IDs differ in ', name)
  score <- ScoreSignatures_UCell(mat, features = signatures, maxRank = 1500,
                                ncores = 1)
  score_ids <- check_ids(rownames(score), length(keep), paste0(name, ' UCell cell IDs'))
  if (!setequal(score_ids, keep)) stop('UCell cell identifiers do not align in ', name)
  score <- score[keep, , drop = FALSE]
  if (ncol(score) != 3L) stop('Expected three UCell signatures in ', name)
  clean_names <- sub('_UCell$', '', colnames(score))
  if (anyDuplicated(clean_names) || !setequal(clean_names, names(signatures))) {
    stop('UCell signature identifiers do not align in ', name)
  }
  score <- score[, match(names(signatures), clean_names), drop = FALSE]
  colnames(score) <- names(signatures)
  gene_frame <- as.data.frame(t(as.matrix(mat[genes, , drop = FALSE])), optional = TRUE)
  colnames(gene_frame) <- genes
  frame <- data.frame(cell = keep, celltype = as.character(obj$celltype),
                      gene_frame, score, check.names = FALSE)
  if (!identical(names(frame), c('cell', 'celltype', genes, names(signatures)))) {
    stop('Program feature names or ordering changed in ', name)
  }
  if (any(!is.finite(as.matrix(frame[, c(genes, names(signatures)), drop = FALSE])))) {
    stop('Program values contain nonfinite values in ', name)
  }

  pt_path <- if (name == 'SingleCell') {
    file.path(root, 'single_three_states_pseudotime.csv')
  } else cfg$methods[[name]]$pt
  if (!file.exists(pt_path)) stop('Saved pseudotime input missing in ', name)
  pt <- read.csv(pt_path, colClasses = c(cell = 'character'), check.names = FALSE)
  if (!all(c('cell', 'celltype', 'pseudotime') %in% names(pt))) {
    stop('Saved pseudotime columns are incomplete in ', name)
  }
  pt$cell <- check_ids(pt$cell, nrow(pt), paste0(name, ' pseudotime cell IDs'))
  pt$celltype <- canon(pt$celltype)
  if (anyNA(pt$celltype) || !setequal(unique(pt$celltype), states) ||
      any(!is.finite(pt$pseudotime))) stop('Invalid saved pseudotime or states in ', name)
  if (!setequal(frame$cell, pt$cell)) stop('Saved pseudotime target cell set differs in ', name)
  idx <- match(frame$cell, pt$cell)
  if (anyNA(idx) || !identical(frame$celltype, pt$celltype[idx])) {
    stop('Saved pseudotime state labels do not align in ', name)
  }
  frame$pseudotime <- pt$pseudotime[idx]
  if (any(!is.finite(frame$pseudotime))) stop('Nonfinite aligned pseudotime in ', name)
  write.csv(frame, file.path(dest, paste0('programs_', name, '.csv')),
             row.names = FALSE, na = '')
  manifest$inputs[[name]] <- list(
    expression_source = normalizePath(paths[[name]], mustWork = TRUE),
    original_input_n = input_n, tumor_n = nrow(frame), n_genes = length(background),
    states = as.list(table(factor(frame$celltype, levels = states))),
    pseudotime_source = normalizePath(pt_path, mustWork = TRUE),
    nonfinite_pseudotime = sum(!is.finite(frame$pseudotime)),
    pseudotime_refitted = FALSE
  )
  cat('DONE', name, '\n')
  flush.console()
  rm(obj, mat, frame, score, gene_frame, pt)
  gc()
}

manifest$gene_background <- list(
  n = length(background), path = file.path(dest, 'gene_background.txt'),
  md5 = unname(tools::md5sum(file.path(dest, 'gene_background.txt'))),
  identical_input_gene_ids_and_order = TRUE
)
manifest$versions <- as.list(sapply(c('Seurat', 'UCell', 'anndata', 'jsonlite', 'Matrix'),
                                    function(x) as.character(packageVersion(x))))
manifest$sessionInfo <- capture.output(sessionInfo())
write_json(manifest, file.path(dest, 'common_programs_manifest.json'),
            pretty = TRUE, auto_unbox = TRUE)
writeLines('SUCCESS', success_path)
cat('ALL_COMMON_PROGRAMS_SUCCESS\n')
