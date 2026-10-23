suppressPackageStartupMessages({library(Seurat);library(Signac);library(Matrix);library(GenomicRanges);library(jsonlite);library(rhdf5);library(future)})
args <- commandArgs(trailingOnly=TRUE)
batch <- args[[1]];arm <- args[[2]]
pilot <- if(length(args)>2) as.integer(args[[3]]) else 0L
workers <- as.integer(Sys.getenv('GARQ_PEAK_WORKERS','4'))
root <- '/workspace/garq/vscode/GARQ20260905/reviewer2_comment4_20260918'
previous <- '/workspace/garq/vscode/GARQ20260905/reviewer2_comment4_20260917/analysis'
out <- file.path(root,if(pilot>0) Sys.getenv('GARQ_PILOT_DIR','peak_pilot') else 'peak_gene',batch,arm)
dir.create(out,recursive=TRUE,showWarnings=FALSE)
if(file.exists(file.path(out,'DONE'))) {cat('Already complete\n');quit(status=0)}
started <- Sys.time()
seed <- if(grepl('matched_seed',batch)) as.integer(sub('matched_seed','',batch)) else 1L
set.seed(seed);options(garq.seed=seed,future.globals.maxSize=30*1024^3)
plan(multicore,workers=workers)
paths <- fromJSON(file.path(previous,batch,'MOFA_input_manifest.json'))[[arm]]
read_counts <- function(path) {
 attrs <- h5readAttributes(path,'X');stopifnot(attrs[['encoding-type']]=='csr_matrix')
 index_name <- function(group) h5readAttributes(path,group)[['_index']]
 vars <- as.character(h5read(path,paste0('var/',index_name('var'))))
 cells <- as.character(h5read(path,paste0('obs/',index_name('obs'))))
 x <- new('dgCMatrix',x=as.numeric(h5read(path,'X/data')),i=as.integer(h5read(path,'X/indices')),p=as.integer(h5read(path,'X/indptr')),Dim=as.integer(rev(attrs$shape)),Dimnames=list(vars,cells))
 h5closeAll()
 x
}
cat('READ',batch,arm,format(Sys.time()),'\n')
if(batch=='single_cell') {
 obj <- readRDS('/workspace/garq/Rworkspace/MetaQ/three/compare/intermediate_data/seurat_sc_processed_celltype.rds')
 # The saved object contains the same full paired raw matrices, with the original normalization.
 stopifnot(ncol(obj)==25517)
 gcdata <- readRDS(file.path(root,'audit/original_peak_metadata.rds'))
 obj[['ATAC']]@meta.features <- gcdata[rownames(obj[['ATAC']]),c('GC.percent','sequence.length'),drop=FALSE]
} else {
 rna <- read_counts(paths$RNA);atac <- read_counts(paths$ATAC)
 stopifnot(identical(colnames(rna),colnames(atac)))
 obj <- CreateSeuratObject(counts=CreateAssayObject(counts=rna),assay='RNA')
 obj[['ATAC']] <- CreateChromatinAssay(counts=atac,sep=c('-','-'),min.cells=0,min.features=0)
 obj <- NormalizeData(obj,assay='RNA',normalization.method='LogNormalize',scale.factor=10000,verbose=FALSE)
 gcdata <- readRDS(file.path(root,'audit/original_peak_metadata.rds'))
 stopifnot(all(rownames(atac) %in% rownames(gcdata)))
 obj[['ATAC']]@meta.features <- gcdata[rownames(atac),c('GC.percent','sequence.length'),drop=FALSE]
 rm(rna,atac);gc()
}
coord_file <- file.path(root,'audit/collapsed_gene_coordinates.rds')
if(!file.exists(coord_file)) {
 ann <- readRDS(file.path(root,'audit/original_annotation.rds'))
 coords <- Signac:::CollapseToLongestTranscript(ann)
 saveRDS(coords,coord_file)
} else coords <- readRDS(coord_file)
genes <- NULL
if(pilot>0) {
 valid <- intersect(rownames(obj[['RNA']]),coords$gene_name)
 set.seed(17);genes <- sort(sample(valid,min(pilot,length(valid))))
}
# Use installed Signac calculation unchanged; only make the parallel RNG explicit.
code <- readLines(file.path(root,'audit/installed_LinkPeaks.R'))
code <- gsub('mylapply <- future_lapply','mylapply <- function(X,FUN) future.apply::future_lapply(X=X,FUN=FUN,future.seed=as.integer(getOption("garq.seed",1)),future.scheduling=2)',code,fixed=TRUE)
if(Sys.getenv('GARQ_FAST_GC','1')=='1') code <- code[!grepl('^[[:space:]]*gc\\(verbose = FALSE\\)',code)]
link <- eval(parse(text=paste(code,collapse='\n')))
environment(link) <- asNamespace('Signac')
if(Sys.getenv('GARQ_CACHE_MATCH','1')=='1') {
 if(Sys.getenv('GARQ_FAST_SAMPLE','1')=='1') {
  Rcpp::sourceCpp(file.path(root,'code/fast_weighted_sampling.cpp'),cacheDir=file.path(root,'code/rcppcache'))
  source(file.path(root,'code/fast_region_match.R'))
 } else source(file.path(root,'code/cached_region_match.R'))
 link_env <- new.env(parent=asNamespace('Signac'))
 link_env$MatchRegionStats <- if(Sys.getenv('GARQ_FAST_SAMPLE','1')=='1') make_fast_match() else make_cached_match()
 environment(link) <- link_env
}
minimum <- if(batch=='single_cell') 10L else 3L
rna_eligible <- rowSums(GetAssayData(obj,assay='RNA',layer='data')>0)>minimum
peak_eligible <- rowSums(GetAssayData(obj,assay='ATAC',layer='counts')>0)>minimum
tested_coords <- coords[coords$gene_name %in% names(rna_eligible)[rna_eligible],]
candidate_pairs <- sum(Signac:::DistanceToTSS(peaks=granges(obj[['ATAC']])[peak_eligible],genes=tested_coords,distance=5e5))
cat('FIT',batch,arm,'profiles',ncol(obj),'pilot',pilot,format(Sys.time()),'\n')
fit_start <- Sys.time()
obj <- link(object=obj,peak.assay='ATAC',expression.assay='RNA',peak.slot='counts',expression.slot='data',gene.coords=coords,distance=5e5,min.cells=minimum,genes.use=genes,n_sample=200,pvalue_cutoff=.05,score_cutoff=.05,method='pearson',verbose=TRUE)
links <- as.data.frame(Links(obj[['ATAC']]))
if(!nrow(links)) stop('No links returned')
# The peak identifier, not the link-span midpoint, gives the peak's genomic center.
pk <- StringToGRanges(links$peak,sep=c('-','-'))
pos <- match(links$gene,coords$gene_name)
tss <- ifelse(as.character(strand(coords[pos]))=='-',end(coords[pos]),start(coords[pos]))
links$peak_center <- (start(pk)+end(pk))/2
links$gene_tss <- tss
links$distance <- abs(links$peak_center-tss)
links$pair_id <- paste(links$peak,links$gene,sep='::')
stopifnot(!anyDuplicated(links$pair_id),all(is.finite(links$score)),all(is.finite(links$pvalue)))
links <- links[order(links$pair_id),]
saveRDS(links,file.path(out,'links.rds'))
conn <- gzfile(file.path(out,'links.csv.gz'),'wt');write.csv(links,conn,row.names=FALSE);close(conn)
summary <- list(batch=batch,arm=arm,profiles=ncol(obj),links=nrow(links),positive_links=sum(links$score>0),strong_positive_links=sum(links$score>.5),median_score=median(links$score),genes=length(unique(links$gene)),peaks=length(unique(links$peak)),distance_limit=500000,min_cells=minimum,n_background=200,pvalue_cutoff=.05,absolute_score_cutoff=.05,seed=seed,pilot_genes=pilot,workers=workers,fit_seconds=as.numeric(difftime(Sys.time(),fit_start,units='secs')),total_seconds=as.numeric(difftime(Sys.time(),started,units='secs')),source_paths=paths,Signac=as.character(packageVersion('Signac')),distance_definition='Absolute peak genomic midpoint to the same annotated TSS used by LinkPeaks')
summary$eligible_genes <- sum(rna_eligible)
summary$eligible_peaks <- sum(peak_eligible)
summary$genes_with_coordinates <- length(tested_coords)
summary$candidate_cis_pairs <- as.numeric(candidate_pairs)
write_json(summary,file.path(out,'summary.json'),auto_unbox=TRUE,pretty=TRUE)
writeLines(format(Sys.time()),file.path(out,'DONE'))
cat(toJSON(summary,auto_unbox=TRUE,pretty=TRUE),'\n')
