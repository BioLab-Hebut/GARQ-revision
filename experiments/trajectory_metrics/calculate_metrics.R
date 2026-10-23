#!/usr/bin/env Rscript
# R32: fixed, signed, original-cell-aligned descriptive trajectory metrics.
# This script reads validated memberships and common-raw-RNA programme exports.
# It does not refit Slingshot, select favourable lineages, or compute p values.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=1L)stop('Usage: Rscript calculate_metrics.R <analysis_root>')
root <- normalizePath(args[1],mustWork=TRUE)
suppressPackageStartupMessages({library(jsonlite);library(splines)})
options(digits=17)
plan_path <- file.path(root,'analysis_plan.json')
amend_path <- file.path(root,'analysis_amendment_before_results.json')
cfg <- fromJSON(plan_path,simplifyVector=FALSE)
amend <- fromJSON(amend_path,simplifyVector=FALSE)
if(!isTRUE(cfg$fixed_before_metric_results)||!isTRUE(amend$before_any_metric_results)||!isTRUE(amend$no_outcome_selection))stop('The fixed pre-outcome plan and amendment are required')
methods <- names(cfg$methods)
if(!identical(methods,c('GARQ','MetaQ','SEACells','MetaCellV2','SuperCell')))stop('Unexpected method set/order')
genes <- as.character(unlist(cfg$gene_panel,use.names=FALSE))
programmes <- names(cfg$signatures)
features <- c(genes,programmes)
states <- as.character(unlist(cfg$states,use.names=FALSE))
if(length(genes)!=15L||length(programmes)!=3L||anyDuplicated(features)||length(states)!=3L)stop('The complete fixed feature/state sets are required')
canon <- function(x) {
  x <- as.character(x)
  x[x=='Tumor (Neuronal-like)' & !is.na(x)] <- 'Tumor Cells (Dedifferentiated)'
  x
}
assert_ids <- function(x,label) {
  if(anyNA(x)||any(!nzchar(x))||anyDuplicated(x))stop(label,' must contain unique, nonempty IDs')
}
as_numeric_exact <- function(x,label,allow_nonfinite=FALSE) {
  v <- suppressWarnings(as.numeric(x))
  if(any(is.na(v)&!is.na(x)))stop('Non-numeric values in ',label)
  if(!allow_nonfinite&&any(!is.finite(v)))stop('Nonfinite values in ',label)
  v
}
read_programs <- function(path,label) {
  x <- read.csv(path,check.names=FALSE,colClasses='character',stringsAsFactors=FALSE,na.strings=c('NA','NaN',''))
  if(anyDuplicated(names(x))||!all(c('cell','celltype','pseudotime',features)%in%names(x)))stop('Missing/duplicate export columns: ',label)
  assert_ids(x$cell,paste(label,'cell'))
  x$celltype <- canon(x$celltype)
  if(anyNA(x$celltype)||!all(x$celltype%in%states))stop('Unexpected trajectory states: ',label)
  x$pseudotime <- as_numeric_exact(x$pseudotime,paste(label,'pseudotime'),TRUE)
  for(f in features)x[[f]] <- as_numeric_exact(x[[f]],paste(label,f))
  x
}
programme_dir <- file.path(root,'common_raw_programs')
programme_files <- setNames(file.path(programme_dir,paste0('programs_',c('SingleCell',methods),'.csv')),c('SingleCell',methods))
if(!all(file.exists(programme_files)))stop('All common-raw programme exports must exist')
validation_path <- file.path(root,'membership_validation_v3.json')
validation <- fromJSON(validation_path,simplifyVector=FALSE)
if(!isTRUE(validation$passed)||!all(vapply(methods,function(m)isTRUE(validation$methods[[m]]$passed),logical(1))))stop('All membership/group-expression validations must pass before metrics')
extract_plan_path <- file.path(programme_dir,'analysis_plan.json')
if(file.exists(extract_plan_path)) {
  extract_cfg <- fromJSON(extract_plan_path,simplifyVector=FALSE)
  for(m in methods)if(basename(extract_cfg$methods[[m]]$rna)!=paste0('common_raw_',m,'.h5ad'))stop('Programme exports must use common raw-count means: ',m)
}
members_path <- file.path(root,'original_cell_memberships.csv')
members <- read.csv(members_path,check.names=FALSE,colClasses='character',stringsAsFactors=FALSE,na.strings=c('NA','NaN',''))
if(anyDuplicated(names(members))||!all(c('cell','celltype',methods)%in%names(members)))stop('Invalid original-cell membership schema')
assert_ids(members$cell,'Original membership cell')
members$celltype <- canon(members$celltype)
if(anyNA(members$celltype))stop('Missing original cell annotations')
tumor <- members[members$celltype%in%states,,drop=FALSE]
if(nrow(tumor)<2L)stop('Insufficient original tumor cohort')
data <- lapply(names(programme_files),function(m)read_programs(programme_files[[m]],m))
names(data) <- names(programme_files)
ref <- data$SingleCell
if(!setequal(ref$cell,tumor$cell)||nrow(ref)!=nrow(tumor))stop('Reference must contain exactly the original three-state tumor cohort')
ref <- ref[match(tumor$cell,ref$cell),,drop=FALSE]
if(!identical(ref$cell,tumor$cell)||!identical(ref$celltype,tumor$celltype))stop('Reference original cell IDs/states do not align')
mapped_index <- mapped_time <- mapped_type <- list()
for(m in methods) {
  p <- data[[m]]
  known_groups <- unique(na.omit(members[[m]]))
  if(!all(p$cell%in%known_groups))stop('Unknown exported trajectory group ID in ',m)
  idx <- match(tumor[[m]],p$cell)
  mapped_index[[m]] <- idx
  mapped_time[[m]] <- p$pseudotime[idx]
  mapped_type[[m]] <- p$celltype[idx]
}
common_flag <- is.finite(ref$pseudotime)
for(m in methods)common_flag <- common_flag & is.finite(mapped_time[[m]])
C <- tumor[common_flag,,drop=FALSE]
if(nrow(C)<2L)stop('Insufficient original cells valid in reference and all five methods')
assert_ids(C$cell,'Common original cell')
refC <- ref[common_flag,,drop=FALSE]
if(!identical(refC$cell,C$cell))stop('Common reference ID order mismatch')

out <- file.path(root,'metrics')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
if(file.exists(file.path(out,'SUCCESS.txt')))stop('Metrics already completed; preserve the fixed original output')
write_csv <- function(x,name)write.csv(x,file.path(out,name),row.names=FALSE,na='NA',fileEncoding='UTF-8')
near_constant <- function(x)length(x)<2L||diff(range(x))<=1e-12*max(1,max(abs(x)))
safe_cor <- function(x,y,kind) {
  if(length(x)!=length(y))stop('Correlation vectors differ in length')
  if(length(x)<2L)return(list(value=NA_real_,status='fewer_than_two_cells'))
  if(any(!is.finite(x))||any(!is.finite(y)))return(list(value=NA_real_,status='nonfinite_prediction'))
  if(near_constant(x))return(list(value=NA_real_,status='constant_reference'))
  if(near_constant(y))return(list(value=NA_real_,status='constant_method'))
  v <- suppressWarnings(stats::cor(x,y,method=kind,use='everything'))
  list(value=if(is.finite(v))v else NA_real_,status=if(is.finite(v))'defined' else 'undefined_correlation')
}

# Full original cohort export exposes coverage exclusions and mixed-state groups.
original_order <- data.frame(cell=tumor$cell,celltype=tumor$celltype,reference_pseudotime=ref$pseudotime,in_common_C=common_flag,check.names=FALSE)
for(m in methods) {
  original_order[[paste0(m,'__metacell')]] <- tumor[[m]]
  original_order[[paste0(m,'__metacell_state')]] <- mapped_type[[m]]
  original_order[[paste0(m,'__pseudotime')]] <- mapped_time[[m]]
}
write_csv(original_order,'original_cell_ordering.csv')
coverage_rows <- list();z <- 0L
for(scope in c('AllTumor',states)) {
  select <- if(scope=='AllTumor')rep(TRUE,nrow(tumor)) else tumor$celltype==scope
  for(m in c('SingleCell',methods)) {
    finite_m <- if(m=='SingleCell')is.finite(ref$pseudotime) else is.finite(mapped_time[[m]])
    z <- z+1L
    coverage_rows[[z]] <- data.frame(method=m,scope=scope,n_original=sum(select),n_reference_finite=sum(select&is.finite(ref$pseudotime)),n_method_finite=sum(select&finite_m),n_paired_with_reference=sum(select&finite_m&is.finite(ref$pseudotime)),n_common_all_methods=sum(select&common_flag),method_coverage_fraction=sum(select&finite_m)/sum(select),common_coverage_fraction=sum(select&common_flag)/sum(select),check.names=FALSE)
  }
}
coverage <- do.call(rbind,coverage_rows)
write_csv(coverage,'cohort_counts.csv')
ordering <- list();by_state <- list();z <- 0L
for(m in methods) {
  tm <- mapped_time[[m]][common_flag]
  rho <- safe_cor(refC$pseudotime,tm,'spearman')
  tau <- safe_cor(refC$pseudotime,tm,'kendall')
  ordering[[m]] <- data.frame(method=m,n_common_cells=nrow(C),n_common_metacells=length(unique(C[[m]])),spearman_rho=rho$value,kendall_tau_b=tau$value,rho_status=rho$status,tau_status=tau$status,check.names=FALSE)
  for(state in states) {
    select <- C$celltype==state
    rs <- safe_cor(refC$pseudotime[select],tm[select],'spearman')
    z <- z+1L
    by_state[[z]] <- data.frame(method=m,celltype=state,n_common_cells=sum(select),n_common_metacells=length(unique(C[[m]][select])),spearman_rho=rs$value,status=rs$status,check.names=FALSE)
  }
}
ordering <- do.call(rbind,ordering);rownames(ordering) <- NULL
by_state <- do.call(rbind,by_state);rownames(by_state) <- NULL
write_csv(ordering,'ordering_metrics.csv')
write_csv(by_state,'ordering_by_state.csv')

# Basis is fitted once per arm, then reused for original-cell predictions.
# One row represents a unique native group; weights count its members in C.
fit_arm <- function(name,fit_table,weights,pred_times,pred_ids) {
  assert_ids(fit_table$cell,paste(name,'fitting units'))
  if(length(weights)!=nrow(fit_table)||any(!is.finite(weights))||any(weights<=0))stop('Invalid common-cell fitting weights in ',name)
  if(!identical(pred_ids,C$cell)||length(pred_times)!=nrow(C)||any(!is.finite(pred_times)))stop('Prediction original-cell IDs/times do not align in ',name)
  t <- fit_table$pseudotime
  if(any(!is.finite(t)))stop('Nonfinite fit times in ',name)
  domain <- range(t)
  pred <- matrix(NA_real_,nrow(C),length(features),dimnames=list(C$cell,features))
  coefficients <- matrix(NA_real_,5L,length(features),dimnames=list(c('Intercept',paste0('ns',1:4)),features))
  statuses <- setNames(rep('zero_pseudotime_range',length(features)),features)
  arm <- list(name=name,n_fitting_units=nrow(fit_table),n_common_cells=nrow(C),fit_unit_ids=as.list(fit_table$cell),fit_unit_weights=as.list(as.numeric(weights)),observed_pseudotime_min=domain[1],observed_pseudotime_max=domain[2],normalization='(t-min(t))/(max(t)-min(t)), using this arm fitting units on C',df=4,intercept=TRUE,ridge_fallback=1e-8,constant_relative_tolerance=1e-12)
  if(diff(domain)<=0)return(list(pred=pred,coefficients=coefficients,status=statuses,manifest=arm,model=NULL))
  x <- (t-domain[1])/diff(domain)
  xp <- (pred_times-domain[1])/diff(domain)
  if(any(xp < -1e-10)||any(xp > 1+1e-10))stop('Prediction lies outside common fitting domain in ',name)
  basis_result <- tryCatch(splines::ns(x,df=4,intercept=FALSE,Boundary.knots=c(0,1)),error=function(e)e)
  if(inherits(basis_result,'error')) {
    statuses[] <- 'spline_basis_failed'
    arm$basis_error <- conditionMessage(basis_result)
    return(list(pred=pred,coefficients=coefficients,status=statuses,manifest=arm,model=NULL))
  }
  basis <- basis_result
  D <- cbind(Intercept=1,basis)
  colnames(D) <- rownames(coefficients)
  Dp <- cbind(Intercept=1,predict(basis,newx=xp))
  colnames(Dp) <- colnames(D)
  Dw <- D*sqrt(weights)
  q <- qr(Dw,tol=1e-10)
  use_ridge <- q$rank<ncol(D)
  arm$internal_knots <- as.list(as.numeric(attr(basis,'knots')))
  arm$boundary_knots <- as.list(as.numeric(attr(basis,'Boundary.knots')))
  arm$basis_intercept <- FALSE
  arm$design_terms <- as.list(colnames(D))
  arm$design_rank <- q$rank
  arm$ridge_applied <- use_ridge
  arm$normalized_fit_pseudotime <- as.list(x)
  for(f in features) {
    y <- fit_table[[f]]
    if(any(!is.finite(y)))stop('Nonfinite molecular values in ',name,': ',f)
    if(near_constant(y)) {
      statuses[[f]] <- 'constant_input'
      coefficients[,f] <- c(y[1],rep(0,4))
      pred[,f] <- rep(y[1],nrow(C))
      next
    }
    yw <- y*sqrt(weights)
    beta <- tryCatch(if(use_ridge)solve(crossprod(Dw)+diag(1e-8,ncol(D)),crossprod(Dw,yw)) else qr.coef(q,yw),error=function(e)e)
    if(inherits(beta,'error')||any(!is.finite(beta))) {
      statuses[[f]] <- 'fit_failed'
      next
    }
    coefficients[,f] <- as.numeric(beta)
    pred[,f] <- as.vector(Dp%*%beta)
    statuses[[f]] <- if(near_constant(pred[,f]))'constant_fitted_trend' else 'defined'
  }
  arm$feature_fit_status <- as.list(statuses)
  arm$coefficients <- lapply(features,function(f)setNames(as.list(coefficients[,f]),rownames(coefficients)))
  names(arm$coefficients) <- features
  list(pred=pred,coefficients=coefficients,status=statuses,manifest=arm,model=list(basis=basis,coefficients=coefficients,pseudotime_domain=domain,fit_unit_ids=fit_table$cell,weights=weights))
}
fits <- list()
fits$SingleCell <- fit_arm('SingleCell',refC,rep(1,nrow(C)),refC$pseudotime,C$cell)
for(m in methods) {
  group_ids <- unique(C[[m]])
  p <- data[[m]]
  fit_table <- p[match(group_ids,p$cell),,drop=FALSE]
  if(!identical(fit_table$cell,group_ids))stop('Unique metacell fitting IDs do not align in ',m)
  weights <- tabulate(match(C[[m]],group_ids),nbins=length(group_ids))
  if(sum(weights)!=nrow(C))stop('Common-cell group weights do not sum to C in ',m)
  fits[[m]] <- fit_arm(m,fit_table,weights,mapped_time[[m]][common_flag],C$cell)
}
predictions <- data.frame(cell=C$cell,celltype=C$celltype,check.names=FALSE)
for(m in c('SingleCell',methods)) {
  if(!identical(rownames(fits[[m]]$pred),C$cell))stop('Aligned trend ID order mismatch in ',m)
  for(f in features)predictions[[paste0(m,'__',f)]] <- fits[[m]]$pred[,f]
}
write_csv(predictions,'aligned_predicted_trends.csv')
molecular_rows <- list();coef_rows <- list();z <- k <- 0L
for(m in c('SingleCell',methods))for(f in features) {
  for(term in rownames(fits[[m]]$coefficients)) {
    k <- k+1L
    coef_rows[[k]] <- data.frame(arm=m,feature=f,term=term,coefficient=fits[[m]]$coefficients[term,f],fit_status=fits[[m]]$status[[f]],check.names=FALSE)
  }
  if(m=='SingleCell')next
  rp <- safe_cor(fits$SingleCell$pred[,f],fits[[m]]$pred[,f],'pearson')
  rs <- safe_cor(fits$SingleCell$pred[,f],fits[[m]]$pred[,f],'spearman')
  z <- z+1L
  molecular_rows[[z]] <- data.frame(method=m,feature=f,feature_class=if(f%in%genes)'gene' else 'pathway',n_common_cells=nrow(C),n_common_metacells=fits[[m]]$manifest$n_fitting_units,pearson_r=rp$value,spearman_rho=rs$value,pearson_status=rp$status,spearman_status=rs$status,reference_fit_status=fits$SingleCell$status[[f]],method_fit_status=fits[[m]]$status[[f]],check.names=FALSE)
}
molecular <- do.call(rbind,molecular_rows);rownames(molecular) <- NULL
write_csv(molecular,'molecular_metrics.csv')
write_csv(do.call(rbind,coef_rows),'smoothing_coefficients.csv')
gene_summary <- lapply(methods,function(m) {
  tab <- molecular[molecular$method==m & molecular$feature_class=='gene',,drop=FALSE]
  if(nrow(tab)!=15L||!setequal(tab$feature,genes))stop('Full fixed gene panel not retained in ',m)
  pvalid <- is.finite(tab$pearson_r);svalid <- is.finite(tab$spearman_rho)
  data.frame(method=m,n_fixed_genes=15L,n_defined_pearson=sum(pvalid),n_defined_spearman=sum(svalid),median_gene_pearson_r=if(any(pvalid))median(tab$pearson_r[pvalid]) else NA_real_,median_gene_spearman_rho=if(any(svalid))median(tab$spearman_rho[svalid]) else NA_real_,check.names=FALSE)
})
gene_summary <- do.call(rbind,gene_summary)
write_csv(gene_summary,'molecular_gene_summary.csv')
write_csv(molecular[molecular$feature_class=='pathway',,drop=FALSE],'pathway_metrics.csv')

input_files <- c(plan_path,amend_path,validation_path,members_path,programme_files)
if(file.exists(extract_plan_path))input_files <- c(input_files,extract_plan_path)
input_hashes <- as.list(as.character(tools::md5sum(input_files)))
names(input_hashes) <- input_files
manifest <- list(revision=32,description='Common raw-count RNA programmes; archived native pseudotimes; fixed shared original-cell comparisons',plan_md5=unname(tools::md5sum(plan_path)),amendment_md5=unname(tools::md5sum(amend_path)),common_cell_ids=as.list(C$cell),common_cell_n=nrow(C),features=as.list(features),gene_features=as.list(genes),pathway_features=as.list(programmes),arms=lapply(fits,function(x)x$manifest),input_md5=input_hashes,normalization_note='Common raw-count means are LogNormalized to scale factor10000 before molecular programme export. Native archived pseudotimes are unchanged.',correlation_note='Signed Pearson/Spearman trends and signed Spearman/Kendall tau-b orders, no sign flips, no independent sorted-row pairing.',inference_note='Descriptive values from one measured cohort and saved fits; member cells and metacells are not independent biological replicates.',R_version=R.version.string,sessionInfo=capture.output(sessionInfo()))
write_json(manifest,file.path(out,'smoothing_manifest.json'),pretty=TRUE,auto_unbox=TRUE,na='null',digits=17)
saveRDS(lapply(fits,function(x)x$model),file.path(out,'smoothing_models.rds'))
summary <- list(revision=32,status='SUCCESS',n_original_cells=nrow(members),n_original_tumor_cells=nrow(tumor),n_common_cells=nrow(C),common_fraction=nrow(C)/nrow(tumor),states=as.list(table(factor(C$celltype,levels=states))),all_methods=as.list(methods),all_genes=as.list(genes),all_pathways=as.list(programmes),ordering=ordering,within_state=by_state,gene_summary=gene_summary,pathways=molecular[molecular$feature_class=='pathway',,drop=FALSE],coverage=coverage,assumptions=c('exact expression-validated native memberships','common original cells and unchanged orientation','common raw-count RNA convention for newly rescored programmes','fixed spline df4 plus intercept with ridge1e-8 only for rank-deficient fits','all15genes and all3signatures retained','no method-superiority p values'),outputs=as.list(list.files(out)),input_md5=input_hashes)
write_json(summary,file.path(out,'summary.json'),pretty=TRUE,auto_unbox=TRUE,na='null',digits=17)
writeLines('SUCCESS',file.path(out,'SUCCESS.txt'),useBytes=TRUE)
cat('METRICS_SUCCESS',nrow(C),'common original cells\n')
