# R80 D: independently selected 200-gene trends; at most six root-only fits.
script_arg <- grep("^--file=", commandArgs(FALSE), value=TRUE)
if (length(script_arg) != 1L) stop("Run this file with Rscript")
package_root <- dirname(dirname(normalizePath(sub("^--file=", "", script_arg))))
R <- Sys.getenv("GARQ_REVISION_OUTPUT", file.path(package_root,"results"))
dir.create(R,recursive=TRUE,showWarnings=FALSE)
old <- Sys.getenv("GARQ_TRAJECTORY_INPUTS")
if (!nzchar(old)) stop("Set GARQ_TRAJECTORY_INPUTS to the archived trajectory input directory")
O <- file.path(R,"D"); dir.create(O,showWarnings=FALSE)
# Set RETICULATE_PYTHON externally when using a non-default Python environment.
suppressPackageStartupMessages({library(Seurat);library(anndata);library(jsonlite);library(splines);library(Matrix);library(slingshot)})
start <- Sys.time(); options(digits=17)
jwrite <- function(x,n)write_json(x,file.path(O,n),pretty=TRUE,auto_unbox=TRUE,na="null",digits=17)
canon <- function(x){x<-as.character(x);x[x=="Tumor (Neuronal-like)"]<-"Tumor Cells (Dedifferentiated)";x}
config_root <- Sys.getenv("GARQ_V2_CONFIG_DIR", file.path(package_root,"configs"))
settings_path <- file.path(config_root,"D_analysis_plan.R")
if (file.exists(settings_path)) {
 settings_env <- new.env(parent=baseenv())
 sys.source(settings_path,envir=settings_env)
 cfg <- settings_env$CONFIG
} else {
 cfg <- fromJSON(file.path(config_root,"D_analysis_plan.json"),simplifyVector=FALSE)
}
input_root <- Sys.getenv("GARQ_INPUT_ROOT")
if (!nzchar(input_root)) stop("Set GARQ_INPUT_ROOT to the directory containing the hash-bound input files")
for (m in names(cfg$methods)) cfg$methods[[m]]$pt <- gsub("${GARQ_INPUT_ROOT}",input_root,cfg$methods[[m]]$pt,fixed=TRUE)
methods <- names(cfg$methods);states<-unlist(cfg$states)
ord <- read.csv(file.path(old,"metrics/original_cell_ordering.csv"),check.names=FALSE)
stopifnot(nrow(ord)==5711L,sum(ord$in_common_C)==3246L,!anyDuplicated(ord$cell))
read_counts <- function(path){
 a<-read_h5ad(path); mat<-t(a$X); dimnames(mat)<-list(as.character(a$var_names),as.character(a$obs_names))
 list(counts=mat,labels=canon(a$obs$celltype),cells=colnames(mat))
}
sc<-read_counts(file.path(old,"single_three_states.h5ad"))
stopifnot(ncol(sc$counts)==5711L,setequal(sc$cells,ord$cell))
sc$counts<-sc$counts[,match(ord$cell,sc$cells),drop=FALSE];sc$cells<-ord$cell
det<-Matrix::rowMeans(sc$counts>0)
set.seed(42)
obj<-CreateSeuratObject(sc$counts,min.cells=0,min.features=0)
obj<-FindVariableFeatures(obj,selection.method="vst",nfeatures=2000,verbose=FALSE)
info<-HVFInfo(obj,selection.method="vst")
scorecol<-grep("variance.standardized",names(info),fixed=TRUE,value=TRUE)
stopifnot(length(scorecol)==1L)
ranked<-rownames(info)[order(-info[[scorecol]],rownames(info))]
genes<-head(ranked[det[ranked]>=.05 & is.finite(info[ranked,scorecol])],200)
stopifnot(length(genes)>0,!anyDuplicated(genes))
panel<-data.frame(gene=genes,rank=seq_along(genes),detection_fraction=det[genes],vst_standardized_variance=info[genes,scorecol])
write.csv(panel,file.path(O,"PANEL_LOCKED_BEFORE_METRICS.csv"),row.names=FALSE)
jwrite(list(rule="original5711 raw cells, detection>=5%, Seurat vst standardized-variance descending, lexical ties; no method/pseudotime correlation used",n_genes=length(genes),seed=42,Seurat=as.character(packageVersion("Seurat"))),"PANEL_PROTOCOL.json")
normalize <- function(x) {
 s<-Matrix::colSums(x); z<-t(as.matrix(x[genes,,drop=FALSE]))
 log1p(z/pmax(s,1)*10000)
}
Y<-list(SingleCell=normalize(sc$counts));rownames(Y$SingleCell)<-sc$cells
PT<-list(SingleCell=read.csv(file.path(old,"single_three_states_pseudotime.csv"),check.names=FALSE))
for(m in methods){
 x<-read_counts(file.path(old,paste0("common_raw_",m,".h5ad")))
 p<-read.csv(cfg$methods[[m]]$pt,check.names=FALSE);p$cell<-as.character(p$cell)
 stopifnot(all(p$cell%in%x$cells),!anyDuplicated(p$cell),setequal(canon(p$celltype),states))
 v<-normalize(x$counts);rownames(v)<-x$cells;Y[[m]]<-v[p$cell,,drop=FALSE];PT[[m]]<-p
}
nearconstant<-function(x) any(!is.finite(x))||length(x)<2L||diff(range(x))<=1e-12*max(1,max(abs(x)))
safe<-function(x,y,m){if(nearconstant(x)||nearconstant(y))NA_real_ else cor(x,y,method=m)}
fit <- function(values,ptime,groupids,weights,predids) {
 t<-ptime[match(groupids,names(ptime))];tp<-ptime[match(predids,names(ptime))]
 stopifnot(all(is.finite(t)),all(is.finite(tp)),all(weights>0))
 domain<-range(t);if(diff(domain)<=0)return(matrix(NA_real_,length(predids),length(genes)))
 t<-(t-domain[1])/diff(domain);tp<-(tp-domain[1])/diff(domain)
 bas<-splines::ns(t,df=4,intercept=FALSE,Boundary.knots=c(0,1))
 D<-cbind(1,bas);Dp<-cbind(1,predict(bas,newx=tp))
 V<-values[groupids,,drop=FALSE];Dw<-D*sqrt(weights);yw<-V*sqrt(weights);q<-qr(Dw,tol=1e-10)
 cf<-if(q$rank<ncol(D))solve(crossprod(Dw)+diag(1e-8,ncol(D)),crossprod(Dw,yw)) else qr.coef(q,yw)
 out<-Dp%*%cf;colnames(out)<-genes
 attr(out,"coefficients")<-cf
 out
}
rows<-list();fit_records<-list();z<-0L
for(scope in c("common","available")){
 for(m in methods){
  keep<-if(scope=="common")ord$in_common_C else is.finite(ord[[paste0(m,"__pseudotime")]]) & is.finite(ord$reference_pseudotime)
  cohort<-ord[keep,,drop=FALSE];sid<-as.character(cohort$cell);gid<-as.character(cohort[[paste0(m,"__metacell")]])
  stopifnot(length(sid)>2,!anyNA(gid))
  st<-setNames(PT$SingleCell$pseudotime,PT$SingleCell$cell);mt<-setNames(PT[[m]]$pseudotime,PT[[m]]$cell)
  ug<-unique(gid);w<-tabulate(match(gid,ug),nbins=length(ug))
  refpred<-fit(Y$SingleCell,st,sid,rep(1,length(sid)),sid)
  pred<-fit(Y[[m]],mt,ug,w,gid)
  fit_records[[paste(scope,m,sep="__")]]<-list(cell_ids=sid,group_ids=gid,ref_coefficients=attr(refpred,"coefficients"),method_coefficients=attr(pred,"coefficients"),ref_prediction=refpred,method_prediction=pred)
  for(g in genes){
   pr<-safe(refpred[,g],pred[,g],"pearson");sp<-safe(refpred[,g],pred[,g],"spearman");z<-z+1L
   rows[[z]]<-data.frame(scope=scope,method=m,gene=g,n_cells=length(sid),n_groups=length(ug),pearson=pr,spearman=sp,status=if(is.finite(pr)&&is.finite(sp))"defined" else "constant_or_undefined")
  }
  cat("D1",scope,m,length(sid),length(genes),"\n");flush.console()
 }
}
res<-do.call(rbind,rows);write.csv(res,file.path(O,"all_gene_metrics.csv"),row.names=FALSE)
saveRDS(fit_records,file.path(O,"prediction_audit.rds"),compress=TRUE)
sm<-list();z<-0L
for(scope in c("common","available"))for(metric in c("pearson","spearman")){
 r<-res[res$scope==scope,];valid<-genes[vapply(genes,function(g)all(is.finite(r[r$gene==g,metric])),logical(1))]
 for(m in methods){
  v<-r[r$method==m & r$gene%in%valid,metric];z<-z+1L
  sm[[z]]<-data.frame(scope=scope,metric=metric,method=m,shared_valid_genes=length(valid),panel_denominator=length(genes),median=median(v),q25=quantile(v,.25),q75=quantile(v,.75),method_valid_genes=sum(is.finite(r[r$method==m,metric])))
 }
}
write.csv(do.call(rbind,sm),file.path(O,"gene_summaries.csv"),row.names=FALSE)
wins<-do.call(rbind,lapply(c("pearson","spearman"),function(metric){
 r<-res[res$scope=="common",];d<-reshape(r[,c("gene","method",metric)],idvar="gene",timevar="method",direction="wide")
 good<-complete.cases(d);v<-as.matrix(d[good,-1]);data.frame(metric=metric,method=sub(paste0(metric,"."),"",colnames(v),fixed=TRUE),leading_or_tied=colSums(abs(v-apply(v,1,max))<1e-12),common_valid_genes=nrow(v))
}))
write.csv(wins,file.path(O,"leading_counts.csv"),row.names=FALSE)
# Keep all original 15 genes, three pathways and ordering results separate.
dir.create(file.path(O,"original_results"),showWarnings=FALSE)
for(f in c("molecular_metrics.csv","pathway_metrics.csv","ordering_metrics.csv","ordering_by_state.csv","cohort_counts.csv"))
 file.copy(file.path(old,"metrics",f),file.path(O,"original_results",f),overwrite=FALSE)
jwrite(list(status="COMPLETE",n_genes=length(genes),common_cells=3246,available_cells="method-specific, all retained",new_metacell_fits=0,source="validated common raw-count mean matrices, archived pseudotimes",scope="same-data descriptive trend comparison"),"D1_STATUS.json")
# Root-only sensitivity: preserve original coordinates and states, drop only end.clus.
topology<-list();fit_count<-0L;new_pt<-list()
extract_edges<-function(x){tryCatch({g<-slingMST(x);if(inherits(g,"igraph"))as.data.frame(igraph::as_edgelist(g)) else as.data.frame(g)},error=function(e)list(error=conditionMessage(e)))}
{ # Evaluate all six root-only conditions without the historical queue deadline.
 for(m in c("SingleCell",methods)){
  p<-PT[[m]];original_obj<-if(m=="SingleCell")file.path(old,"single_three_states_slingshot.rds") else sub("_pseudotime.csv$","_slingshot.rds",cfg$methods[[m]]$pt)
  original_info<-list(path=original_obj,status="object_not_saved")
  objold<-NULL
  if(file.exists(original_obj)){
   objold<-readRDS(original_obj);original_info<-list(path=original_obj,status="read",lineages=slingLineages(objold),edges=extract_edges(objold))
  }
  coords<-NULL
  if(all(c("UMAP_1","UMAP_2")%in%names(p))){coords<-as.matrix(p[,c("UMAP_1","UMAP_2")]);rownames(coords)<-p$cell}
  else if(!is.null(objold)){
   coords<-slingReducedDim(objold);coords<-coords[match(p$cell,rownames(coords)),,drop=FALSE]
  }
  if(is.null(coords)||any(!is.finite(coords))){
   topology[[m]]<-list(original=original_info,root_only_status="NO_MATCHED_SAVED_COORDINATES");next
  }
  fit_count<-fit_count+1L;stopifnot(fit_count<=6L)
  rr<-tryCatch({
   lin<-getLineages(coords,clusterLabels=canon(p$celltype),start.clus="Tumor Epithelial")
   info<-list(lineages=slingLineages(lin),edges=extract_edges(lin))
   cur<-getCurves(lin,stretch=0);times<-slingPseudotime(cur);saveRDS(cur,file.path(O,paste0("root_only_",m,".rds")))
   write.csv(data.frame(cell=rownames(times),times,check.names=FALSE),file.path(O,paste0("root_only_",m,"_pseudotime.csv")),row.names=FALSE)
   info$coverage<-colMeans(is.finite(times));info$n_lineages<-ncol(times)
   if(ncol(times)==1L && setequal(unlist(slingLineages(cur)[[1]]),states))new_pt[[m]]<-setNames(times[,1],rownames(times))
   info
  },error=function(e)list(error=conditionMessage(e)))
  topology[[m]]<-list(original=original_info,root_only=rr);jwrite(topology,"root_only_topology.json");jwrite(list(attempts=fit_count),"SLINGSHOT_ATTEMPTS.json")
 }
}
rootmetrics<-list()
for(m in methods){
 if(is.null(new_pt$SingleCell)||is.null(new_pt[[m]])){rootmetrics[[m]]<-data.frame(method=m,n=NA,spearman=NA,kendall=NA,status="no comparable single three-state lineage");next}
 c<-ord[ord$in_common_C,];a<-new_pt$SingleCell[c$cell];b<-new_pt[[m]][as.character(c[[paste0(m,"__metacell")]])];valid<-is.finite(a)&is.finite(b)
 rootmetrics[[m]]<-data.frame(method=m,n=sum(valid),spearman=safe(a[valid],b[valid],"spearman"),kendall=safe(a[valid],b[valid],"kendall"),status="root-only single three-state lineage, fixed common cohort valid subset")
}
write.csv(do.call(rbind,rootmetrics),file.path(O,"root_only_ordering.csv"),row.names=FALSE)
jwrite(list(status="COMPLETE",D1="complete",D2="see per-object status and all paths",gene_count=length(genes),slingshot_attempts=fit_count,full_GARQ_fits=0,minutes=as.numeric(difftime(Sys.time(),start,units="mins")),sessionInfo=capture.output(sessionInfo())),"STATUS.json")
