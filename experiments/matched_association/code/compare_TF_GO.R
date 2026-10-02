# Apply the uploaded TCF7/TBX21 GO workflow to every saved LASSO fit.
suppressPackageStartupMessages({library(clusterProfiler);library(org.Hs.eg.db);library(dplyr);library(jsonlite)})
args<-commandArgs(trailingOnly=TRUE);root<-normalizePath(args[[1]]);D<-file.path(root,'data')
T<-read.csv(file.path(D,'all_TF_gene_coefficients.csv'),stringsAsFactors=FALSE)
map<-suppressMessages(bitr(unique(T$gene),fromType='SYMBOL',toType='ENTREZID',OrgDb=org.Hs.eg.db))
sets<-unique(T[,c('batch','arm')]);cache<-new.env();status<-list();results<-list()
for(i in seq_len(nrow(sets))) for(tf in c('TCF7','TBX21')) for(direction in c('Positive','Negative')) {
 b<-sets$batch[i];a<-sets$arm[i];d<-T[T$batch==b&T$arm==a&T$tf==tf,]
 genes<-d$gene[if(direction=='Positive')d$coefficient>.1 else d$coefficient<(-.1)]
 ids<-unique(map$ENTREZID[map$SYMBOL%in%genes]);key<-paste(sort(ids),collapse=',');res<-NULL
 if(length(ids)>=3) {
  if(exists(key,envir=cache,inherits=FALSE)) res<-get(key,envir=cache) else {
   ego<-suppressMessages(enrichGO(gene=ids,OrgDb=org.Hs.eg.db,ont='BP',pAdjustMethod='BH',pvalueCutoff=.05,qvalueCutoff=.2,readable=TRUE))
   res<-as.data.frame(ego);assign(key,res,envir=cache)
  }
 }
 nr<-if(is.null(res))0L else nrow(res)
 status[[length(status)+1]]<-data.frame(batch=b,arm=a,tf=tf,direction=direction,n_target_symbols=length(genes),n_mapped_ids=length(ids),n_retained_GO_terms=nr,targets=paste(genes,collapse=';'),status=if(length(ids)<3)'Fewer than three mapped genes'else if(nr==0)'No term passes cutoffs'else'Retained GO terms')
 if(nr>0)results[[length(results)+1]]<-cbind(data.frame(batch=b,arm=a,tf=tf,direction=direction),res)
}
write.csv(bind_rows(status),file.path(D,'TF_GO_status.csv'),row.names=FALSE)
write.csv(bind_rows(results),file.path(D,'TF_GO_enrichment.csv'),row.names=FALSE)
write_json(list(coefficient_threshold=.1,ontology='BP',p_adjust='BH',pvalueCutoff=.05,qvalueCutoff=.2,min_mapped_genes=3,background='Default org.Hs.eg.db GO-annotated human genes, matching the uploaded workflow',clusterProfiler=as.character(packageVersion('clusterProfiler')),org.Hs.eg.db=as.character(packageVersion('org.Hs.eg.db')),number_of_models=nrow(sets)),file.path(root,'evidence/TF_GO_parameters.json'),auto_unbox=TRUE,pretty=TRUE)
cat('GO_DONE',nrow(sets),'models\n')
