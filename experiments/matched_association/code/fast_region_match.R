# Cache deterministic matching distributions, while drawing fresh backgrounds
# for each peak-gene test. The sequential weighted sampling law is unchanged.
make_fast_match <- function(max_entries=256L) {
 cache <- new.env(hash=TRUE,parent=emptyenv());pools<-new.env(hash=TRUE,parent=emptyenv());queue<-character()
 function(meta.feature,query.feature,features.match=c('GC.percent'),n=10000,verbose=TRUE,...) {
  stopifnot(nrow(query.feature)==1L)
  poolkey<-paste(nrow(meta.feature),rownames(meta.feature)[1],tail(rownames(meta.feature),1),paste(features.match,collapse=','),sep='|')
  key<-paste(rownames(query.feature),poolkey,sep='|')
  if(exists(key,envir=cache,inherits=FALSE)) item<-get(key,envir=cache) else {
   if(exists(poolkey,envir=pools,inherits=FALSE)) pool<-get(poolkey,envir=pools) else {
    pool<-na.omit(meta.feature[,features.match,drop=FALSE]);assign(poolkey,pool,envir=pools)
   }
   for(i in seq_along(features.match)) {
    f<-features.match[[i]]
    d<-density(x=query.feature[[f]],kernel='gaussian',bw=1)
    w<-approx(x=d$x,y=d$y,xout=pool[[f]],yright=1e-4,yleft=1e-4)$y
    if(i==1L) weights<-w else weights<-weights*w
   }
   item<-list(ids=rownames(pool),prepared=prepare_weighted_draw(weights))
   assign(key,item,envir=cache);queue<<-c(queue,key)
   if(length(queue)>max_entries) {rm(list=queue[[1]],envir=cache);queue<<-queue[-1]}
  }
  item$ids[cached_weighted_draw(item$prepared,min(n,length(item$ids)))]
 }
}
