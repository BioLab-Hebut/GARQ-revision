# Exact MatchRegionStats weights cached within an arm; a fresh draw is still made
# on every call. This changes computation reuse, not the matching distribution.
make_cached_match <- function(max_entries=512L) {
 cache <- new.env(hash=TRUE,parent=emptyenv());queue <- character()
 function(meta.feature,query.feature,features.match=c('GC.percent'),n=10000,verbose=TRUE,...) {
  stopifnot(is.data.frame(meta.feature),is.data.frame(query.feature),nrow(query.feature)==1L)
  key <- paste(rownames(query.feature),nrow(meta.feature),rownames(meta.feature)[1],tail(rownames(meta.feature),1),paste(features.match,collapse=','),sep='|')
  if(exists(key,envir=cache,inherits=FALSE)) {
   item <- get(key,envir=cache,inherits=FALSE)
  } else {
   meta.feature <- na.omit(meta.feature[,features.match,drop=FALSE])
   for(i in seq_along(features.match)) {
    featmatch <- features.match[[i]]
    density.estimate <- density(x=query.feature[[featmatch]],kernel='gaussian',bw=1)
    weights <- approx(x=density.estimate$x,y=density.estimate$y,xout=meta.feature[[featmatch]],yright=1e-4,yleft=1e-4)$y
    if(i>1) feature.weights <- feature.weights*weights else feature.weights <- weights
   }
   item <- list(ids=rownames(meta.feature),weights=feature.weights)
   assign(key,item,envir=cache);queue <<- c(queue,key)
   if(length(queue)>max_entries) {rm(list=queue[[1]],envir=cache);queue <<- queue[-1]}
  }
  if(length(item$ids)<n) {n <- length(item$ids);warning('Requested more features than available')}
  selected <- sample.int(n=length(item$ids),size=n,prob=item$weights)
  item$ids[selected]
 }
}
