#include <Rcpp.h>
#include <R_ext/Utils.h>
using namespace Rcpp;

// Preserve R's normalization and probability ordering, then accelerate the
// same sequential probability-proportional draw without replacement.
// [[Rcpp::export]]
List prepare_weighted_draw(NumericVector weights) {
  int n=weights.size();
  NumericVector p=clone(weights), tree(n+1);
  IntegerVector order(n);
  double sum=0;
  for(int i=0;i<n;i++) {if(!R_finite(p[i]) || p[i]<0) stop("Invalid weights"); sum+=p[i]; order[i]=i+1;}
  if(sum<=0) stop("Zero total weight");
  for(int i=0;i<n;i++) p[i]/=sum;
  revsort(p.begin(),order.begin(),n);
  for(int i=1;i<=n;i++) {tree[i]+=p[i-1]; int j=i+(i & -i); if(j<=n) tree[j]+=tree[i];}
  return List::create(_["p"]=p,_["tree"]=tree,_["order"]=order);
}

// [[Rcpp::export]]
IntegerVector cached_weighted_draw(List prepared,int size) {
  NumericVector p=prepared["p"], tree=clone(as<NumericVector>(prepared["tree"]));
  IntegerVector order=prepared["order"];
  int n=p.size(), bit=1; while((bit<<1)<=n) bit<<=1;
  if(size>n) stop("Sample larger than population");
  IntegerVector answer(size); double total=1;
  for(int i=0;i<size;i++) {
    double target=total*R::unif_rand(), prefix=0; int index=0;
    for(int step=bit;step>0;step>>=1) {int j=index+step; if(j<=n && prefix+tree[j]<target) {prefix+=tree[j];index=j;}}
    if(index>=n) stop("Sampling precision failure");
    answer[i]=order[index]; double weight=p[index]; total-=weight;
    for(int j=index+1;j<=n;j+=j & -j) tree[j]-=weight;
  }
  return answer;
}
