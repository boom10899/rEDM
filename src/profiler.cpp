#include <Rcpp.h>
#include <gperftools/profiler.h>

using namespace Rcpp;

// [[Rcpp::export]]
SEXP start_profiler() {
    ProfilerStart("profiler.out");
    return R_NilValue;
}

// [[Rcpp::export]]
SEXP stop_profiler() {
    ProfilerStop();
    return R_NilValue;
}