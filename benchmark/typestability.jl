using InteractiveUtils

include("benchmarks.jl")

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.5

covars, counts = bdata(n, p, 10^1)
# sum(all(iszero.(counts), dims=1))
@time m = fit(HDMR, covars, counts)
@code_warntype coef(m)

@code_warntype HurdleDMR.dmr_local_cluster(covars,counts,true,false,false,true)
@code_warntype HurdleDMR.dmr_remote_cluster(covars,counts,true,false,false,true)

@btime HurdleDMR.dmr_local_cluster(covars,counts,true,false,false,true)
@btime HurdleDMR.dmr_remote_cluster(covars,counts,true,false,false,true)

@code_warntype HurdleDMR.hdmr_local_cluster(covars,counts,1:p,1:p,true,true,false,false)
@code_warntype HurdleDMR.hdmr_remote_cluster(covars,counts,1:p,1:p,true,true,false,false)

@btime HurdleDMR.hdmr_local_cluster(covars,counts,1:p,1:p,true,true,false,false)
@btime HurdleDMR.hdmr_remote_cluster(covars,counts,1:p,1:p,true,true,false,false)

paths = dmrpaths(covars, counts)
DMRCoefs(paths, HurdleDMR.defsegselect)
@code_warntype DMRCoefs(paths, HurdleDMR.defsegselect)
@code_warntype coef(paths, HurdleDMR.defsegselect)

path = paths.nlpaths[1]
@code_warntype coef(path, HurdleDMR.defsegselect)

paths = hdmrpaths(covars, counts)
HDMRCoefs(paths, HurdleDMR.defsegselect)
@code_warntype HDMRCoefs(paths, HurdleDMR.defsegselect)
@code_warntype coef(paths, HurdleDMR.defsegselect)

@edit dmr_local_cluster(covars, counts)
