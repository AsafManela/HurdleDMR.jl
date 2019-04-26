using BenchmarkTools

# Add parallel workers and make package available to workers
using Distributed, CSV, GLM, Distributions, Random, SparseArrays, CategoricalArrays
import HurdleDMR
addprocs(Sys.CPU_THREADS-2)
@everywhere using HurdleDMR

pminmax(x, xmin, xmax) = max(xmin, min(xmax, x))

function bdata(n, p, d; seed=13, qmin=1e-4, qmax=1.0-1e-4)
  Random.seed!(seed)

  λ0 = 500
  m = 1 .+ rand(Poisson(λ0*log(d)),n)
  covars = rand(n,p)
  ηfn(vi) = exp.([0 + λ0/d*log(1+i)*sum(vi) for i=1:d])

  # raw category probabilities
  q = [ηfn(covars[i,:]) for i=1:n]

  # rescale once and chop extremes
  for i=1:n
    q[i] = pminmax.(q[i]/sum(q[i]), qmin, qmax)
  end

  # rescale again to make sure sums to 1
  for i=1:n
    q[i] ./= sum(q[i])
  end

  counts = convert(SparseMatrixCSC{Float64,Int},hcat(broadcast((qi,mi)->rand(Multinomial(mi, qi)),q,m)...)')
  # projdir = p

  covars, counts
end


const SUITE = BenchmarkGroup()

kwargs = Dict(:verbose=>false)

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 60

n = 300
p = 3
# covars, counts = bdata(n, p, 10^4)
# sum(all(iszero.(counts), dims=1))
# @btime m = hdmr(covars, counts)

for d in 10 .^ (1:4)
  SUITE[d] = BenchmarkGroup()
  covars, counts = bdata(n, p, d)
  for f in (DMR, HDMR)
    SUITE[d][string(f)] = BenchmarkGroup()
    for par = ("parallel", "serial")
      parallel = (par == "parallel")
      SUITE[d][string(f)][par] = BenchmarkGroup()
      if parallel
        clustertypes = ["local", "remote"]
      else
        clustertypes = ["local"]
      end
      for clus in clustertypes
        local_cluster = (clus == "local")
        # SUITE[d][f][par][clus] = @benchmarkable $(f)($covars, $counts; parallel=$parallel, local_cluster=$local_cluster, $kwargs...)
        SUITE[d][string(f)][par][clus] = @benchmarkable fit($f, $covars, $counts; parallel=$parallel, local_cluster=$local_cluster, $kwargs...)
      end
    end
  end
end

