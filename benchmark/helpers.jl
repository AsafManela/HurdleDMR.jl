# Add parallel workers and make package available to workers
using Distributed, CSV, GLM, Distributions, Random, SparseArrays
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
