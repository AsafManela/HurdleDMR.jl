using BenchmarkTools

include("helpers.jl")

const SUITE = BenchmarkGroup()

kwargs = Dict(:verbose=>false)

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 60

n = 300
p = 3
covars, counts = bdata(n, p, 10^2)

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
