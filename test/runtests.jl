@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

include("testutils.jl")

using Distributions

include("addworkers.jl")

using CSV, GLM, Lasso, DataFrames

import HurdleDMR; @everywhere using HurdleDMR

@testset "HurdleDMR" begin
  include("positive_poisson.jl")
  include("hurdle.jl")
  include("testdata.jl")
  include("helpers.jl")
  include("dmr.jl")
  include("hdmr.jl")
  include("multicounts.jl")
end
