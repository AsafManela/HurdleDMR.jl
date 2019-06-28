using Test, Distributions, CSV, GLM, Lasso, DataFrames, MLBase
using Random, Distributed, LinearAlgebra, SparseArrays, SharedArrays, Logging

include("testutils.jl")

include("addworkers.jl")

import HurdleDMR; @everywhere using HurdleDMR

@testset "HurdleDMR" begin

include("positive_poisson.jl")
include("hurdle.jl")
include("inrep.jl")
include("testdata.jl")
include("helpers.jl")
include("dmr.jl")
include("hdmr.jl")
include("multicounts.jl")

end
