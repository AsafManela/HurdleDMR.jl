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
@static if !Sys.iswindows()
    # NOTE: we don't run tests on windows because the randomly fail due to mmap() throwing:
    # LoadError: could not create mapping view: The operation completed successfully.
    include("multicounts.jl")
end

end
