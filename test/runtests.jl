@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

include("positive_poisson.jl")
include("hurdle.jl")
include("dmr.jl")
include("hdmr.jl")
include("multicounts.jl")
include("cross_validation.jl")
