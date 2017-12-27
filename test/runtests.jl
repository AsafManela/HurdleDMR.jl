using FactCheck

include("positive_poisson.jl")
include("hurdle.jl")
include("dmr.jl")
include("hdmr.jl")
include("multicounts.jl")
include("cross_validation.jl")

FactCheck.exitstatus()
