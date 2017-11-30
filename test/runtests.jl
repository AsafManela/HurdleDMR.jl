# codefolder = joinpath(dirname(@__FILE__),"..")
# push!(LOAD_PATH, codefolder)
# push!(LOAD_PATH, joinpath(codefolder,"src"))

using FactCheck

rtol=0.05
rdist(x::Number,y::Number) = abs(x-y)/max(abs(x),abs(y))
rdist{T<:Number,S<:Number}(x::AbstractArray{T}, y::AbstractArray{S}; norm::Function=vecnorm) = norm(x - y) / max(norm(x), norm(y))

testfolder = dirname(@__FILE__)

include("positive_poisson.jl")
include("hurdle.jl")
include("dmr.jl")
include("hdmr.jl")
include("multicounts.jl")
include("cross_validation.jl")

FactCheck.exitstatus()
