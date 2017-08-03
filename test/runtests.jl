codefolder = joinpath(dirname(@__FILE__),"..")
push!(LOAD_PATH, codefolder)
push!(LOAD_PATH, joinpath(codefolder,"src"))

using FactCheck

include("positive_poisson.jl")
include("hurdle.jl")
include("dmr.jl")

FactCheck.exitstatus()
