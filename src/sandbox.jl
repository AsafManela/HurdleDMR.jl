include("testutils.jl")

using FactCheck, Gadfly, Distributions

include("addworkers.jl")

using GLM, DataFrames, LassoPlot

import HurdleDMR; @everywhere using HurdleDMR
# reload("HurdleDMR")

using Coverage
# defaults to src/; alternatively, supply the folder name as argument
coverage = process_folder(srcfolder)
# Get total coverage for all Julia files
covered_lines, total_lines = get_summary(coverage)
# Or process a single file
@show get_summary(process_file("src/HurdleDMR.jl"))

# clean_folder("/home/amanela/.julia/v0.4")

function fun(x; kwargs...)
  kwargs
end
kwargs = fun(3;b=1,c=2)
kwargsdict = Dict(kwargs)
haskey(kwargsdict,:b)
kwargsdict[:a]
getindex(kwargsdict,:b)
