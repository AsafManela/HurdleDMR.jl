testfolder = dirname(@__FILE__)
srcfolder = joinpath(testfolder,"..","src")
# push!(LOAD_PATH, joinpath(testfolder,".."))
push!(LOAD_PATH, srcfolder)

using FactCheck, Gadfly

if nworkers() > 1
  rmprocs(workers())
end
info("Starting $(Sys.CPU_CORES) parallel workers for dmr tests...")
addprocs(Sys.CPU_CORES)

using GLM, DataFrames, LassoPlot
@everywhere using HurdleDMR
# uncomment following for debugging and comment the previous @everywhere line. then use reload after making changes
# import HurdleDMR

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
