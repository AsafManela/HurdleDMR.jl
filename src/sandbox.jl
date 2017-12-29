include("testutils.jl")

using Base.Test, Gadfly, Distributions

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

function focus(X; j=:)
  X[:,j]
end
function focus2(X; j=indices(X,2))
  X[:,j]
end

A = sprand(5000,4000,0.2)
@time focus(A)
@time focus2(A)
@time focus(A; j=3)
@time focus2(A; j=3)

j = indices(A,2)
j == 3:3
A
in(4000,indices(A,2))
in(4000,3)
A[indices(A,1),indices(A,2)]
j = 4
focusj = to_indices(A,(:,4))
A[indices(A,1),indices(A,2)]
A[:,4]
A[focusj[1],focusj[2]]

typeof(j)
typeof(:)
colon(1,1,3)
j == 3:3

# clean_folder("/home/amanela/.julia/v0.4")

A = spzeros(Float64,4,5)
convert(SharedArray, A)
A = zeros(Float64,4,5)
@time convert(SharedArray{Float64}, A)
@time convert(SharedArray, A)

function fun(x; kwargs...)
  kwargs
end
kwargs = fun(3;b=1,c=2)
kwargsdict = Dict(kwargs)
haskey(kwargsdict,:b)
kwargsdict[:a]
getindex(kwargsdict,:b)
