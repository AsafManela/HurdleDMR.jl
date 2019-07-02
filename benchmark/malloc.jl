##############################
# memory allocation
##############################

# # first start julia with:
# julia --track-allocation=user malloc.jl <args>
# where args is e.g. true
local_cluster = parse(Bool, ARGS[1])
@info "tracking memory allocation with local_cluster=$local_cluster"

using BenchmarkTools, Profile, Traceur, InteractiveUtils, Lasso

include("helpers.jl")

kwargs = Dict(:verbose=>false)

n = 300
p = 3
covars, counts = bdata(n, p, 10)
@assert sum(all(iszero.(counts), dims=1)) == 0 "some columns are always zero"

@info "First run is to ensure that everything is compiled (because compilation allocates memory)."
hdmr(covars, counts; local_cluster=local_cluster, kwargs...)

@info "clear stuff contaminated by compilation"
Profile.clear_malloc_data()

@info "Run your commands again"
hdmr(covars, counts; local_cluster=local_cluster, kwargs...)

@info "done tracking memory allocation"

@info "Quit julia!"
exit()

# Finally, navigate to the directory holding your source code.
# Start julia (without command-line flags), and analyze the results using

# using Coverage
# analyze_malloc(".")  # could be "." for the current directory, or "src", etc.

# This will return a vector of MallocInfo objects, specifying the number of bytes allocated
# , the file name, and the line number. These are sorted in increasing order of allocation size.
