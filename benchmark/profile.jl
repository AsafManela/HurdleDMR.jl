using BenchmarkTools, Profile, Traceur, InteractiveUtils, Lasso

include("helpers.jl")

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 5

kwargs = Dict(:verbose=>false)

n = 300
p = 3
covars, counts = bdata(n, p, 1000)
@assert sum(all(iszero.(counts), dims=1)) == 0 "some columns are always zero"

@info "dmr:"
m1 = @btime dmr(covars, counts; local_cluster=true, kwargs...)
m2 = @btime dmr(covars, counts; local_cluster=false, kwargs...)
m3 = @btime dmr(covars, counts; parallel=false, kwargs...)

@assert m1.coefs == m2.coefs
@assert m1.coefs == m3.coefs

@info "hdmr:"
m1 = @btime hdmr(covars, counts; local_cluster=true, kwargs...)
m2 = @btime hdmr(covars, counts; local_cluster=false, kwargs...)
m3 = @btime hdmr(covars, counts; parallel=false, kwargs...)

@assert m1.coefspos == m2.coefspos
@assert m1.coefszero == m2.coefszero
@assert m1.coefspos == m3.coefspos
@assert m1.coefszero == m3.coefszero

@info "fit(InclusionRepetition,...)"
function minifit(counts, covars)
    n,p = size(covars)
    inpos = 1:p-1
    inzero = 1:p
    j = 1
    M = InclusionRepetition
    offsetpos = zeros(n)
    offsetzero = zeros(n)

    cj = Vector(counts[:,j])
    covarspos, covarszero = HurdleDMR.incovars(covars,inpos,inzero)
    # we use the same offsets for pos and zeros
    path = HurdleDMR.fit(M,GammaLassoPath,covarszero,cj;
        Xpos=covarspos, offsetpos=offsetpos, offsetzero=offsetzero)
    # coef(path)
end

@code_warntype minifit(counts, covars)

function minifit2(counts, covars)
    n,p = size(covars)
    inpos = 1:p-1
    inzero = 1:p
    j = 1
    M = InclusionRepetition
    offsetpos = zeros(n)
    offsetzero = zeros(n)

    cj = Vector(counts[:,j])
    covarspos, covarszero = HurdleDMR.incovars2(covars,inpos,inzero)
    # we use the same offsets for pos and zeros
    path = HurdleDMR.fit(M,GammaLassoPath,covarszero,cj;
        Xpos=covarspos, offsetpos=offsetpos, offsetzero=offsetzero)
    # coef(path)
end
@code_warntype minifit2(counts, covars)

cpos1, czero1 = @btime minifit($counts, $covars)
cpos2, czero2 = @btime minifit2($counts, $covars)
@assert cpos1 == cpos2
@assert czero1 == czero2

Profile.init(delay=0.0005)
Juno.@profiler dmr(covars, counts; local_cluster=true, kwargs...)
Juno.@profiler dmr(covars, counts; local_cluster=false, kwargs...)

@trace hdmr(covars, counts; local_cluster=true, kwargs...) maxdepth=3 modules=["HurdleDMR", "Lasso"]

@trace hdmr(covars, counts; local_cluster=true, kwargs...) maxdepth=3 modules=["HurdleDMR", "Lasso"]

##############################
# memory allocation
##############################

# # first start julia with:
# julia --track-allocation=user

# Run whatever commands you wish to test. This first run is to ensure that everything is compiled (because compilation allocates memory).
hdmr(covars, counts; local_cluster=true, kwargs...)

# clear stuff contaminated by compilation
Profile.clear_malloc_data()

# Run your commands again
hdmr(covars, counts; local_cluster=true, kwargs...)

# Quit julia!

# Finally, navigate to the directory holding your source code.
# Start julia (without command-line flags), and analyze the results using

using Coverage
analyze_malloc(".")  # could be "." for the current directory, or "src", etc.

# This will return a vector of MallocInfo objects, specifying the number of bytes allocated
# , the file name, and the line number. These are sorted in increasing order of allocation size.

######################################
# focus
######################################
y = Vector(counts[:,1])
y = counts[:,1]
ixpos1, Iy1 = @btime HurdleDMR.getIy(y)
ixpos2, Iy2 = @btime HurdleDMR.getIy2(y)
@assert ixpos2 == collect(1:n)[ixpos1]
@assert Iy1 == Iy2

@code_warntype HurdleDMR.getIy(y)
@code_warntype HurdleDMR.getIy2(y)
