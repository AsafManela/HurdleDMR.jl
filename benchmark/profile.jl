using BenchmarkTools, Profile, Traceur, InteractiveUtils, Lasso

include("helpers.jl")

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1

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

@which(hcat(v1, v2))

@assert m1.coefspos == m2.coefspos
@assert m1.coefszero == m2.coefszero
@assert m1.coefspos == m3.coefspos
@assert m1.coefszero == m3.coefszero
@assert m1.coefszero == m2.coefszero

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
    @code_warntype HurdleDMR.fit(M,GammaLassoPath,covarszero,cj;
        Xpos=covarspos, offsetpos=offsetpos, offsetzero=offsetzero)
    # coef(path)
end

minifit(counts, covars)

function minifit2(counts, covars)
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
    @code_warntype HurdleDMR.fit(M,GammaLassoPath,covarszero,cj;
        Xpos=covarspos, offsetpos=offsetpos, offsetzero=offsetzero)
    # coef(path)
end
minifit2(counts, covars)

function minifitzero(counts, covars)
    n,p = size(covars)
    inpos = 1:p-1
    inzero = 1:p
    j = 1
    offsetpos = zeros(n)
    offsetzero = zeros(n)

    covarspos, covarszero = HurdleDMR.incovars(covars,inpos,inzero)

    X = covarszero
    y = Vector(counts[:,j])
    M = GammaLassoPath
    Xpos = covarspos
    dzero = Binomial()
    dpos = Poisson()
    lzero = canonicallink(dzero)
    lpos = canonicallink(dpos)
    dofit = true
    wts = fill(one(eltype(y)),size(y))
    offset = similar(y, 0)
    verbose = false
    showwarnings = false

    ixpos, Iy = HurdleDMR.getIy(y)

    offsetzero, offsetpos = HurdleDMR.setoffsets(y, ixpos, offset, offsetzero, offsetpos)

    # @code_warntype HurdleDMR.finiteoffsetobs(:zeros, X, Iy, offsetzero, wts, showwarnings)
    HurdleDMR.fitzero(M, X, Iy, dzero, lzero, dofit, wts, offsetzero, verbose, showwarnings)
end
@btime minifitzero(counts, covars)

function minifitpos(counts, covars, j)
    n,p = size(covars)
    inpos = 1:p-1
    inzero = 1:p
    offsetpos = zeros(n)
    offsetzero = zeros(n)

    covarspos, covarszero = HurdleDMR.incovars(covars,inpos,inzero)

    X = covarszero
    y = Vector(counts[:,j])
    M = GammaLassoPath
    Xpos = covarspos
    dzero = Binomial()
    dpos = Poisson()
    lzero = canonicallink(dzero)
    lpos = canonicallink(dpos)
    dofit = true
    wts = fill(one(eltype(y)),size(y))
    offset = similar(y, 0)
    verbose = false
    showwarnings = true

    ixpos, Iy = HurdleDMR.getIy(y)

    offsetzero, offsetpos = HurdleDMR.setoffsets(y, ixpos, offset, offsetzero, offsetpos)

    # Xpos optional argument allows to specify a data matrix only for positive counts
    if Xpos == nothing
      # use X for Xpos too
      Xpos = X[ixpos,:]
    elseif size(Xpos,1) == length(y)
      # Xpos has same dimensions as X, take only positive y ones
      Xpos = Xpos[ixpos,:]
    end

    ypos = y[ixpos]
    wtspos = wts[ixpos]

    HurdleDMR.excessy!(ypos, InclusionRepetition)
    # @code_warntype HurdleDMR.finiteoffsetobs(:positives, Xpos, ypos, offsetpos, wtspos, showwarnings)
    # @btime HurdleDMR.finiteoffsetobs(:positives, $Xpos, $ypos, $offsetpos, $wtspos, $showwarnings)
    X, y, offset, wts = HurdleDMR.finiteoffsetobs(:positives, Xpos, ypos, offsetpos, wtspos, showwarnings)
    if length(y) < length(ypos)
        @warn("j=$j has inf offsets")
    end
    X, y, offset, wts
end
function minifitpos2(counts, covars, j)
    n,p = size(covars)
    inpos = 1:p-1
    inzero = 1:p
    offsetpos = zeros(n)
    offsetzero = zeros(n)

    covarspos, covarszero = HurdleDMR.incovars(covars,inpos,inzero)

    X = covarszero
    y = Vector(counts[:,j])
    M = GammaLassoPath
    Xpos = covarspos
    dzero = Binomial()
    dpos = Poisson()
    lzero = canonicallink(dzero)
    lpos = canonicallink(dpos)
    dofit = true
    wts = fill(one(eltype(y)),size(y))
    offset = similar(y, 0)
    verbose = false
    showwarnings = true

    ixpos, Iy = HurdleDMR.getIy(y)

    offsetzero, offsetpos = HurdleDMR.setoffsets(y, ixpos, offset, offsetzero, offsetpos)

    # Xpos optional argument allows to specify a data matrix only for positive counts
    if Xpos == nothing
      # use X for Xpos too
      Xpos = X[ixpos,:]
    elseif size(Xpos,1) == length(y)
      # Xpos has same dimensions as X, take only positive y ones
      Xpos = Xpos[ixpos,:]
    end

    ypos = y[ixpos]
    wtspos = wts[ixpos]

    HurdleDMR.excessy!(ypos, InclusionRepetition)
    # @code_warntype HurdleDMR.finiteoffsetobs!(Xpos, ypos, offsetpos, wtspos, showwarnings, :positives)
    @btime HurdleDMR.finiteoffsetobs!($Xpos, $ypos, $offsetpos, $wtspos, $showwarnings, :positives)
    HurdleDMR.finiteoffsetobs!(Xpos, ypos, offsetpos, wtspos, showwarnings, :positives)
end
for j=1:1000
    Xpos1, ypos1, offsetpos1, wtspos1 = minifitpos(counts, covars, j)
end
Xpos2, ypos2, offsetpos2, wtspos2 = minifitpos2(counts, covars, 3)
minifitpos2(counts, covars)

@code_warntype HurdleDMR.getlogger(true)
@code_warntype HurdleDMR.getlogger(Logging.Error)
using Logging
with_logger(HurdleDMR.getlogger(true)) do
    @info "infoing!"
    @warn "warning!"
    @error "erroring!"
end


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
local_cluster = true

# Run whatever commands you wish to test. This first run is to ensure that everything is compiled (because compilation allocates memory).
hdmr(covars, counts; local_cluster=local_cluster, kwargs...)

# clear stuff contaminated by compilation
Profile.clear_malloc_data()

# Run your commands again
hdmr(covars, counts; local_cluster=local_cluster, kwargs...)

# Quit julia!
exit()

# Finally, navigate to the directory holding your source code.
# Start julia (without command-line flags), and analyze the results using

using Coverage
ma = analyze_malloc(".")  # could be "." for the current directory, or "src", etc.
ma

local_cluster = true
f = open("benchmark/malloc.$local_cluster.log", "w")
    for m in ma
        write(f, string(m), "\n")
    end
close(f)

# This will return a vector of MallocInfo objects, specifying the number of bytes allocated
# , the file name, and the line number. These are sorted in increasing order of allocation size.

# now cleanup
memfiles = Coverage.find_malloc_files(["."])
rm.(memfiles)

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
