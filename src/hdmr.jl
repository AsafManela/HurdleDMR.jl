##############################################################
# Hurdle Distributed Multinomial Regression (HDMR)
##############################################################

# default model used when not specified explicitly, as in fit(HDMR,...)
const HDMR_DEFAULT_MODEL = InclusionRepetition

"Abstract HDMR returned object"
abstract type HDMR{M<:Union{Missing, <:TwoPartModel}} <: DCR end

"""
Relatively heavy object used to return HDMR results when we care about the regulatrization paths.
"""
struct HDMRPaths{M<:Union{Missing, <:TwoPartModel}, X} <: HDMR{M}
  nlpaths::Vector{M} # independent TwoPartModel{GammaLassoPath} for each phrase
  intercept::Bool               # whether to include an intercept in each Poisson regression
                                # (only kept with remote cluster, not with local cluster)
  n::Int                        # number of observations. May be lower than provided after removing all zero obs.
  d::Int                        # number of categories (terms/words/phrases)
  inpos::X                      # indices of covars columns included in positives model
  inzero::X                     # indices of covars columns included in zeros model
end

"""
Relatively light object used to return HDMR results when we only care about estimated coefficients.
"""
struct HDMRCoefs{M<:TwoPartModel, T<:AbstractMatrix, S<:SegSelect, X} <: HDMR{M}
  coefspos::T                   # positives model coefficients
  coefszero::T                  # zeros model coefficients
  intercept::Bool               # whether to include an intercept in each Poisson regression
  n::Int                        # number of observations. May be lower than provided after removing all zero obs.
  d::Int                        # number of categories (terms/words/phrases)
  inpos::X                      # indices of covars columns included in positives model
  inzero::X                     # indices of covars columns included in zeros model
  select::S                     # path segment selector
end

HDMRCoefs{M}(coefspos::T, coefszero::T, intercept::Bool, n::Int, d::Int,
  inpos::X, inzero::X, select::S) where {M<:TwoPartModel, T<:AbstractMatrix, S<:SegSelect, X} =
  HDMRCoefs{M,T,S,X}(coefspos, coefszero, intercept, n, d, inpos, inzero, select)

function HDMRCoefs(m::HDMRPaths{M}, select::SegSelect=defsegselect) where {M<:TwoPartModel}
  coefspos, coefszero = coef(m, select)
  HDMRCoefs{M}(coefspos, coefszero, m.intercept, m.n, m.d, m.inpos, m.inzero, select)
end

"""
    fit(HDMR,covars,counts; <keyword arguments>)
    hdmr(covars,counts; <keyword arguments>)

Fit a Hurdle Distributed Multinomial Regression (HDMR) of counts on covars.

HDMR fits independent hurdle lasso regressions to each column of counts to
approximate a multinomial, picks a segement of each path, and
returns a coefficient matrix (wrapped in HDMRCoefs) representing point estimates
for the entire multinomial (includes the intercept if one was included).

# Example:
```julia
  m = fit(HDMR,covars,counts)
```
# Arguments
- `covars` n-by-p matrix of covariates
- `counts` n-by-d matrix of counts (usually sparse)

# Keywords
- `inpos=1:p` indices of covars columns included in model for positive counts
- `inzero=1:p` indices of covars columns included in model for zero counts
- `intercept::Bool=false` include a intercepts in each hurdle regression
- `parallel::Bool=true` parallelize the poisson fits
- `local_cluster::Bool=true` use local_cluster mode that shares memory across
    parallel workers that is appropriate on a single multicore machine, or
    remote cluster mode that is more appropriate when distributing across machines
    for which sharing memory is costly.
- `verbose::Bool=true`
- `showwarnings::Bool=false`
- `select::SegSelect=MinAICc()` path segment selection criterion
- `kwargs...` additional keyword arguments passed along to fit(Hurdle,...)
"""
function StatsBase.fit(::Type{<:HDMR{M}}, covars::AbstractMatrix{T}, counts::AbstractMatrix;
  kwargs...) where {T<:AbstractFloat, M<:TwoPartModel}

  hdmr(covars, counts, M; kwargs...)
end

"""
    fit(HDMRPaths,covars,counts; <keyword arguments>)
    hdmrpaths(covars,counts; <keyword arguments>)

Fit a Hurdle Distributed Multinomial Regression (HDMR) of counts on covars, and returns
the entire regulatrization paths, which may be useful for plotting or picking
coefficients other than the AICc optimal ones. Same arguments as
[`fit(::HDMR)`](@ref).
"""
function StatsBase.fit(::Type{HDMRPaths{M}}, covars::AbstractMatrix{T}, counts::AbstractMatrix;
  local_cluster=false, # ignored. will always assume remote_cluster
  kwargs...) where {T<:AbstractFloat, M<:TwoPartModel}

  hdmrpaths(covars, counts, M; kwargs...)
end

# default is to use HDMR_DEFAULT_MODEL
StatsBase.fit(::Type{HDMR}, covars::AbstractMatrix{T}, counts::AbstractMatrix;
  kwargs...) where {T<:AbstractFloat} = fit(HDMRCoefs{HDMR_DEFAULT_MODEL}, covars, counts; kwargs...)
StatsBase.fit(::Type{HDMRCoefs}, covars::AbstractMatrix{T}, counts::AbstractMatrix;
    kwargs...) where {T<:AbstractFloat} = fit(HDMRCoefs{HDMR_DEFAULT_MODEL}, covars, counts; kwargs...)
StatsBase.fit(::Type{HDMRPaths}, covars::AbstractMatrix{T}, counts::AbstractMatrix;
  kwargs...) where {T<:AbstractFloat} = fit(HDMRPaths{HDMR_DEFAULT_MODEL}, covars, counts; kwargs...)

# fit wrapper that takes a model (two formulas) and dataframe instead of the covars matrix
# e.g. @model(h ~ x1 + x2, c ~ x1)
# h and c on the lhs indicate model for zeros and positives, respectively.
"""
    fit(HDMR,@model(h ~ x1 + x2, c ~ x1),df,counts; <keyword arguments>)

Fits a HDMR but takes a model formula and dataframe instead of the covars matrix.
See also [`fit(::HDMR)`](@ref).

`h` and `c` on the lhs indicate the model for zeros and positives, respectively.
"""
function StatsBase.fit(::Type{T}, m::Model, df, counts::AbstractMatrix, args...;
  contrasts::Dict{Symbol,<:Any} = Dict{Symbol,Any}(), kwargs...) where {T<:HDMR}

  # parse and merge rhs terms
  trmszero = getrhsterms(m, :h)
  trmspos = getrhsterms(m, :c)
  trms, inzero, inpos = mergerhsterms(trmszero,trmspos)

  # create model matrix
  covars, counts, as = modelcols(trms, df, counts; model=T, contrasts=contrasts)

  # inzero and inpos may be different in the applied schema with factor variables
  inzero, inpos = mapins(inzero, inpos, as)

  # fit and wrap in TableCountsRegressionModel
  TableCountsRegressionModel(fit(T, covars, counts, args...;
    inzero=inzero, inpos=inpos, kwargs...), df, counts, m, as)
end

"""
    coef(m::HDMRCoefs)

Returns the coefficient matrices fitted with HDMR using
the segment selected during fit (MinAICc by default).

# Example:
```julia
  m = fit(HDMR,covars,counts)
  coefspos, coefszero = coef(m)
```
"""
StatsBase.coef(m::HDMRCoefs) = m.coefspos, m.coefszero

coeffill!(coefszero, coefspos, path::Missing, pzero, ppos, j, select::SegSelect) = nothing

function coeffill!(coefszero, coefspos, path::TwoPartModel, pzero, ppos, j, select::SegSelect)
  cjpos, cjzero = coef(path, select)
  coeffill!(coefspos, cjpos, ppos, j, select)
  coeffill!(coefszero, cjzero, pzero, j, select)
end

"""
    coef(m::HDMRPaths, select::SegSelect=MinAICc())

Returns all or selected coefficient matrices fitted with HDMR.

# Example:
```julia
  m = fit(HDMRPaths,covars,counts)
  coefspos, coefszero = coef(m, MinCVKfold{MinCVmse}(5))
```
"""
function StatsBase.coef(m::HDMRPaths, select::SegSelect=defsegselect)
  # get dims
  d = length(m.nlpaths)
  d < 1 && return nothing, nothing

  # drop missing paths
  nonmsngpaths = skipmissing(m.nlpaths)

  # get number of variables from paths object
  ppos = ncoefspos(m)
  pzero = ncoefszero(m)

  # establish maximum path lengths
  nλzero = nλpos = 0
  if !isempty(nonmsngpaths)
    nλzero = maximum(size(nlpath.mzero)[2] for nlpath in nonmsngpaths)
    nλpos = maximum(size(nlpath.mpos)[2] for nlpath in nonmsngpaths)
  end

  nλ = max(nλzero,nλpos)
  # p = pzero + ppos

  # allocate space
  coefszero = coefspace(pzero, d, nλ, select)
  coefspos = coefspace(ppos, d, nλ, select)

  # iterate over paths
  for j=1:d
    path = m.nlpaths[j]
    coeffill!(coefszero, coefspos, path, pzero, ppos, j, select::SegSelect)
  end

  coefspos, coefszero
end

"""
    predict(m,newcovars; <keyword arguments>)

Predict counts using a fitted HDMRPaths object and given newcovars.

# Example:
```julia
  m = fit(HDMRPaths,covars,counts)
  newcovars = covars[1:10,:]
  countshat = predict(m, newcovars; select=MinAICc())
```

# Arguments
- `m::HDMRPaths` fitted DMRPaths model (HDMRCoefs currently not supported)
- `newcovars` n-by-p matrix of covariates of same dimensions used to fit m.

# Keywords
- `select=MinAICc()` See [`coef(::RegularizationPath)`](@ref).
- `kwargs...` additional keyword arguments passed along to predict() for each
  category j=1..size(counts,2)
"""
function StatsBase.predict(m::HDMRPaths, newcovars::AbstractMatrix{T};
  select=defsegselect, kwargs...) where {T<:AbstractFloat}

  covarspos, covarszero = incovars(newcovars,m.inpos,m.inzero)

  _predict(m,covarszero;Xpos=covarspos,select=select,kwargs...)
end

function StatsBase.predict(m::M, newcovars::AbstractMatrix{T};
  select=defsegselect, kwargs...) where {T<:AbstractFloat, M<:HDMR}

  error("predict(m::HDMR,...) can currently only be evaluated for HDMRPaths structs returned from fit(HDMRPaths,...)")
end

"Number of covariates used for HDMR estimation of zeros model"
ncovarszero(m::HDMR) = length(m.inzero)

"Number of covariates used for HDMR estimation of positives model"
ncovarspos(m::HDMR) = length(m.inpos)

"Number of coefficient potentially including intercept used by model for zeros"
ncoefszero(m::HDMR) = ncovarszero(m) + (hasintercept(m) ? 1 : 0)

"Number of coefficient potentially including intercept used by model for positives"
ncoefspos(m::HDMR) = ncovarspos(m) + (hasintercept(m) ? 1 : 0)

"Destandardize two part model estimated using a potentially standardized covars matrix"
function destandardize!(tpm::TwoPartModel, covarsnorm::AbstractVector{T},
    inpos, inzero, standardize) where T

  if standardize
    destandardize!(tpm.mzero, covarsnorm[inzero], standardize)
    destandardize!(tpm.mpos, covarsnorm[inpos], standardize)
  end

  tpm
end

"Shorthand for fit(HDMRPaths,covars,counts). See also [`fit(::HDMRPaths)`](@ref)"
function hdmrpaths(covars::AbstractMatrix{T},counts::AbstractMatrix,::Type{M}=HDMR_DEFAULT_MODEL;
      inpos=1:size(covars,2), inzero=1:size(covars,2),
      intercept=true,
      parallel=true,
      verbose=true, showwarnings=false,
      standardize=true,
      m=nothing, l=nothing, D=nothing,
      kwargs...) where {T<:AbstractFloat,M<:TwoPartModel}
  # get dimensions
  n, d = size(counts)
  n1,p = size(covars)
  @assert n==n1 "counts and covars should have the same number of observations"

  ppos = length(inpos)
  pzero = length(inzero)

  verbose && @info("fitting $n observations on $d categories \n$ppos covariates for positive and $pzero for zero counts")

  # add one coef for intercept
  ncoefpos = ppos + (intercept ? 1 : 0)
  ncoefzero = pzero + (intercept ? 1 : 0)

  covars, counts, μpos, μzero, n = shifters(M, covars, counts, showwarnings, m, l, D)

  # standardize covars only once if needed
  covars, covarsnorm = Lasso.standardizeX(covars, standardize)

  covarspos, covarszero = incovars(covars,inpos,inzero)

  function tryfith(countsj::AbstractVector)
    try
      # we make it dense remotely to reduce communication costs
      # we use the same offsets for pos and zeros
      tpm = fit(M,GammaLassoPath,covarszero,Vector(countsj); Xpos=covarspos, offsetpos=μpos, offsetzero=μzero, verbose=false, standardize=false,
        # showwarnings=showwarnings, # uncomment do help with debugging
        kwargs...)
      destandardize!(tpm, covarsnorm, inpos, inzero, standardize)
    catch e
      showwarnings && @warn("fit($M...) failed for countsj with frequencies $(sort(countmap(countsj))) and will return missing path ($e)")
      missing
    end
  end

  # counts generator
  countscols = (counts[:,j] for j=1:d)

  if parallel
    verbose && @info("distributed $M run on remote cluster with $(nworkers()) nodes")
    mapfn = pmap
  else
    verbose && @info("serial $M run on a single node")
    mapfn = map
  end

  nlpaths = mapfn(tryfith,countscols)

  HDMRPaths(nlpaths, intercept, n, d, inpos, inzero)
end

"Returns a (covarspos, covarszero) tuple from covars"
function incovars(covars,inpos,inzero)
  #NOTE: we don't use views for typesafety
  inall = 1:size(covars,2)

  if inzero == inall
    covarszero = covars
  else
    covarszero = covars[:,inzero]
  end

  if inzero == inpos
    covarspos = covarszero
  elseif inpos == inall
    covarspos = covars
  else
    covarspos = covars[:,inpos]
  end

  covarspos, covarszero
end

function shifters(::Type{Hurdle}, covars::AbstractMatrix, counts::AbstractMatrix, showwarnings::Bool,
  prespecm::Union{Nothing, AbstractVector}, prespecl, prespecd)

  showwarnings && @warn("Old Hurdle-DMR model does not use prespecified l")
  shifters(Hurdle, covars, counts, showwarnings, prespecm, nothing, nothing)
end

# same as DMR, and uses same shifters for both model parts
function shifters(::Type{Hurdle}, covars::AbstractMatrix, counts::AbstractMatrix, showwarnings::Bool,
  prespecm::Union{Nothing, AbstractVector}, prespecl::Nothing, prespecd)

  covars, counts, μ, n = shifters(DMR, covars, counts, showwarnings, prespecm)
  covars, counts, μ, μ, n
end

vocab(counts, prespecd::Nothing) = size(counts, 2)
vocab(counts, prespecd::Int) = prespecd

function shifters(::Type{InclusionRepetition}, covars::AbstractMatrix, counts::AbstractMatrix{C}, showwarnings::Bool,
  prespecm::Union{Nothing, AbstractVector},
  prespecl::Union{Nothing, AbstractVector},
  prespecd::Union{Nothing, Int}) where C

  # standardize counts matrix to conform to GLM.FP
  counts = fpcounts(counts)

  # mi = total word count per observation
  m = totalcounts(counts, prespecm)

  # li = vocabulary per observation
  l = totalcounts(posindic(counts), prespecl)

  if any(iszero,m)
      # omit observations with no counts
      ixposm = findall(x->x!=zero(C), m)
      showwarnings && @warn("omitting $(length(m)-length(ixposm)) observations with no counts")
      m = m[ixposm]
      l = l[ixposm]
      counts = counts[ixposm,:]
      covars = covars[ixposm,:]
  end

  # get new dimensions
  # NOTE: d here is total vocabulary as opposed to obs-specific lexicon l
  d = vocab(counts, prespecd)
  n = length(m)

  # μipos = log(mi-li)
  μpos = broadcast((mi,li)->log(mi-li), m, l)

  # μizero = log(li/(D-li))
  μzero = broadcast(li->log(li/(d-li)), l)

  covars, counts, μpos, μzero, n
end

"Fits a regularized hurdle regression counts[:,j] ~ covars saving the coefficients in coefs[:,j]"
function hurdle_regression!(::Type{M}, coefspos::AbstractMatrix{T}, coefszero::AbstractMatrix{T}, j::Int, covars::AbstractMatrix{T},counts::AbstractMatrix{V},
            inpos, inzero, offsetpos, offsetzero;
            select=defsegselect,
            kwargs...) where {T<:AbstractFloat,V,M<:TwoPartModel}
  cj = Vector(counts[:,j])
  covarspos, covarszero = incovars(covars,inpos,inzero)
  # we use the same offsets for pos and zeros
  path = fit(M,GammaLassoPath,covarszero,cj;
    Xpos=covarspos, offsetpos=offsetpos, offsetzero=offsetzero, kwargs...)
  (coefspos[:,j], coefszero[:,j]) = coef(path, select)
  nothing
end

"Wrapper for hurdle_regression! that catches exceptions in which case it sets coefs to zero"
function tryfith!(::Type{M}, coefspos::AbstractMatrix{T}, coefszero::AbstractMatrix{T}, j::Int, covars::AbstractMatrix{T},counts::AbstractMatrix{V},
            inpos, inzero, offsetpos, offsetzero;
            showwarnings = false,
            kwargs...) where {T<:AbstractFloat,V,M<:TwoPartModel}
  try
    hurdle_regression!(M, coefspos, coefszero, j, covars, counts, inpos, inzero,
      offsetpos, offsetzero;
      # showwarnings=showwarnings, # uncomment do help with debugging
      kwargs...)
  catch e
    showwarnings && @warn("hurdle_regression! failed on count dimension $j with frequencies $(sort(countmap(counts[:,j]))) and will return zero coefs ($e)")
    # redudant ASSUMING COEFS ARRAY INTIAILLY FILLED WITH ZEROS, but can happen in serial mode
    for i=1:size(coefszero,1)
      coefszero[i,j] = zero(T)
    end
    for i=1:size(coefspos,1)
      coefspos[i,j] = zero(T)
    end
  end
end

"Shorthand for fit(HDMR,covars,counts). See also [`fit(::HDMR)`](@ref)"
function hdmr(covars::AbstractMatrix{T},counts::AbstractMatrix{V},::Type{M}=HDMR_DEFAULT_MODEL;
          inpos=1:size(covars,2), inzero=1:size(covars,2),
          intercept=true,
          parallel=true, local_cluster=true,
          verbose=true, showwarnings=false,
          kwargs...) where {T<:AbstractFloat,V,M<:TwoPartModel}
  if local_cluster || !parallel
    hdmr_local_cluster(M, covars,counts,inpos,inzero,intercept,parallel,verbose,showwarnings; kwargs...)
  else
    hdmr_remote_cluster(M, covars,counts,inpos,inzero,intercept,parallel,verbose,showwarnings; kwargs...)
  end
end

"""
This version is built for local clusters and shares memory used by both inputs
and outputs if run in parallel mode.
"""
function hdmr_local_cluster(::Type{M}, covars::AbstractMatrix{T},counts::AbstractMatrix{V},
          inpos,inzero,intercept,parallel,verbose,showwarnings;
          select=defsegselect,
          m=nothing, l=nothing, D=nothing,
          standardize=true, kwargs...) where {T<:AbstractFloat,V,M<:TwoPartModel}
  # get dimensions
  n, d = size(counts)
  n1,p = size(covars)
  @assert n==n1 "counts and covars should have the same number of observations"

  ppos = length(inpos)
  pzero = length(inzero)

  verbose && @info("fitting $n observations on $d categories \n$ppos covariates for positive and $pzero for zero counts")

  # add one coef for intercept
  ncoefpos = ppos + (intercept ? 1 : 0)
  ncoefzero = pzero + (intercept ? 1 : 0)

  covars, counts, μpos, μzero, n = shifters(M, covars, counts, showwarnings, m, l, D)

  # standardize covars only once if needed
  covars, covarsnorm = Lasso.standardizeX(covars, standardize)

  # fit separate GammaLassoPath's to each dimension of counts j=1:d and pick its min AICc segment
  if parallel
    verbose && @info("distributed $M run on local cluster with $(nworkers()) nodes")
    scounts = convert(SharedArray,counts)
    scoefszero = SharedMatrix{T}(ncoefzero,d)
    scoefspos = SharedMatrix{T}(ncoefpos,d)
    scovars = convert(SharedArray,covars)
    # μ = convert(SharedArray,μ) incompatible with GLM

    @sync @distributed for j=1:d
      tryfith!(M, scoefspos, scoefszero, j, scovars, scounts, inpos, inzero, μpos, μzero;
        verbose=false, showwarnings=showwarnings, intercept=intercept,
        standardize=false, select=select, kwargs...)
    end

    coefszero = convert(Matrix{T}, scoefszero)
    coefspos = convert(Matrix{T}, scoefspos)
  else
    verbose && @info("serial $M run on a single node")
    coefszero = Matrix{T}(undef,ncoefzero,d)
    coefspos = Matrix{T}(undef,ncoefpos,d)
    for j=1:d
      tryfith!(M, coefspos, coefszero, j, covars, counts, inpos, inzero, μpos, μzero;
        verbose=false, showwarnings=showwarnings, intercept=intercept,
        standardize=false, select=select, kwargs...)
    end
  end

  if standardize
    # destandardize coefs only once if needed
    destandardize!(coefspos, covarsnorm[inpos], standardize, intercept)
    destandardize!(coefszero, covarsnorm[inzero], standardize, intercept)
  end

  HDMRCoefs{M}(coefspos, coefszero, intercept, n, d, inpos, inzero, select)
end

"""
This version does not share memory across workers, so may be more efficient for
 small problems, or on remote clusters.
"""
function hdmr_remote_cluster(::Type{M}, covars::AbstractMatrix{T},counts::AbstractMatrix{V},
          inpos,inzero,intercept,parallel,verbose,showwarnings;
          select=defsegselect, kwargs...) where {T<:AbstractFloat,V,M<:TwoPartModel}
  paths = hdmrpaths(covars, counts, M; inpos=inpos, inzero=inzero, parallel=parallel, verbose=verbose, showwarnings=showwarnings, kwargs...)
  HDMRCoefs(paths, select)
end

"""
    posindic(A)

Returns an array of the same dimensions of indicators for positive entries in A.
"""
function posindic(A::AbstractArray{T}) where T
  # find positive y entries
  ixpos = (LinearIndices(A))[findall(x->x!=zero(T), A)]

  # build positive indicators matrix
  Ia = deepcopy(A)
  Ia[ixpos] .= one(T)
  Ia
end

function dropexplicitzeros(I, J, V::F) where {T, F<:AbstractVector{T}}
  if any(iszero, V)
    ixnonzero = broadcast(x->!iszero(x), V)
    @inbounds I = I[ixnonzero]
    @inbounds J = J[ixnonzero]
    nnonzero = sum(ixnonzero)
  else
    nnonzero = length(V)
  end

  I, J, nnonzero
end

"Sparse version simply replaces all the non-zero values with ones."
function posindic(A::SparseMatrixCSC{T}) where T
  m,n = size(A)
  I,J,V = findnz(A)
  I, J, nnonzero = dropexplicitzeros(I, J, V)
  sparse(I, J, fill(one(T),nnonzero), m, n)
end
