##############################################################
# Hurdle Distributed Multinomial Regression (HDMR)
##############################################################

"Abstract HDMR returned object"
abstract type HDMR <: DCR end

"""
Relatively heavy object used to return HDMR results when we care about the regulatrization paths.
"""
struct HDMRPaths <: HDMR
  nlpaths::Vector{Union{Missing,Hurdle}} # independent Hurdle{GammaLassoPath} for each phrase
  intercept::Bool               # whether to include an intercept in each Poisson regression
                                # (only kept with remote cluster, not with local cluster)
  n::Int                      # number of observations. May be lower than provided after removing all zero obs.
  d::Int                      # number of categories (terms/words/phrases)
  inpos                         # indices of covars columns included in positives model
  inzero                        # indices of covars columns included in zeros model

  HDMRPaths(nlpaths::Vector{Union{Missing,Hurdle}}, intercept::Bool, n::Int, d::Int, inpos, inzero) =
    new(nlpaths, intercept, n, d, inpos, inzero)
end

"""
Relatively light object used to return HDMR results when we only care about estimated coefficients.
"""
struct HDMRCoefs <: HDMR
  coefspos::AbstractMatrix      # positives model coefficients
  coefszero::AbstractMatrix     # zeros model coefficients
  intercept::Bool               # whether to include an intercept in each Poisson regression
  n::Int                      # number of observations. May be lower than provided after removing all zero obs.
  d::Int                      # number of categories (terms/words/phrases)
  inpos                         # indices of covars columns included in positives model
  inzero                        # indices of covars columns included in zeros model
  select::Symbol              # path segment selector

  HDMRCoefs(coefspos::AbstractMatrix, coefszero::AbstractMatrix, intercept::Bool,
    n::Int, d::Int, inpos, inzero, select::Symbol) =
    new(coefspos, coefszero, intercept, n, d, inpos, inzero, select)

  function HDMRCoefs(m::HDMRPaths; select=:AICc)
    coefspos, coefszero = coef(m; select=select)
    new(coefspos, coefszero, m.intercept, m.n, m.d, m.inpos, m.inzero, select)
  end
end

"""
    fit(HDMR,covars,counts; <keyword arguments>)
    hdmr(covars,counts; <keyword arguments>)

Fit a Hurdle Distributed Multiple Regression (HDMR) of counts on covars.

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
- `select::Symbol=:AICc` path segment selection criterion
- `kwargs...` additional keyword arguments passed along to fit(Hurdle,...)
"""
function StatsBase.fit(::Type{H}, covars::AbstractMatrix{T}, counts::AbstractMatrix;
  kwargs...) where {T<:AbstractFloat, H<:HDMR}

  hdmr(covars,counts; kwargs...)
end

"""
    fit(HDMRPaths,covars,counts; <keyword arguments>)
    hdmrpaths(covars,counts; <keyword arguments>)

Fit a Hurdle Distributed Multiple Regression (HDMR) of counts on covars, and returns
the entire regulatrization paths, which may be useful for plotting or picking
coefficients other than the AICc optimal ones. Same arguments as
[`fit(::HDMR)`](@ref).
"""
function StatsBase.fit(::Type{HDMRPaths}, covars::AbstractMatrix{T}, counts::AbstractMatrix;
  local_cluster=false, # ignored. will always assume remote_cluster
  kwargs...) where {T<:AbstractFloat}

  hdmrpaths(covars, counts; kwargs...)
end

# fit wrapper that takes a model (two formulas) and dataframe instead of the covars matrix
# e.g. @model(h ~ x1 + x2, c ~ x1)
# h and c on the lhs indicate model for zeros and positives, respectively.
"""
    fit(HDMR,@model(h ~ x1 + x2, c ~ x1),df,counts; <keyword arguments>)

Fits a HDMR but takes a model formula and dataframe instead of the covars matrix.
See also [`fit(::HDMR)`](@ref).

`h` and `c` on the lhs indicate the model for zeros and positives, respectively.
"""
function StatsBase.fit(::Type{T}, m::Model, df::AbstractDataFrame, counts::AbstractMatrix, args...;
  contrasts::Dict = Dict(), kwargs...) where {T<:HDMR}

  # parse and merge rhs terms
  trmszero = getrhsterms(m, :h)
  trmspos = getrhsterms(m, :c)
  trms, inzero, inpos = mergerhsterms(trmszero,trmspos)

  # create model matrix
  mf, mm, counts = createmodelmatrix(trms, df, counts, contrasts)

  # inzero and inpos may be different in mm with factor variables
  inzero, inpos = mapins(inzero, inpos, mm)

  # fit and wrap in DataFrameRegressionModel
  StatsModels.DataFrameRegressionModel(fit(T, mm.m, counts, args...; inzero=inzero, inpos=inpos, kwargs...), mf, mm)
end

"""
    coef(m::HDMRCoefs)

Returns the AICc optimal coefficient matrices fitted with HDMR.

# Example:
```julia
  m = fit(HDMR,covars,counts)
  coefspos, coefszero = coef(m)
```
"""
function StatsBase.coef(m::HDMRCoefs; select=m.select)
  if select == m.select
    m.coefspos, m.coefszero
  else
    error("coef(m::HDMRCoefs) supports only the regulatrization path segement
      selector $(m.select) specified during fit().")
  end
end

"""
    coef(m::HDMRPaths; select=:all)

Returns all or selected coefficient matrices fitted with HDMR.

# Example:
```julia
  m = fit(HDMRPaths,covars,counts)
  coefspos, coefszero = coef(m; select=:CVmin)
```

# Keywords
- `select=:AICc` See [`coef(::RegularizationPath)`](@ref).
"""
function StatsBase.coef(m::HDMRPaths; select=:AICc)
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
  if select==:all
    coefszero = zeros(nλ,pzero,d)
    coefspos = zeros(nλ,ppos,d)
  else
    coefszero = zeros(pzero,d)
    coefspos = zeros(ppos,d)
  end

  # iterate over paths
  for j=1:d
    path = m.nlpaths[j]
    if !ismissing(path)
      cjpos, cjzero = coef(path;select=select)
      if select==:all
        for i=1:ppos
          for s=1:size(cjpos,2)
            coefspos[s,i,j] = cjpos[i,s]
          end
        end
        for i=1:pzero
          for s=1:size(cjzero,2)
            coefszero[s,i,j] = cjzero[i,s]
          end
        end
      else
        for i=1:ppos
          coefspos[i,j] = cjpos[i]
        end
        for i=1:pzero
          coefszero[i,j] = cjzero[i]
        end
      end
    end
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
  countshat = predict(m, newcovars; select=:AICc)
```

# Arguments
- `m::HDMRPaths` fitted DMRPaths model (HDMRCoefs currently not supported)
- `newcovars` n-by-p matrix of covariates of same dimensions used to fit m.

# Keywords
- `select=:AICc` See [`coef(::RegularizationPath)`](@ref).
- `kwargs...` additional keyword arguments passed along to predict() for each
  category j=1..size(counts,2)
"""
function StatsBase.predict(m::HDMRPaths, newcovars::AbstractMatrix{T};
  select=:AICc, kwargs...) where {T<:AbstractFloat}

  covarspos, covarszero = incovars(newcovars,m.inpos,m.inzero)

  _predict(m,covarszero;Xpos=covarspos,select=select,kwargs...)
end

function StatsBase.predict(m::M, newcovars::AbstractMatrix{T};
  select=:AICc, kwargs...) where {T<:AbstractFloat, M<:HDMR}

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

"Shorthand for fit(HDMRPaths,covars,counts). See also [`fit(::HDMRPaths)`](@ref)"
function hdmrpaths(covars::AbstractMatrix{T},counts::AbstractMatrix;
      inpos=1:size(covars,2), inzero=1:size(covars,2),
      intercept=true,
      parallel=true,
      verbose=true, showwarnings=false,
      kwargs...) where {T<:AbstractFloat}
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

  covars, counts, μ, n = shifters(covars, counts, showwarnings)

  covarspos, covarszero = incovars(covars,inpos,inzero)

  function tryfith(countsj::AbstractVector)
    try
      # we make it dense remotely to reduce communication costs
      # we use the same offsets for pos and zeros
      fit(Hurdle,GammaLassoPath,covarszero,Vector(countsj); Xpos=covarspos, offsetpos=μ, offsetzero=μ, verbose=false, showwarnings=showwarnings, kwargs...)
    catch e
      showwarnings && @warn("fit(Hurdle...) failed for countsj with frequencies $(sort(countmap(countsj))) and will return missing path ($e)")
      missing
    end
  end

  # counts generator
  countscols = (counts[:,j] for j=1:d)

  if parallel
    verbose && @info("distributed hurdle run on remote cluster with $(nworkers()) nodes")
    mapfn = pmap
  else
    verbose && @info("serial hurdle run on a single node")
    mapfn = map
  end

  # TODO: the conversion here may be redudant
  nlpaths = convert(Vector{Union{Missing,Hurdle}},mapfn(tryfith,countscols))

  HDMRPaths(nlpaths, intercept, n, d, inpos, inzero)
end

"Returns a (covarspos, covarszero) tuple with views into covars"
function incovars(covars,inpos,inzero)
  inall = 1:size(covars,2)

  if inzero == inall
    covarszero = covars
  else
    covarszero = view(covars,:,inzero)
  end

  if inzero == inpos
    covarspos = covarszero
  elseif inpos == inall
    covarspos = covars
  else
    covarspos = view(covars,:,inpos)
  end

  covarspos, covarszero
end

"Fits a regularized hurdle regression counts[:,j] ~ covars saving the coefficients in coefs[:,j]"
function hurdle_regression!(coefspos::AbstractMatrix{T}, coefszero::AbstractMatrix{T}, j::Int, covars::AbstractMatrix{T},counts::AbstractMatrix{V},
            inpos, inzero;
            offset::AbstractVector=similar(y, 0),
            select=:AICc,
            kwargs...) where {T<:AbstractFloat,V}
  cj = Vector(counts[:,j])
  covarspos, covarszero = incovars(covars,inpos,inzero)
  # we use the same offsets for pos and zeros
  path = fit(Hurdle,GammaLassoPath,covarszero,cj; Xpos=covarspos, offsetpos=offset, offsetzero=offset, kwargs...)
  (coefspos[:,j], coefszero[:,j]) = coef(path; select=select)
  nothing
end

"Wrapper for hurdle_regression! that catches exceptions in which case it sets coefs to zero"
function tryfith!(coefspos::AbstractMatrix{T}, coefszero::AbstractMatrix{T}, j::Int, covars::AbstractMatrix{T},counts::AbstractMatrix{V},
            inpos, inzero;
            showwarnings = false,
            kwargs...) where {T<:AbstractFloat,V}
  try
    hurdle_regression!(coefspos, coefszero, j, covars, counts, inpos, inzero; kwargs...)
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
function hdmr(covars::AbstractMatrix{T},counts::AbstractMatrix{V};
          inpos=1:size(covars,2), inzero=1:size(covars,2),
          intercept=true,
          parallel=true, local_cluster=true,
          verbose=true, showwarnings=false,
          kwargs...) where {T<:AbstractFloat,V}
  if local_cluster || !parallel
    hdmr_local_cluster(covars,counts,inpos,inzero,intercept,parallel,verbose,showwarnings; kwargs...)
  else
    hdmr_remote_cluster(covars,counts,inpos,inzero,intercept,parallel,verbose,showwarnings; kwargs...)
  end
end

"""
This version is built for local clusters and shares memory used by both inputs
and outputs if run in parallel mode.
"""
function hdmr_local_cluster(covars::AbstractMatrix{T},counts::AbstractMatrix{V},
          inpos,inzero,intercept,parallel,verbose,showwarnings;
          select=:AICc,
          kwargs...) where {T<:AbstractFloat,V}
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

  covars, counts, μ, n = shifters(covars, counts, showwarnings)

  # fit separate GammaLassoPath's to each dimension of counts j=1:d and pick its min AICc segment
  if parallel
    verbose && @info("distributed hurdle run on local cluster with $(nworkers()) nodes")
    counts = convert(SharedArray,counts)
    coefszero = SharedMatrix{T}(ncoefzero,d)
    coefspos = SharedMatrix{T}(ncoefpos,d)
    covars = convert(SharedArray,covars)
    # μ = convert(SharedArray,μ) incompatible with GLM

    @sync @distributed for j=1:d
      tryfith!(coefspos, coefszero, j, covars, counts, inpos, inzero; offset=μ,
        verbose=false, showwarnings=showwarnings, intercept=intercept,
        select=select, kwargs...)
    end
  else
    verbose && @info("serial hurdle run on a single node")
    coefszero = Matrix{T}(undef,ncoefzero,d)
    coefspos = Matrix{T}(undef,ncoefpos,d)
    for j=1:d
      tryfith!(coefspos, coefszero, j, covars, counts, inpos, inzero; offset=μ,
        verbose=false, showwarnings=showwarnings, intercept=intercept,
        select=select, kwargs...)
    end
  end

  HDMRCoefs(coefspos, coefszero, intercept, n, d, inpos, inzero, select)
end

"""
This version does not share memory across workers, so may be more efficient for
 small problems, or on remote clusters.
"""
function hdmr_remote_cluster(covars::AbstractMatrix{T},counts::AbstractMatrix{V},
          inpos,inzero,intercept,parallel,verbose,showwarnings;
          select=:AICc, kwargs...) where {T<:AbstractFloat,V}
  paths = hdmrpaths(covars, counts; inpos=inpos, inzero=inzero, parallel=parallel, verbose=verbose, showwarnings=showwarnings, kwargs...)
  HDMRCoefs(paths; select=select)
end

"""
    posindic(A)

Returns an array of the same dimensions of indicators for positive entries in A.
"""
function posindic(A::AbstractArray{T}) where T
  # find positive y entries
  ixpos = (LinearIndices(A))[findall(x->x!=0, A)]

  # build positive indicators matrix
  Ia = deepcopy(A)
  Ia[ixpos] .= one(T)
  Ia
end

"Sparse version simply replaces all the non-zero values with ones."
function posindic(A::SparseMatrixCSC)
  m,n = size(A)
  I,J,V = findnz(A)
  sparse(I, J, fill(1,length(V)), m, n)
end
