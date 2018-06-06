##############################################################
# Hurdle Distributed Multinomial Regression (HDMR)
##############################################################

# immutable HDMRPaths
#     nlpaths::Vector{Nullable{Hurdle}}
#     ppos::Int64
#     pzero::Int64
# end

"Abstract HDMR returned object"
abstract type HDMR{T<:AbstractFloat,V} <: DCR{T,V} end

"""
Relatively heavy object used to return results when we care about the regulatrization paths.
It is returned whenever we use a remote cluster.
"""
struct HDMRPaths{T<:AbstractFloat,V} <: HDMR{T,V}
  counts::AbstractMatrix{V}     # n×d counts (document-term) matrix
  covars::AbstractMatrix{T}     # n×p covariates matrix for zeros or both model
  covarspos::Union{AbstractMatrix{T},Void}  # n×ppos covariates matrix for positives model
  intercept::Bool               # whether to include an intercept in each Poisson regression
  nlpaths::Vector{Nullable{Hurdle}} # independent Hurdle{GammaLassoPath} for each phrase
                                # (only kept with remote cluster, not with local cluster)
  n::Int64                      # number of observations. May be lower than provided after removing all zero obs.
  d::Int64                      # number of categories (terms/words/phrases)
  pzero::Int64                  # number of covariates for zeros model
  ppos::Int64                   # number of covariates for positives model

  HDMRPaths{T,V}(counts::AbstractMatrix{V}, covars::AbstractMatrix{T}, covarspos::Union{AbstractMatrix{T},Void}, intercept::Bool,
    nlpaths::Vector{Nullable{Hurdle}}, n::Int64, d::Int64, pzero::Int64, ppos::Int64) where {T<:AbstractFloat,V} =
    new(counts, covars, covarspos, intercept, nlpaths, n, d, pzero, ppos)
end

"""
Relatively light object used to return results when we only care about estimated coefficients.
It is returned whenever we use a local cluster.
"""
struct HDMRCoefs{T<:AbstractFloat,V} <: HDMR{T,V}
  coefspos::AbstractMatrix{T}   # positives model coefficients
  coefszero::AbstractMatrix{T}  # zeros model coefficients
  intercept::Bool               # whether to include an intercept in each Poisson regression
  n::Int64                      # number of observations. May be lower than provided after removing all zero obs.
  d::Int64                      # number of categories (terms/words/phrases)
  pzero::Int64                  # number of covariates for zeros model
  ppos::Int64                   # number of covariates for positives model

  HDMRCoefs{T,V}(coefspos::AbstractMatrix{T}, coefszero::AbstractMatrix{T}, intercept::Bool,
    n::Int64, d::Int64, pzero::Int64, ppos::Int64) where {T<:AbstractFloat,V} =
    new(coefspos, coefszero, intercept, n, d, pzero, ppos)

  function HDMRCoefs{T,V}(paths::HDMRPaths{T,V}) where {T<:AbstractFloat,V}
    coefspos, coefszero = coef(paths;select=:AICc)
    new(coefspos, coefszero, paths.intercept, paths.n, paths.d, paths.pzero, paths.ppos)
  end
end


function StatsBase.fit(::Type{H}, covars::AbstractMatrix{T}, counts::AbstractMatrix{V};
  covarspos::Union{AbstractMatrix{T},Void} = nothing,
  kwargs...) where {T<:AbstractFloat, V, H<:HDMR}

  hdmrpaths(covars, counts; covarspos=covarspos, kwargs...)
end

function StatsBase.fit(::Type{HDMRCoefs}, covars::AbstractMatrix{T}, counts::AbstractMatrix{V};
  covarspos::Union{AbstractMatrix{T},Void} = nothing,
  intercept=true, parallel=true, local_cluster=true, verbose=true, showwarnings=false,
  kwargs...) where {T<:AbstractFloat, V}

  if local_cluster || !parallel
    hdmr_local_cluster(covars,covarspos,counts,parallel,verbose,showwarnings,intercept; kwargs...)
  else
    hdmr_remote_cluster(covars,covarspos,counts,parallel,verbose,showwarnings,intercept; kwargs...)
  end
end

"Number of covariates used for HDMR estimation of zeros model"
ncovarszero(m::HDMR) = m.pzero

"Number of covariates used for HDMR estimation of positives model"
ncovarspos(m::HDMR) = m.ppos

"Number of coefficient potentially including intercept used by model for zeros"
ncoefszero(m::HDMR) = ncovarszero(m) + (hasintercept(m) ? 1 : 0)

"Number of coefficient potentially including intercept used by model for positives"
ncoefspos(m::HDMR) = ncovarspos(m) + (hasintercept(m) ? 1 : 0)

# shifters
function shifters{T<:AbstractFloat,V}(covars::AbstractMatrix{T}, covarspos::Union{Void,AbstractMatrix{T}}, counts::AbstractMatrix{V}, showwarnings::Bool)
    m = vec(sum(counts,2))

    if any(iszero,m)
        # omit observations with no counts
        ixposm = find(m)
        showwarnings && warn("omitting $(length(m)-length(ixposm)) observations with no counts")
        m = m[ixposm]
        counts = counts[ixposm,:]
        covars = covars[ixposm,:]
        if covarspos != nothing
          covarspos = covarspos[ixposm,:]
        end
    end

    μ = log.(m)
    # display(μ)

    n = length(m)

    covars, covarspos, counts, μ, n
end

"Returns a vector of paths by map/pmap-ing a hurlde gamma lasso regression to each column of counts separately"
function hdmrpaths{T<:AbstractFloat,V}(covars::AbstractMatrix{T},counts::AbstractMatrix{V};
      covarspos::Union{AbstractMatrix{T},Void} = nothing,
      intercept=true,
      parallel=true,
      verbose=true, showwarnings=false,
      kwargs...)
  # get dimensions
  n, d = size(counts)
  n1,p = size(covars)
  @assert n==n1 "counts and covars should have the same number of observations"
  verbose && info("fitting $n observations on $d categories, $p covariates ")

  # hurdle optionally allows for different covarspos
  if covarspos != nothing
    n1,ppos = size(covarspos)
    @assert n==n1 "counts and covarspos should have the same number of observations"
    verbose && info("for zeros and $ppos covariates for positive counts")
  else
    ppos = p
    verbose && info("for both zeros and positive counts")
  end

  covars, covarspos, counts, μ, n = shifters(covars, covarspos, counts, showwarnings)

  function tryfith(countsj::AbstractVector{V})
    try
      # we make it dense remotely to reduce communication costs
      # we use the same offsets for pos and zeros
      Nullable{Hurdle}(fit(Hurdle,GammaLassoPath,covars,full(countsj); Xpos=covarspos, offsetpos=μ, offsetzero=μ, verbose=false, showwarnings=showwarnings, kwargs...))
    catch e
      showwarnings && warn("fit(Hurdle...) failed for countsj with frequencies $(sort(countmap(countsj))) and will return null path ($e)")
      Nullable{Hurdle}()
    end
  end

  # counts generator
  countscols = (counts[:,j] for j=1:d)

  if parallel
    verbose && info("distributed hurdle run on remote cluster with $(nworkers()) nodes")
    mapfn = pmap
  else
    verbose && info("serial hurdle run on a single node")
    mapfn = map
  end

  nlpaths = convert(Vector{Nullable{Hurdle}},mapfn(tryfith,countscols))

  HDMRPaths{T,V}(counts, covars, covarspos, intercept, nlpaths, n, d, p, ppos)
end

function hurdle_regression!{T<:AbstractFloat,V}(coefspos::AbstractMatrix{T}, coefszero::AbstractMatrix{T}, j::Int64, covars::AbstractMatrix{T},counts::AbstractMatrix{V};
            covarspos::Union{AbstractMatrix{T},Void} = nothing,
            offset::AbstractVector=similar(y, 0),
            kwargs...)
  cj = vec(full(counts[:,j]))
  # we use the same offsets for pos and zeros
  path = fit(Hurdle,GammaLassoPath,covars,cj; Xpos=covarspos, offsetpos=offset, offsetzero=offset, kwargs...)
  (coefspos[:,j], coefszero[:,j]) = coef(path;select=:AICc)
  nothing
end

"""
Distributed Multinomial Regression by running independent poisson gamma lasso regression to each column of counts,
picks the minimum AICc segement of each path, and returns a coefficient matrix (coefs) representing point estimates
for the entire multinomial (includes the intercept if one was included).
"""
function hdmr{T<:AbstractFloat,V}(covars::AbstractMatrix{T},counts::AbstractMatrix{V};
          covarspos::Union{AbstractMatrix{T},Void} = nothing,
          intercept=true,
          parallel=true, local_cluster=true,
          verbose=true, showwarnings=false,
          kwargs...)
  if local_cluster || !parallel
    hdmr_local_cluster(covars,covarspos,counts,parallel,verbose,showwarnings,intercept; kwargs...)
  else
    hdmr_remote_cluster(covars,covarspos,counts,parallel,verbose,showwarnings,intercept; kwargs...)
  end
end

"""
This version is built for local clusters and shares memory used by both inputs and outputs if run in parallel mode.
"""
function hdmr_local_cluster{T<:AbstractFloat,V}(covars::AbstractMatrix{T},covarspos::Union{AbstractMatrix{T},Void},counts::AbstractMatrix{V},
          parallel,verbose,showwarnings,intercept; kwargs...)
  # get dimensions
  n, d = size(counts)
  n1,p = size(covars)
  @assert n==n1 "counts and covars should have the same number of observations"
  verbose && info("fitting $n observations on $d categories, $p covariates ")

  # add one coef for intercept
  ncoefzero = p + (intercept ? 1 : 0)

  # hurdle optionally allows for different covarspos
  ppos = 0
  if covarspos != nothing
    n1,ppos = size(covarspos)
    @assert n==n1 "counts and covarspos should have the same number of observations"
    verbose && info("for zeros and $ppos covariates for positive counts")
    ncoefpos = ppos + (intercept ? 1 : 0)
  else
    ppos = p
    ncoefpos = ncoefzero
    verbose && info("for both zeros and positive counts")
  end

  covars, covarspos, counts, μ, n = shifters(covars, covarspos, counts, showwarnings)

  function tryfith!(coefspos::AbstractMatrix{T}, coefszero::AbstractMatrix{T}, j::Int64, covars::AbstractMatrix{T},counts::AbstractMatrix{V};
    covarspos::Union{AbstractMatrix{T},Void} = nothing, kwargs...)
    try
      hurdle_regression!(coefspos, coefszero, j, covars, counts; covarspos=covarspos, kwargs...)
    catch e
      showwarnings && warn("hurdle_regression! failed on count dimension $j with frequencies $(sort(countmap(counts[:,j]))) and will return zero coefs ($e)")
      # redudant ASSUMING COEFS ARRAY INTIAILLY FILLED WITH ZEROS, but can happen in serial mode
      for i=1:size(coefszero,1)
        coefszero[i,j] = zero(T)
      end
      for i=1:size(coefspos,1)
        coefspos[i,j] = zero(T)
      end
    end
  end

  # fit separate GammaLassoPath's to each dimension of counts j=1:d and pick its min AICc segment
  if parallel
    verbose && info("distributed hurdle run on local cluster with $(nworkers()) nodes")
    counts = convert(SharedArray,counts)
    coefszero = SharedMatrix{T}(ncoefzero,d)
    coefspos = SharedMatrix{T}(ncoefpos,d)
    covars = convert(SharedArray,covars)
    # μ = convert(SharedArray,μ) incompatible with GLM

    if covarspos != nothing
      covarspos = convert(SharedArray,covarspos)
    end

    @sync @parallel for j=1:d
      tryfith!(coefspos, coefszero, j, covars, counts; covarspos=covarspos, offset=μ, verbose=false, intercept=intercept, kwargs...)
    end
  else
    verbose && info("serial hurdle run on a single node")
    coefszero = Matrix{T}(ncoefzero,d)
    coefspos = Matrix{T}(ncoefpos,d)
    for j=1:d
      tryfith!(coefspos, coefszero, j, covars, counts; covarspos=covarspos, offset=μ, verbose=false, intercept=intercept, kwargs...)
    end
  end

  HDMRCoefs{T,V}(coefspos, coefszero, intercept, n, d, p, ppos)
end

"""
This version does not share memory across workers, so may be more efficient for small problems, or on non remote clusters.
"""
function hdmr_remote_cluster{T<:AbstractFloat,V}(covars::AbstractMatrix{T},covarspos::Union{AbstractMatrix{T},Void},counts::AbstractMatrix{V},
          parallel,verbose,showwarnings,intercept; kwargs...)
  paths = hdmrpaths(covars, counts; covarspos=covarspos, parallel=parallel, verbose=verbose, showwarnings=showwarnings, kwargs...)
  HDMRCoefs{T,V}(paths)
end

function StatsBase.coef(m::HDMRCoefs; select=:AICc)
  if select == :AICc
    m.coefspos, m.coefszero
  else
    error("coef(m::HDMRCoefs) currently supports only AICc regulatrization path segement selection.")
  end
end

function StatsBase.coef(m::HDMRPaths; select=:all)
  # get dims
  d = length(m.nlpaths)
  d < 1 && return nothing, nothing

  # drop null paths
  nonnullpaths = dropnull(m.nlpaths)

  # get number of variables from paths object
  ppos = ncoefspos(m)
  pzero = ncoefszero(m)

  # establish maximum path lengths
  nλzero = nλpos = 0
  if size(nonnullpaths,1) > 0
    nλzero = maximum(map(nlpath->size(nlpath.value.mzero)[2],nonnullpaths))
    nλpos = maximum(map(nlpath->size(nlpath.value.mpos)[2],nonnullpaths))
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
    nlpath = m.nlpaths[j]
    if !Base.isnull(nlpath)
      path = nlpath.value
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


"returns an array of the same dimensions of indicators for positive entries in A"
function posindic(A::AbstractArray)
  # find positive y entries
  ixpos = find(A)

  # build positive indicators matrix
  Ia = deepcopy(A)
  Ia[ixpos] = 1
  Ia
end

# sparse version simply replaces all the non-zero values with ones
function posindic(A::SparseMatrixCSC)
  m,n = size(A)
  I,J,V = findnz(A)
  sparse(I,J,ones(V),m,n)
end
