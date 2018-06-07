##############################################################
# Distributed Multinomial Regression (DMR)
##############################################################

"Collapse a vector of categories to a DataFrame of indicators"
function collapse(categories)
  cat_indicators = DataFrame()
  unique_categories=union(categories)
  for c in unique_categories
       cat_indicators[symbol(c)] = map(x->ifelse(x == c,1,0),categories)
  end
  (unique_categories,cat_indicators)
end

"Returns a DMRPaths object containing paths by map/pmap-ing a poisson gamma lasso regression to each column of counts separately"
function dmrpaths{T<:AbstractFloat,V}(covars::AbstractMatrix{T},counts::AbstractMatrix{V};
      intercept=true,
      parallel=true,
      verbose=true, showwarnings=false,
      kwargs...)
  # get dimensions
  n, d = size(counts)
  n1,p = size(covars)
  @assert n==n1 "counts and covars should have the same number of observations"
  verbose && info("fitting $n observations on $d categories, $p covariates ")

  covars, counts, μ, n = shifters(covars, counts, showwarnings)

  function tryfitgl(countsj::AbstractVector{V})
    try
      # we make it dense remotely to reduce communication costs
      Nullable{GammaLassoPath}(fit(GammaLassoPath,covars,full(countsj),Poisson(),LogLink(); offset=μ, verbose=false, kwargs...))
    catch e
      showwarnings && warn("fitgl failed for countsj with frequencies $(sort(countmap(countsj))) and will return null path ($e)")
      Nullable{GammaLassoPath}()
    end
  end

  # counts generator
  countscols = (counts[:,j] for j=1:d)

  if parallel
    verbose && info("distributed poisson run on remote cluster with $(nworkers()) nodes")
    mapfn = pmap
  else
    verbose && info("serial poisson run on a single node")
    mapfn = map
  end

  nlpaths = convert(Vector{Nullable{GammaLassoPath}},mapfn(tryfitgl,countscols))

  DMRPaths{T,V}(counts, covars, intercept, nlpaths, n, d, p)
end

function poisson_regression!{T<:AbstractFloat,V}(coefs::AbstractMatrix{T}, j::Int64, covars::AbstractMatrix{T},counts::AbstractMatrix{V}; kwargs...)
  cj = vec(full(counts[:,j]))
  path = fit(GammaLassoPath,covars,cj,Poisson(),LogLink(); kwargs...)
  # coefs[:,j] = vcat(coef(path;select=:AICc)...)
  coefs[:,j] = coef(path;select=:AICc)
  nothing
end

"""
Distributed Multinomial Regression by running independent poisson gamma lasso regression to each column of counts,
picks the minimum AICc segement of each path, and returns a coefficient matrix (coefs) representing point estimates
for the entire multinomial (includes the intercept if one was included).
"""
function dmr{T<:AbstractFloat,V}(covars::AbstractMatrix{T},counts::AbstractMatrix{V};
          intercept=true,
          parallel=true, local_cluster=true,
          verbose=true, showwarnings=false,
          kwargs...)
  if local_cluster || !parallel
    dmr_local_cluster(covars,counts,parallel,verbose,showwarnings,intercept; kwargs...)
  else
    dmr_remote_cluster(covars,counts,parallel,verbose,showwarnings,intercept; kwargs...)
  end
end

"Abstract Distributed Counts Regression (DCR) returned object"
abstract type DCR{T<:AbstractFloat,V} <: RegressionModel end

"Abstract DMR returned object"
abstract type DMR{T<:AbstractFloat,V} <: DCR{T,V} end

"""
Relatively heavy object used to return results when we care about the regulatrization paths.
It is returned whenever we use a remote cluster.
"""
struct DMRPaths{T<:AbstractFloat,V} <: DMR{T,V}
  counts::AbstractMatrix{V}     # n×d counts (document-term) matrix
  covars::AbstractMatrix{T}     # n×p covariates matrix
  intercept::Bool               # whether to include an intercept in each Poisson regression
  nlpaths::Vector{Nullable{GammaLassoPath}} # independent Poisson GammaLassoPath for each phrase
                                # (only kept with remote cluster, not with local cluster)
  n::Int64                      # number of observations. May be lower than provided after removing all zero obs.
  d::Int64                      # number of categories (terms/words/phrases)
  p::Int64                      # number of covariates

  DMRPaths{T,V}(counts::AbstractMatrix{V}, covars::AbstractMatrix{T}, intercept::Bool,
    nlpaths::Vector{Nullable{GammaLassoPath}}, n::Int64, d::Int64, p::Int64) where {T<:AbstractFloat,V} =
    new(counts, covars, intercept, nlpaths, n, d, p)
end

"""
Relatively light object used to return results when we only care about estimated coefficients.
It is returned whenever we use a local cluster.
"""
struct DMRCoefs{T<:AbstractFloat,V} <: DMR{T,V}
  coefs::AbstractMatrix{T}      # model coefficients
  intercept::Bool               # whether to include an intercept in each Poisson regression
  n::Int64                      # number of observations. May be lower than provided after removing all zero obs.
  d::Int64                      # number of categories (terms/words/phrases)
  p::Int64                      # number of covariates

  DMRCoefs{T,V}(coefs::AbstractMatrix{T}, intercept::Bool, n::Int64, d::Int64, p::Int64) where {T<:AbstractFloat,V} =
    new(coefs, intercept, n, d, p)
end

function StatsBase.fit(::Type{D}, covars::AbstractMatrix{T}, counts::AbstractMatrix{V};
  intercept=true, parallel=true, local_cluster=true, verbose=true, showwarnings=false,
  kwargs...) where {T<:AbstractFloat, V, D<:DMR}

  if local_cluster || !parallel
    dmr_local_cluster(covars,counts,parallel,verbose,showwarnings,intercept; kwargs...)
  else
    dmr_remote_cluster(covars,counts,parallel,verbose,showwarnings,intercept; kwargs...)
  end
end

function StatsBase.fit(::Type{DMRPaths}, covars::AbstractMatrix{T}, counts::AbstractMatrix{V};
  kwargs...) where {T<:AbstractFloat, V}

  dmrpaths(covars, counts; kwargs...)
end

hasintercept(m::DCR) = m.intercept

"Number of observations used. May be lower than provided after removing all zero obs."
StatsBase.nobs(m::DCR) = m.n

"Number of categories (terms/words/phrases) used for DMR estimation"
Distributions.ncategories(m::DCR) = m.d

"Number of covariates used for DMR estimation"
ncovars(m::DMR) = m.p

"Number of coefficient potentially including intercept used for each independent Poisson regression"
ncoefs(m::DMR) = ncovars(m) + (hasintercept(m) ? 1 : 0)

# some helpers for converting to SharedArray
Base.convert(::Type{SharedArray}, A::SubArray) = (S = SharedArray{eltype(A)}(size(A)); copy!(S, A))
function Base.convert(::Type{SharedArray}, A::SparseMatrixCSC)
  S = SharedArray{eltype(A)}(size(A))
  rows = rowvals(A)
  vals = nonzeros(A)
  n, m = size(A)
  for j = 1:m
     for i in nzrange(A, j)
        row = rows[i]
        val = vals[i]
        S[row,j] = val
     end
  end
  S
end

# shifters
function shifters{T<:AbstractFloat,V}(covars::AbstractMatrix{T}, counts::AbstractMatrix{V}, showwarnings::Bool)
    m = vec(sum(counts,2))

    if any(iszero,m)
        # omit observations with no counts
        ixposm = find(m)
        showwarnings && warn("omitting $(length(m)-length(ixposm)) observations with no counts")
        m = m[ixposm]
        counts = counts[ixposm,:]
        covars = covars[ixposm,:]
    end

    μ = log.(m)
    # display(μ)

    n = length(m)

    covars, counts, μ, n
end

"""
This version is built for local clusters and shares memory used by both inputs and outputs if run in parallel mode.
"""
function dmr_local_cluster{T<:AbstractFloat,V}(covars::AbstractMatrix{T},counts::AbstractMatrix{V},
          parallel,verbose,showwarnings,intercept; kwargs...)
  # get dimensions
  n, d = size(counts)
  n1,p = size(covars)
  @assert n==n1 "counts and covars should have the same number of observations"
  verbose && info("fitting $n observations on $d categories, $p covariates ")

  # add one coef for intercept
  ncoef = p + (intercept ? 1 : 0)

  covars, counts, μ, n = shifters(covars, counts, showwarnings)

  function tryfitgl!(coefs::AbstractMatrix{T}, j::Int64, covars::AbstractMatrix{T},counts::AbstractMatrix{V}; kwargs...)
    try
      poisson_regression!(coefs, j, covars, counts; kwargs...)
    catch e
      showwarnings && warn("fitgl! failed on count dimension $j with frequencies $(sort(countmap(counts[:,j]))) and will return zero coefs ($e)")
      # redudant ASSUMING COEFS ARRAY INTIAILLY FILLED WITH ZEROS, but can be uninitialized in serial case
      for i=1:size(coefs,1)
        coefs[i,j] = zero(T)
      end
    end
  end

  # fit separate GammaLassoPath's to each dimension of counts j=1:d and pick its min AICc segment
  if parallel
    verbose && info("distributed poisson run on local cluster with $(nworkers()) nodes")
    counts = convert(SharedArray,counts)
    coefs = SharedMatrix{T}(ncoef,d)
    covars = convert(SharedArray,covars)
    # μ = convert(SharedArray,μ) incompatible with GLM

    @sync @parallel for j=1:d
      tryfitgl!(coefs, j, covars, counts; offset=μ, verbose=false, intercept=intercept, kwargs...)
    end
  else
    verbose && info("serial poisson run on a single node")
    coefs = Matrix{T}(ncoef,d)
    for j=1:d
      tryfitgl!(coefs, j, covars, counts; offset=μ, verbose=false, intercept=intercept, kwargs...)
    end
  end

  DMRCoefs{T,V}(coefs, intercept, n, d, p)
end

"""
This version does not share memory across workers, so may be more efficient for small problems, or on remote clusters.
"""
function dmr_remote_cluster{T<:AbstractFloat,V}(covars::AbstractMatrix{T},counts::AbstractMatrix{V},
          parallel,verbose,showwarnings,intercept; kwargs...)
  paths = dmrpaths(covars, counts; parallel=parallel, verbose=verbose, showwarnings=showwarnings, intercept=intercept, kwargs...)
  coefs = coef(paths;select=:AICc)
  DMRCoefs{T,V}(coefs, intercept, paths.n, paths.d, paths.p)
end

dropnull{T<:Nullable}(v::Vector{T}) = v[.!isnull.(v)]

function StatsBase.coef(m::DMRCoefs; select=:AICc)
  if select == :AICc
    m.coefs
  else
    error("coef(m::DMRCoefs) currently supports only AICc regulatrization path segement selection.")
  end
end

function StatsBase.coef(m::DMRPaths; select=:AICc)
  # get dims
  d = length(m.nlpaths)
  d < 1 && return nothing

  # drop null paths
  nonnullpaths = dropnull(m.nlpaths)

  # get number of coefs from paths object
  p = ncoefs(m)

  # establish maximum path lengths
  nλ = 0
  if size(nonnullpaths,1) > 0
    nλ = maximum(map(nlpath->size(nlpath.value)[2],nonnullpaths))
  end

  # allocate space
  if select==:all
    coefs = zeros(nλ,p,d)
  else
    coefs = zeros(p,d)
  end

  # iterate over paths
  for j=1:d
    nlpath = m.nlpaths[j]
    if !Base.isnull(nlpath)
      path = nlpath.value
      cj = coef(path;select=select)
      if select==:all
        for i=1:p
          for s=1:size(cj,2)
            coefs[s,i,j] = cj[i,s]
          end
        end
      else
        for i=1:p
          coefs[i,j] = cj[i]
        end
      end
    end
  end

  coefs
end

aicc{R<:RegularizationPath}(paths::Vector{R}; k=2) = map(path->Lasso.aicc(path;k=k),paths)
