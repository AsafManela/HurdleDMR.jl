##############################################################
# Distributed Multinomial Regression (DMR)
##############################################################

immutable DMRPaths
  nlpaths::Vector{Nullable{GammaLassoPath}}
  p::Int64 # number of covariates
end

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
      verbose=true, showwarnings=true,
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
      showwarnings && warn("fitgl failed for countsj=$countsj and will return null path ($e)")
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

  DMRPaths(convert(Vector{Nullable{GammaLassoPath}},mapfn(tryfitgl,countscols)), p)
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
          verbose=true, showwarnings=true,
          kwargs...)
  if local_cluster || !parallel
    dmr_local_cluster(covars,counts,parallel,verbose,showwarnings,intercept; kwargs...)
  else
    dmr_remote_cluster(covars,counts,parallel,verbose,showwarnings,intercept; kwargs...)
  end
end

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
  if intercept
    p += 1
  end

  covars, counts, μ, n = shifters(covars, counts, showwarnings)

  function tryfitgl!(coefs::AbstractMatrix{T}, j::Int64, covars::AbstractMatrix{T},counts::AbstractMatrix{V}; kwargs...)
    try
      poisson_regression!(coefs, j, covars, counts; kwargs...)
    catch e
      showwarnings && warn("fitgl! failed on count dimension $j and will return zero coefs ($e)")
      # redudant ASSUMING COEFS ARRAY INTIAILLY FILLED WITH ZEROS
      # for i=1:size(coefs,1)
      #   coefs[i,j] = zero(T)
      # end
    end
  end

  # fit separate GammaLassoPath's to each dimension of counts j=1:d and pick its min AICc segment
  if parallel
    verbose && info("distributed poisson run on local cluster with $(nworkers()) nodes")
    counts = convert(SharedArray,counts)
    coefs = SharedArray{T}(p,d)
    covars = convert(SharedArray,covars)
    # μ = convert(SharedArray,μ) incompatible with GLM

    @sync @parallel for j=1:d
      tryfitgl!(coefs, j, covars, counts; offset=μ, verbose=false, intercept=intercept, kwargs...)
    end
  else
    verbose && info("serial poisson run on a single node")
    coefs = Array(T,p,d)
    for j=1:d
      tryfitgl!(coefs, j, covars, counts; offset=μ, verbose=false, intercept=intercept, kwargs...)
    end
  end

  coefs
end

"""
This version does not share memory across workers, so may be more efficient for small problems, or on non remote clusters.
"""
function dmr_remote_cluster{T<:AbstractFloat,V}(covars::AbstractMatrix{T},counts::AbstractMatrix{V},
          parallel,verbose,showwarnings,intercept; kwargs...)
  paths = dmrpaths(covars, counts; parallel=parallel, verbose=verbose, showwarnings=showwarnings, intercept=intercept, kwargs...)
  coef(paths;select=:AICc)
end

dropnull{T<:Nullable}(v::Vector{T}) = v[.!isnull.(v)]

function StatsBase.coef(paths::DMRPaths; select=:all)
  # get dims
  d = length(paths.nlpaths)
  d < 1 && return nothing

  # drop null paths
  nonnullpaths = dropnull(paths.nlpaths)

  # get number of variables from paths object
  p = paths.p + 1

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
    nlpath = paths.nlpaths[j]
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

"""
srproj calculates the MNIR Sufficient Reduction projection from text counts on
to the attribute dimensions of interest (covars in mnlm). In particular, for
counts C, with row sums m, and mnlm coefficients φ_j corresponding to attribute
j, z_j = C'φ_j/m is the SR projection in the direction of j.
The MNIR paper explains how V=[v_1 ... v_K],
your original covariates/attributes, are independent of text counts C given SR
projections Z=[z_1 ... z_K].

Note: for hurdle models, best to give only the relevant subset of the coefs
matrix (including intercept if one is included)
"""
function srproj(coefs, counts, dir=nothing; intercept=true)
   ixoffset = intercept ? 1 : 0 # omitting the intercept
   if dir==nothing
     Φ=coefs[ixoffset+1:end,:] # all directions
   else
     Φ=coefs[ixoffset+dir:ixoffset+dir,:] # keep only desired directions
   end
   m = sum(counts,2) # total counts per observation
   z = counts*Φ' ./ (m+(m.==0)) # scale to get frequencies
   [z m] # m is part of the sufficient reduction
end

"""
Like srproj but efficiently interates over a sparse counts matrix, and
only projects in a single direction (dir).
"""
function srproj(coefs, counts::SparseMatrixCSC, dir::Int; intercept=true)
   ixoffset = intercept ? 1 : 0 # omitting the intercept
   n,d = size(counts)
   zm = zeros(n,2)
   φ = vec(coefs[ixoffset+dir,:]) # keep only desired directions
   rows = rowvals(counts)
   vals = nonzeros(counts)
   for j = 1:d
      for i in nzrange(counts, j)
         row = rows[i]
         val = vals[i]
         zm[row,1] += val*φ[j]  # projection part
         zm[row,2] += val       # m = total count
      end
   end
   for i=1:n
     mi = zm[i,2]
     if mi > 0
       # scale to get frequencies
       zm[i,1] /= mi
     end
   end
   zm # m is part of the sufficient reduction
end

"""
  Builds the design matrix X for predicting covar in direction projdir
  dmr version
"""
function srprojX(coefs,counts,covars,projdir; includem=true)
  # dims
  n,p = size(covars)
  # ixnotdir = 1:p .!= projdir
  ixnotdir = setdiff(1:p,[projdir])

  # design matrix w/o counts data
  X_nocounts = [ones(n) getindex(covars,:,ixnotdir)]

  # add srproj of counts data to X
  Z = srproj(coefs,counts,projdir)
  X = [X_nocounts Z]

  if !includem
    # drop last column with total counts m
    X = X[:,1:end-1]
  end

  X, X_nocounts
end
