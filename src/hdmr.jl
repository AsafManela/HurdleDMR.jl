##############################################################
# Hurdle Distributed Multinomial Regression (HDMR)
##############################################################

immutable HDMRPaths
    nlpaths::Vector{Nullable{Hurdle}}
    ppos::Int64
    pzero::Int64
end

"Returns a vector of paths by map/pmap-ing a hurlde gamma lasso regression to each column of counts separately"
function hdmrpaths{T<:AbstractFloat,V}(covars::AbstractMatrix{T},counts::AbstractMatrix{V};
      covarspos::@compat(Union{AbstractMatrix{T},Void}) = nothing,
      intercept=true,
      parallel=true,
      verbose=true, showwarnings=true,
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

  # shifters
  m = sum(counts,2)
  μ = vec(log.(m))
  # display(μ)

  function tryfith(countsj::SparseVector{V,Int64})
    try
      # we make it dense remotely to reduce communication costs
      # we use the same offsets for pos and zeros
      Nullable{Hurdle}(fit(Hurdle,GammaLassoPath,covars,full(countsj); Xpos=covarspos, offsetpos=μ, offsetzero=μ, verbose=false, kwargs...))
    catch e
      showwarnings && warn("fit(Hurdle...) failed for countsj with frequencies $(countmap(countsj)) and will return null path ($e)")
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

  HDMRPaths(convert(Vector{Nullable{Hurdle}},mapfn(tryfith,countscols)), ppos, p)
end

function hurdle_regression!{T<:AbstractFloat,V}(coefszero::AbstractMatrix{T}, coefspos::AbstractMatrix{T}, j::Int64, covars::AbstractMatrix{T},counts::AbstractMatrix{V};
            covarspos::@compat(Union{AbstractMatrix{T},Void}) = nothing,
            offset::AbstractVector=similar(y, 0),
            kwargs...)
  cj = vec(full(counts[:,j]))
  # we use the same offsets for pos and zeros
  path = fit(Hurdle,GammaLassoPath,covars,cj; Xpos=covarspos, offsetpos=offset, offsetzero=offset, kwargs...)
  (coefspos[:,j], coefszero[:,j]) = coef(path;select=:AICc)
  # coefs[:,j] = vcat(coef(path;select=:AICc)...)
  nothing
end

"""
Distributed Multinomial Regression by running independent poisson gamma lasso regression to each column of counts,
picks the minimum AICc segement of each path, and returns a coefficient matrix (coefs) representing point estimates
for the entire multinomial (includes the intercept if one was included).
"""
function hdmr{T<:AbstractFloat,V}(covars::AbstractMatrix{T},counts::AbstractMatrix{V};
          covarspos::@compat(Union{AbstractMatrix{T},Void}) = nothing,
          intercept=true,
          parallel=true, local_cluster=true,
          verbose=true, showwarnings=true,
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
function hdmr_local_cluster{T<:AbstractFloat,V}(covars::AbstractMatrix{T},covarspos::@compat(Union{AbstractMatrix{T},Void}),counts::AbstractMatrix{V},
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

  # hurdle optionally allows for different covarspos
  ppos = 0
  if covarspos != nothing
    n1,ppos = size(covarspos)
    @assert n==n1 "counts and covarspos should have the same number of observations"
    verbose && info("for zeros and $ppos covariates for positive counts")
    if intercept
      ppos += 1
    end
  else
    ppos = p
    verbose && info("for both zeros and positive counts")
  end

  # shifters
  m = sum(counts,2)
  μ = vec(log.(m))
  # display(μ)

  function tryfith!(coefszero::AbstractMatrix{T}, coefspos::AbstractMatrix{T}, j::Int64, covars::AbstractMatrix{T},counts::AbstractMatrix{V};
    covarspos::@compat(Union{AbstractMatrix{T},Void}) = nothing, kwargs...)
    try
      hurdle_regression!(coefszero, coefspos, j, covars, counts; covarspos=covarspos, kwargs...)
    catch e
      showwarnings && warn("hurdle_regression! failed on count dimension $j with frequencies $(countmap(counts[:,j])) and will return zero coefs ($e)")
      # redudant ASSUMING COEFS ARRAY INTIAILLY FILLED WITH ZEROS
      # for i=1:size(coefs,1)
      #   coefs[i,j] = zero(T)
      # end
    end
  end

  # fit separate GammaLassoPath's to each dimension of counts j=1:d and pick its min AICc segment
  if parallel
    verbose && info("distributed hurdle run on local cluster with $(nworkers()) nodes")
    counts = convert(SharedArray,counts)
    coefszero = SharedArray{T}(p,d)
    coefspos = SharedArray{T}(ppos,d)
    covars = convert(SharedArray,covars)
    # μ = convert(SharedArray,μ) incompatible with GLM

    if covarspos != nothing
      covarspos = convert(SharedArray,covarspos)
    end

    @sync @parallel for j=1:d
      tryfith!(coefszero, coefspos, j, covars, counts; covarspos=covarspos, offset=μ, verbose=false, intercept=intercept, kwargs...)
    end
  else
    verbose && info("serial hurdle run on a single node")
    coefszero = Array(T,p,d)
    coefspos = Array(T,ppos,d)
    for j=1:d
      tryfith!(coefszero, coefspos, j, covars, counts; covarspos=covarspos, offset=μ, verbose=false, intercept=intercept, kwargs...)
    end
  end

  coefspos, coefszero
end

"""
This version does not share memory across workers, so may be more efficient for small problems, or on non remote clusters.
"""
function hdmr_remote_cluster{T<:AbstractFloat,V}(covars::AbstractMatrix{T},covarspos::@compat(Union{AbstractMatrix{T},Void}),counts::AbstractMatrix{V},
          parallel,verbose,showwarnings,intercept; kwargs...)
  paths = hdmrpaths(covars, counts; covarspos=covarspos, parallel=parallel, verbose=verbose, showwarnings=showwarnings, kwargs...)
  coef(paths;select=:AICc)
end

function StatsBase.coef(paths::HDMRPaths; select=:all)
  # get dims
  d = length(paths.nlpaths)
  d < 1 && return nothing, nothing

  # drop null paths
  nonnullpaths = dropnull(paths.nlpaths)

  # get number of variables from paths object
  ppos = paths.ppos + 1
  pzero = paths.pzero + 1

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
    nlpath = paths.nlpaths[j]
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

"""
srproj for hurdle dmr takes two coefficent matrices
coefspos, coefszero, and a specific direction
and returns an n-by-3 matrix Z = [zpos zzero m]
"""
function srproj(coefspos, coefszero, counts, dirpos::Int, dirzero::Int; intercept=true)
  if dirpos>0 && dirzero>0
    zpos = srproj(coefspos, counts, dirpos; intercept=intercept)
    zzero = srproj(coefszero, posindic(counts), dirzero; intercept=intercept)
    # second element should be same m in both, but because zero model
    # only sums indicators it generates smaller totals, so use the one
    # from the pos model
    # TODO: this needs to be fleshed out better in the theory to guide this choice
    [zpos[:,1] zzero[:,1] zpos[:,2]]
  elseif dirpos>0
    srproj(coefspos, counts, dirpos; intercept=intercept)
  elseif dirzero>0
    srproj(coefszero, posindic(counts), dirzero; intercept=intercept)
  else
    error("No direction to project to (dirpos=$dirpos,dirzero=$dirzero)")
  end
end

function ixcovars(p::Int, dir::Int, inpos, inzero)
  # @assert dir ∈ inzero "projection direction $dir must be included in coefzero estimation!"
  # @assert dir ∈ inpos "projection direction $dir must be included in coefpos estimation!"

  # findfirst returns 0 if not found
  dirpos = findfirst(inpos,dir)
  dirzero = findfirst(inzero,dir)

  ineither = union(inzero,inpos)
  ixnotdir = setdiff(ineither,[dir])

  dirpos,dirzero,ineither,ixnotdir
end

"""
  Builds the design matrix X for predicting covar in direction projdir
  hdmr version
"""
function srprojX(coefspos, coefszero, counts, covars, dir::Int; inpos=1:size(covars,2), inzero=1:size(covars,2), includem=true, includezpos=true)
  # dims
  n,p = size(covars)

  # get pos and zero subset indecies
  dirpos,dirzero,ineither,ixnotdir = ixcovars(p, dir, inpos, inzero)

  # design matrix w/o counts data
  X_nocounts = [ones(n) getindex(covars,:,ixnotdir)]

  # add srproj of counts data to X
  if includezpos
    Z = srproj(coefspos, coefszero, counts, dirpos, dirzero; intercept=true)
    X = [X_nocounts Z]
  end

  if !includezpos || rank(X) < size(X,2)
    if includezpos
      info("rank(X) = $(rank(X)) < $(size(X,2)) = size(X,2). dropping zpos.")
    else
      info("includezpos == false. dropping zpos.")
    end
    Z = srproj(coefszero, posindic(counts), dirzero; intercept=true)
    X = [X_nocounts Z]
    includezpos = false
  end

  if !includem
    # drop last column with total counts m
    X = X[:,1:end-1]
  end

  X, X_nocounts, includezpos
end
