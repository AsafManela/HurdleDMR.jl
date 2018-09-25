#############################################
# hurdle
#############################################

"Hurdle returned object"
mutable struct Hurdle <: RegressionModel
  mzero::RegressionModel  # model for zeros
  mpos::RegressionModel   # model for positive counts
  fittedzero::Bool        # whether the model for zeros was fitted
  fittedpos::Bool         # whether the model for positives was fitted
end

"Returns an (offsetzero, offsetpos) tuple of offset vector"
function setoffsets(y::AbstractVector, ixpos::Vector{Int}, offset::AbstractVector, offsetzero::AbstractVector, offsetpos::AbstractVector)
    # set offsets
    if length(offset) != 0
      if length(offsetpos) == 0
        offsetpos = offset
      end
      if length(offsetzero) == 0
        offsetzero = offset
      end
    end

    if length(offsetpos) == length(y)
      offsetpos = offsetpos[ixpos]
    end

    offsetzero, offsetpos
end

"Returns positives indicators for y"
function getIy(y::AbstractVector{T}) where {T}
    # find positive y entries
    ixpos = findall(x->x!=zero(T), y)

    # build positive indicators vector
    Iy = zero(y)
    Iy[ixpos] .= one(T)

    ixpos, Iy
end

"Fits the model for zeros Iy ~ X"
function fitzero(::Type{M},
  X::AbstractMatrix{T}, Iy::V,
  dzero::UnivariateDistribution,
  lzero::Link,
  dofit::Bool,
  wts::V,
  offsetzero::AbstractVector,
  verbose::Bool,
  showwarnings::Bool,
  fitargs...) where {M<:RegressionModel,T<:FP,V<:FPVector}

  # fit zero model to entire sample
  mzero = nothing
  fittedzero = false
  if var(Iy) > zero(T)
    try
      mzero = fit(M, X, Iy, dzero, lzero; dofit=dofit, wts=wts, offset=offsetzero, verbose=verbose, fitargs...)
      fittedzero = dofit
    catch e
      showwarnings && @warn("failed to fit zero counts model, possibly not enough variation in I(y). countmap(Iy)=$(countmap(Iy))")
      if typeof(e) <: ErrorException && (occursin("step-halving", e.msg) || occursin("failure to converge", e.msg) || occursin("failed to converge", e.msg)) ||
          typeof(e) == PosDefException || typeof(e) == DomainError
        fittedzero = false
      else
        showwarnings && @warn("X'=$(X')")
        showwarnings && @warn("Iy=$Iy)")
        rethrow(e)
      end
    end
  else
    if verbose
      if all(iszero,Iy)
        showwarnings && @warn("I(y) is all zeros. There is nothing to explain.")
      else
        showwarnings && @warn("I(y) is all ones. Data may be fully described by a poisson model.")
      end
    end
  end

  if !fittedzero
    # create blank zeros model without fitting
    if M <: RegularizationPath
      mzero = fit(M, X, Iy, dzero, lzero; dofit=false, λ=[0.0], wts=wts, offset=offsetzero, verbose=verbose, fitargs...)
    else
      mzero = fit(M, X, Iy, dzero, lzero; dofit=false, wts=wts, offset=offsetzero, verbose=verbose, fitargs...)
    end
  end

  mzero, fittedzero
end

"Fits the model for positives ypos ~ Xpos"
function fitpos(::Type{M},
  Xpos::AbstractMatrix{T}, ypos::V,
  dpos::UnivariateDistribution,
  lpos::Link,
  dofit::Bool,
  wtspos::V,
  offsetpos::AbstractVector,
  verbose::Bool,
  showwarnings::Bool,
  fitargs...) where {M<:RegressionModel,T<:FP,V<:FPVector}

  # fit truncated counts model to positive subsample
  mpos=nothing
  fittedpos = false
  if any(x->x>1, ypos)
    try
      mpos = fit(M, Xpos, ypos, dpos, lpos; dofit=dofit, wts=wtspos, offset=offsetpos, verbose=verbose, fitargs...)
      fittedpos = dofit
    catch e
      showwarnings && @warn("failed to fit truncated counts model to positive subsample, possibly not enough variation in ypos. countmap(ypos)=$(sort(countmap(ypos)))")
      if typeof(e) <: ErrorException && (occursin("step-halving", e.msg) || occursin("failure to converge", e.msg) || occursin("failed to converge", e.msg)) ||
          typeof(e) == PosDefException || typeof(e) == DomainError
        fittedpos = false
      else
        showwarnings && @warn("Xpos'=$(Xpos')")
        showwarnings && @warn("ypos=$ypos)")
        rethrow(e)
      end
    end
  else
    if length(ypos) == 0
      error("y is all zeros! There is nothing to explain.")
    else
      showwarnings && @warn("ypos has no elements larger than 1! Data may be fully described by a probability model.")
    end
  end

  if !fittedpos
    # create blank positives model without fitting
    if M <: RegularizationPath
      mpos = fit(M, Xpos, ypos, dpos, lpos; dofit=false, λ=[0.0], wts=wtspos, offset=offsetpos, verbose=verbose, fitargs...)
    else
      mpos = fit(M, Xpos, ypos, dpos, lpos; dofit=false, wts=wtspos, offset=offsetpos, verbose=verbose, fitargs...)
    end
  end

  mpos, fittedpos
end

"""
    fit(Hurdle,M,X,y; Xpos=Xpos, <keyword arguments>)

Fit a Hurdle (Mullahy, 1986) of count vector y on X with potentially another
covariates matrix Xpos used to model positive counts.

# Example with GLM:
```julia
  m = fit(Hurdle,GeneralizedLinearModel,X,y; Xpos=Xpos)
  yhat = predict(m, X; Xpos=Xpos)
```

# Example with Lasso regularization:
```julia
  m = fit(Hurdle,GammaLassoPath,X,y; Xpos=Xpos)
  yhat = predict(m, X; Xpos=Xpos, select=:AICc)
```

# Arguments
- `M::RegressionModel`
- `counts` n-by-d matrix of counts (usually sparse)
- `dzero::UnivariateDistribution = Binomial()` distribution for zeros model
- `dpos::UnivariateDistribution = PositivePoisson()` distribution for positives model
- `lzero::Link=canonicallink(dzero)` link function for zeros model
- `lpos::Link=canonicallink(dpos)` link function for positives model

# Keywords
- `Xpos::Union{AbstractMatrix{T},Nothing} = nothing` covariates matrix for positives
  model or nothing to use X for both parts
- `dofit::Bool = true` fit the model or just construct its shell
- `wts::V = ones(y)` observation weights
- `offsetzero::AbstractVector = similar(y, 0)` offsets for zeros model
- `offsetpos::AbstractVector = similar(y, 0)` offsets for positives model
- `offset::AbstractVector = similar(y, 0)` offsets for both model parts
- `verbose::Bool=true`
- `showwarnings::Bool=false`
- `fitargs...` additional keyword arguments passed along to fit(M,...)
"""
function StatsBase.fit(::Type{Hurdle},::Type{M},
  X::AbstractMatrix{T}, y::V,
  dzero::UnivariateDistribution = Binomial(),
  dpos::UnivariateDistribution = PositivePoisson(),
  lzero::Link = canonicallink(dzero),
  lpos::Link = canonicallink(dpos);
  Xpos::Union{AbstractMatrix{T},Nothing} = nothing,
  dofit::Bool = true,
  wts::V = fill(one(eltype(y)),size(y)),
  offsetzero::AbstractVector = similar(y, 0),
  offsetpos::AbstractVector = similar(y, 0),
  offset::AbstractVector = similar(y, 0),
  verbose::Bool = false,
  showwarnings::Bool = false,
  fitargs...) where {M<:RegressionModel,T<:FP,V<:FPVector}

  ixpos, Iy = getIy(y)

  offsetzero, offsetpos = setoffsets(y, ixpos, offset, offsetzero, offsetpos)

  mzero, fittedzero = fitzero(M, X, Iy, dzero, lzero, dofit, wts, offsetzero, verbose, showwarnings, fitargs...)

  # Xpos optional argument allows to specify a data matrix only for positive counts
  if Xpos == nothing
    # use X for Xpos too
    Xpos = X[ixpos,:]
  elseif size(Xpos,1) == length(y)
    # Xpos has same dimensions as X, take only positive y ones
    Xpos = Xpos[ixpos,:]
  end

  mpos, fittedpos = fitpos(M, Xpos, y[ixpos], dpos, lpos, dofit, wts[ixpos], offsetpos, verbose, showwarnings, fitargs...)

  Hurdle(mzero,mpos,fittedzero,fittedpos)
end

"""
    fit(Hurdle,M,f,df; fpos=Xpos, <keyword arguments>)

Takes dataframe and two formulas, one for each model part. Otherwise same arguments
as [`fit(::Hurdle)`](@ref)

# Example
```julia
  fit(Hurdle,GeneralizedLinearModel,@formula(y ~ x1*x2), df; fpos=@formula(y ~ x1*x2+x3))
```
"""
function StatsBase.fit(::Type{Hurdle},::Type{M},
                      f::Formula,
                      df::AbstractDataFrame,
                      dzero::UnivariateDistribution = Binomial(),
                      dpos::UnivariateDistribution = PositivePoisson(),
                      lzero::Link = canonicallink(dzero),
                      lpos::Link = canonicallink(dpos);
                      fpos::Formula = f,
                      dofit::Bool = true,
                      wts = fill(1.0,size(df,1)),
                      offsetzero = Float64[],
                      offsetpos = Float64[],
                      offset = Float64[],
                      verbose::Bool = false,
                      showwarnings::Bool = false,
                      fitargs...) where {M<:RegressionModel}

  mfzero = ModelFrame(f, df)
  mmzero = ModelMatrix(mfzero)
  y = model_response(mfzero)

  ixpos, Iy = getIy(y)

  offsetzero, offsetpos = setoffsets(y, ixpos, offset, offsetzero, offsetpos)

  # fit zero model to entire sample
  # TODO: should be wrapped in DataFrameRegressionModel but can't figure out right now why it complains it is not defined
  # mzero = DataFrameRegressionModel(fit(GeneralizedLinearModel, mmzero.m, Iy, dzero, lzero; wts=wts, offset=offsetzero, fitargs...), mfzero, mmzer)
  mzero, fittedzero = fitzero(M, mmzero.m, Iy, dzero, lzero, dofit, wts, offsetzero, verbose, showwarnings, fitargs...)

  mfpos = (f===fpos) ? mfzero : ModelFrame(fpos, df)
  mmpos = (f===fpos) ? mmzero : ModelMatrix(mfpos)
  mmpos.m = mmpos.m[ixpos,:]

  mpos, fittedpos = fitpos(M, mmpos.m, y[ixpos], dpos, lpos, dofit, wts[ixpos], offsetpos, verbose, showwarnings, fitargs...)

  Hurdle(mzero,mpos,fittedzero,fittedpos)
end

function Base.show(io::IO, hurdle::Hurdle)
  println(io, "Hurdle regression\n")

  if typeof(hurdle.mpos) <: RegularizationPath
    println(io, "Count model regularization path ($(distfun(hurdle.mpos)) with $(linkfun(hurdle.mpos)) link):")
    println(io, hurdle.fittedpos ? hurdle.mpos : "Not Fitted")

    println(io, "Zero hurdle regularization path ($(distfun(hurdle.mzero)) with $(linkfun(hurdle.mzero)) link):")
    println(io, hurdle.fittedzero ? hurdle.mzero : "Not Fitted")
  else
    println(io, "Count model coefficients ($(Distribution(hurdle.mpos)) with $(Link(hurdle.mpos)) link):")
    println(io, hurdle.fittedpos ? hurdle.mpos : "Not Fitted")

    println(io, "Zero hurdle coefficients ($(Distribution(hurdle.mzero)) with $(Link(hurdle.mzero)) link):")
    println(io, hurdle.fittedzero ? hurdle.mzero : "Not Fitted")
  end
end

# function copypad!{T}(destA::AbstractMatrix{T},srcA::AbstractMatrix{T},padvalue=zero(T))
#   for i=eachindex(srcA)
#     destA[i] = srcA[i]
#   end
#   destA
# end

"""
    coef(m::Hurdle; <keyword arguments>)

Returns the AICc optimal coefficient matrices fitted the Hurdle.

# Example:
```julia
  m = fit(Hurdle,GeneralizedLinearModel,X,y; Xpos=Xpos)
  coefspos, coefszero = coef(m)
```

# Keywords
- `kwargs...` are passed along to two coef() calls on the two model parts.
"""
function StatsBase.coef(hurdle::Hurdle; kwargs...)
  czero = coef(hurdle.mzero; kwargs...)
  cpos = coef(hurdle.mpos; kwargs...)
  # # dimensions of this differ if regularization paths terminates early at
  # # different segments
  # if size(czero,2) > size(cpos,2)
  #   cpos = copypad!(similar(czero),cpos)
  # elseif size(czero,2) < size(cpos,2)
  #   czero = copypad!(similar(cpos),czero)
  # end
  cpos,czero
end

"""
    predict(m,X; Xpos=Xpos, <keyword arguments>)

Predict using a fitted Hurdle given new X (and potentially Xpos).

# Example with GLM:
```julia
  m = fit(Hurdle,GeneralizedLinearModel,X,y; Xpos=Xpos)
  yhat = predict(m, X; Xpos=Xpos)
```

# Example with Lasso regularization:
```julia
  m = fit(Hurdle,GammaLassoPath,X,y; Xpos=Xpos)
  yhat = predict(m, X; Xpos=Xpos, select=:AICc)
```

# Arguments
- `m::Hurdle` fitted Hurdle model
- `X` n-by-p matrix of covariates of same dimensions used to fit m.

# Keywords
- `Xpos::Union{AbstractMatrix{T},Nothing} = nothing` covariates matrix for positives
  model or nothing to use X for both parts
- `kwargs...` additional keyword arguments passed along to predict() for each
  of the two model parts.
"""
function StatsBase.predict(hurdle::Hurdle, X::AbstractMatrix{T};
  Xpos::AbstractMatrix{T} = X,
  offsetzero::AbstractVector = Array{T}(undef, 0),
  offsetpos::AbstractVector = Array{T}(undef, 0),
  offset::AbstractVector=Array{T}(undef, 0),
  kwargs...) where {T<:AbstractFloat}

  # set offsets
  if length(offset) != 0
    if length(offsetpos) == 0
      offsetpos = offset
    end
    if length(offsetzero) == 0
      offsetzero = offset
    end
  end

  muzero = predict(hurdle.mzero, X; offset=offsetzero, kwargs...)
  mupos = predict(hurdle.mpos, Xpos; offset=offsetpos, kwargs...)

  @assert size(muzero) == size(mupos) "Predicted values from zero and positives models have different dimensions, $(size(muzero)) != $(size(mupos))\nCan result from fitting a RegularizationPath with autoλ and select=:all is specified."

  muzero .* mupos
end
