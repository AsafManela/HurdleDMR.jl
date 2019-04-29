#############################################
# hurdle
#############################################

abstract type TwoPartModel <: RegressionModel end

"Hurdle returned object"
mutable struct Hurdle{Z<:RegressionModel,P<:RegressionModel} <: TwoPartModel
  mzero::Z                # model for zeros
  mpos::P                 # model for positive counts
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
  yhat = predict(m, X; Xpos=Xpos, select=MinAICc())
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

function Base.show(io::IO, tpm::TwoPartModel)
  name = typeof(tpm).name
  println(io, "$name regression\n")

  if typeof(tpm.mpos) <: RegularizationPath
    println(io, "Positive part regularization path ($(distfun(tpm.mpos)) with $(linkfun(tpm.mpos)) link):")
    println(io, tpm.fittedpos ? tpm.mpos : "Not Fitted")

    println(io, "Zero part regularization path ($(distfun(tpm.mzero)) with $(linkfun(tpm.mzero)) link):")
    println(io, tpm.fittedzero ? tpm.mzero : "Not Fitted")
  else
    println(io, "Positive part coefficients ($(Distribution(tpm.mpos)) with $(Link(tpm.mpos)) link):")
    println(io, tpm.fittedpos ? tpm.mpos : "Not Fitted")

    println(io, "Zero part coefficients ($(Distribution(tpm.mzero)) with $(Link(tpm.mzero)) link):")
    println(io, tpm.fittedzero ? tpm.mzero : "Not Fitted")
  end
end

"""
    coef(m::Hurdle; select=MinAICc())

Returns a selected segment of the coefficient matrices of the fitted the TwoPartModel.

# Example:
```julia
  m = fit(Hurdle,GeneralizedLinearModel,X,y; Xpos=Xpos)
  coefspos, coefszero = coef(m)
```

# Keywords
- `kwargs...` are passed along to two coef() calls on the two model parts.
"""
function StatsBase.coef(tpm::TwoPartModel; kwargs...)
  czero = coef(tpm.mzero; kwargs...)
  cpos = coef(tpm.mpos; kwargs...)
  cpos,czero
end

# the following coef definitions override those in Lasso for Hurdles
function StatsBase.coef(tpm::TwoPartModel, select::SegSelect; kwargs...)
  czero = coef(tpm.mzero, select; kwargs...)
  cpos = coef(tpm.mpos, select; kwargs...)
  cpos,czero
end

function StatsBase.coef(tpm::TwoPartModel, select::S; kwargs...) where {S<:CVSegSelect}
  error("""
    Specifying an instance of a `CVSegSelect` is not supported because there is
    more than one path and its generator has a fixed number of observation indices.
    Instead, consider passing a `MinCVKfold{$S}(k)`.
    """)
end

"Selects the RegularizationPath segment according to `CVSegSelect` with `k`-fold cross-validation"
struct MinCVKfold{S<:CVSegSelect} <: CVSegSelect
  k::Int  # number of CV folds
end

"Selects the RegularizationPath segment coefficients according to `S` with `k`-fold cross-validation"
function StatsBase.coef(path::RegularizationPath, select::MinCVKfold{S};
  kwargs...) where {S<:CVSegSelect}

  selector = S(path, select.k)
  coef(path, selector; kwargs...)
end

"Selects the RegularizationPath segment coefficients according to `S` with `k`-fold cross-validation"
function StatsBase.coef(tpm::TwoPartModel, select::MinCVKfold{S};
  kwargs...) where {S<:CVSegSelect}

  selectzero = S(tpm.mzero, select.k)
  selectpos = S(tpm.mpos, select.k)
  czero = coef(tpm.mzero, selectzero; kwargs...)
  cpos = coef(tpm.mpos, selectpos; kwargs...)
  cpos,czero
end

# predicted counts vector given expected inclusion (μzero) and repetition (μpos)
μtpm(m::Hurdle, μzero, μpos) = μzero .* μpos

"""
    predict(m,X; Xpos=Xpos, <keyword arguments>)

Predict using a fitted TwoPartModel given new X (and potentially Xpos).

# Example with GLM:
```julia
  m = fit(Hurdle,GeneralizedLinearModel,X,y; Xpos=Xpos)
  yhat = predict(m, X; Xpos=Xpos)
```

# Example with Lasso regularization:
```julia
  m = fit(Hurdle,GammaLassoPath,X,y; Xpos=Xpos)
  yhat = predict(m, X; Xpos=Xpos, select=MinAICc())
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
function StatsBase.predict(tpm::TwoPartModel, X::AbstractMatrix{T};
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

  μzero = predict(tpm.mzero, X; offset=offsetzero, kwargs...)
  μpos = predict(tpm.mpos, Xpos; offset=offsetpos, kwargs...)

  @assert size(μzero) == size(μpos) "Predicted values from zero and positives models have different dimensions, $(size(μzero)) != $(size(μpos))\nCan result from fitting a RegularizationPath with autoλ and select=AllSeg() is specified."

  μtpm(tpm, μzero, μpos)
end
