# InclusionRepetition model is a two-part model like the hurdle but instead of a
# PositivePoisson multiplied by the Bernulli, it uses a regular Poisson for
# repetition (counts in excess of 1), conditional on inclusion.

"Selection returned object"
mutable struct InclusionRepetition{Z<:RegressionModel,P<:RegressionModel} <: TwoPartModel{Z,P}
  mzero::Z                # model for zeros
  mpos::P                 # model for positive counts
  fittedzero::Bool        # whether the model for zeros was fitted
  fittedpos::Bool         # whether the model for positives was fitted
end

# predicted counts vector given expected inclusion (μzero) and repetition (μpos)
μtpm(m::InclusionRepetition, μzero::V, μpos::V) where {T, V<:AbstractArray{T}} = μzero .* (one(T) .+ μpos)

function excessy!(ypos::V, ::Type{InclusionRepetition}) where {T, V<:AbstractVector{T}}
  @simd for i = 1:length(ypos)
    @inbounds ypos[i] -= one(T)
  end
  ypos
end

minypos(::Type{InclusionRepetition}) = 0.0

"""
    fit(InclusionRepetition,M,X,y; Xpos=Xpos, <keyword arguments>)

Fit an Inclusion-Repetition model of count vector y on X with potentially another
covariates matrix Xpos used to model positive counts (repetitions).

# Example with GLM:
```julia
  m = fit(InclusionRepetition,GeneralizedLinearModel,X,y; Xpos=Xpos)
  yhat = predict(m, X; Xpos=Xpos)
```

# Example with Lasso regularization:
```julia
  m = fit(InclusionRepetition,GammaLassoPath,X,y; Xpos=Xpos)
  yhat = predict(m, X; Xpos=Xpos, select=MinAICc())
```

# Arguments
- `M::RegressionModel`
- `counts` n-by-d matrix of counts (usually sparse)
- `dzero::UnivariateDistribution = Binomial()` distribution for zeros (inclusion) model
- `dpos::UnivariateDistribution = Poisson()` distribution for positives (repetition) model
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
function StatsBase.fit(::Type{InclusionRepetition},::Type{M},
  X::AbstractMatrix{T}, y::V,
  dzero::UnivariateDistribution = Binomial(),
  dpos::UnivariateDistribution = Poisson(),
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
    @inbounds Xpos = X[ixpos,:]
  elseif size(Xpos,1) == length(y)
    # Xpos has same dimensions as X, take only positive y ones
    @inbounds Xpos = Xpos[ixpos,:]
  end

  @inbounds ypos = y[ixpos]
  @inbounds wtspos = wts[ixpos]

  mpos, fittedpos = fitpos(InclusionRepetition, M, Xpos, ypos, dpos, lpos, dofit, wtspos, offsetpos, verbose, showwarnings, fitargs...)

  InclusionRepetition(mzero,mpos,fittedzero,fittedpos)
end

"""
    fit(InclusionRepetition,M,f,df; fpos=Xpos, <keyword arguments>)

Takes dataframe and two formulas, one for each model part. Otherwise same arguments
as [`fit(::InclusionRepetition)`](@ref)

# Example
```julia
  fit(InclusionRepetition,GeneralizedLinearModel,@formula(y ~ x1*x2), df; fpos=@formula(y ~ x1*x2+x3))
```
"""
function StatsBase.fit(::Type{InclusionRepetition},::Type{M},
                      f::FormulaTerm,
                      data,
                      dzero::UnivariateDistribution = Binomial(),
                      dpos::UnivariateDistribution = Poisson(),
                      lzero::Link = canonicallink(dzero),
                      lpos::Link = canonicallink(dpos);
                      fpos::FormulaTerm = f,
                      dofit::Bool = true,
                      wts = fill(1.0,size(data,1)),
                      offsetzero = Float64[],
                      offsetpos = Float64[],
                      offset = Float64[],
                      verbose::Bool = false,
                      showwarnings::Bool = false,
                      contrasts::Dict{Symbol,<:Any} = Dict{Symbol,Any}(),
                      fitargs...) where {M<:RegressionModel}

  Tables.istable(data) || throw(ArgumentError("expected data in a Table, got $(typeof(data))"))

  cols = columntable(data)

  mfzero = ModelFrame(f, cols, model=M, contrasts=contrasts)
  mmzero = ModelMatrix(mfzero)
  y = response(mfzero)

  ixpos, Iy = getIy(y)

  offsetzero, offsetpos = setoffsets(y, ixpos, offset, offsetzero, offsetpos)

  # fit zero model to entire sample
  # TODO: should be wrapped in TableRegressionModel but can't figure out right now why it complains it is not defined
  # mzero = TableRegressionModel(fit(M, mmzero.m, Iy, dzero, lzero; wts=wts, offset=offsetzero, fitargs...), mfzero, mmzero)
  mzero, fittedzero = fitzero(M, mmzero.m, Iy, dzero, lzero, dofit, wts, offsetzero, verbose, showwarnings, fitargs...)

  mfpos = (f===fpos) ? mfzero : ModelFrame(fpos, cols, model=M, contrasts=contrasts)
  mmpos = (f===fpos) ? mmzero : ModelMatrix(mfpos)
  mmpos.m = mmpos.m[ixpos,:]

  mpos, fittedpos = fitpos(InclusionRepetition, M, mmpos.m, y[ixpos], dpos, lpos, dofit, wts[ixpos], offsetpos, verbose, showwarnings, fitargs...)

  InclusionRepetition(mzero,mpos,fittedzero,fittedpos)
end
