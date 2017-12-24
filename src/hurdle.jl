#############################################
# hurdle
#############################################

using Reexport, GLM.FPVector, GLM.FP, Compat, StatsBase
@reexport using GLM, StatsBase
using Lasso

mutable struct Hurdle <: RegressionModel
  mzero::RegressionModel  # model for zeros
  mpos::RegressionModel   # model for positive counts
  fittedzero::Bool        # whether the model for zeros was fitted
  fittedpos::Bool         # whether the model for positives was fitted
end

function setoffsets{T}(y::AbstractVector{T}, ixpos::Vector{Int64}, offset::AbstractVector, offsetzero::AbstractVector, offsetpos::AbstractVector)
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

function setIy{T}(y::AbstractVector{T})
    # find positive y entries
    ixpos = find(y)

    # build positive indicators vector
    Iy = zeros(y)
    Iy[ixpos] = one(T)

    ixpos, Iy
end

function fitzero{M<:RegressionModel,T<:FP,V<:FPVector}(::Type{M},
  X::AbstractMatrix{T}, Iy::V,
  dzero::UnivariateDistribution,
  # dpos::UnivariateDistribution = PositivePoisson(),
  lzero::Link,
  # lpos::Link = canonicallink(dpos);
  # Xpos::@compat(Union{AbstractMatrix{T},Void}) = nothing,
  dofit::Bool,
  wts::V,
  offsetzero::AbstractVector,
  # offsetpos::AbstractVector = similar(y, 0),
  # offset::AbstractVector = similar(y, 0),
  verbose::Bool,
  fitargs...)

  # fit zero model to entire sample
  mzero = nothing
  fittedzero = false
  if var(Iy) > zero(T)
    try
      verbose && info("fitting zero model: $dzero, $lzero")
      mzero = fit(M, X, Iy, dzero, lzero; dofit=dofit, wts=wts, offset=offsetzero, verbose=verbose, fitargs...)
      fittedzero = dofit
    catch e
      verbose && warn("failed to fit zero counts model, possibly not enough variation in I(y):")
      verbose && warn("countmap(Iy)=$(countmap(Iy))")
      if typeof(e) <: ErrorException && (contains(e.msg,"step-halving") || contains(e.msg,"failure to converge") || contains(e.msg,"failed to converge")) ||
          typeof(e) == Base.LinAlg.PosDefException || typeof(e) == DomainError
        fittedzero = false
      else
        verbose && warn("countmap(Iy)=$(countmap(Iy))")
        verbose && warn("X'=$(X')")
        verbose && warn("Iy=$Iy)")
        rethrow(e)
      end
    end
  else
    if verbose
      if all(iszero,Iy)
        warn("I(y) is all zeros. There is nothing to explain.")
      else
        warn("I(y) is all ones. Data may be fully described by a poisson model.")
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

function fitpos{M<:RegressionModel,T<:FP,V<:FPVector}(::Type{M},
  Xpos::AbstractMatrix{T}, ypos::V,
  dpos::UnivariateDistribution,
  lpos::Link,
  dofit::Bool,
  wtspos::V,
  offsetpos::AbstractVector,
  verbose::Bool,
  fitargs...)

  # fit truncated counts model to positive subsample
  mpos=nothing
  fittedpos = false
  if sum(ypos.>1) > 1
    try
      mpos = fit(M, Xpos, ypos, dpos, lpos; dofit=dofit, wts=wtspos, offset=offsetpos, verbose=verbose, fitargs...)
      fittedpos = dofit
    catch e
      verbose && warn("failed to fit truncated counts model to positive subsample, possibly not enough variation in ypos:")
      verbose && warn("countmap(y)=$(countmap(y))")
      if typeof(e) <: ErrorException && (contains(e.msg,"step-halving") || contains(e.msg,"failure to converge") || contains(e.msg,"failed to converge")) ||
          typeof(e) == Base.LinAlg.PosDefException || typeof(e) == DomainError
        fittedpos = false
      else
        verbose && warn("countmap(y)=$(countmap(y))")
        verbose && warn("X'=$(X')")
        verbose && warn("y=$y)")
        rethrow(e)
      end
    end
  else
    verbose && warn("ypos has no elements larger than 1! Data may be fully described by a probability model.")
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

function StatsBase.fit{M<:RegressionModel,T<:FP,V<:FPVector}(::Type{Hurdle},::Type{M},
  X::AbstractMatrix{T}, y::V,
  dzero::UnivariateDistribution = Binomial(),
  dpos::UnivariateDistribution = PositivePoisson(),
  lzero::Link = canonicallink(dzero),
  lpos::Link = canonicallink(dpos);
  Xpos::Union{AbstractMatrix{T},Void} = nothing,
  dofit::Bool = true,
  wts::V = ones(y),
  offsetzero::AbstractVector = similar(y, 0),
  offsetpos::AbstractVector = similar(y, 0),
  offset::AbstractVector = similar(y, 0),
  verbose::Bool = false,
  fitargs...)

  ixpos, Iy = setIy(y)

  offsetzero, offsetpos = setoffsets(y, ixpos, offset, offsetzero, offsetpos)

  mzero, fittedzero = fitzero(M, X, Iy, dzero, lzero, dofit, wts, offsetzero, verbose, fitargs...)

  # Xpos optional argument allows to specify a data matrix only for positive counts
  if Xpos == nothing
    # use X for Xpos too
    Xpos = X[ixpos,:]
  elseif size(Xpos,1) == length(y)
    # Xpos has same dimensions as X, take only positive y ones
    Xpos = Xpos[ixpos,:]
  end

  mpos, fittedpos = fitpos(M, Xpos, y[ixpos], dpos, lpos, dofit, wts[ixpos], offsetpos, verbose, fitargs...)

  Hurdle(mzero,mpos,fittedzero,fittedpos)
end

function StatsBase.fit{M<:RegressionModel}(::Type{Hurdle},::Type{M},
                                      f::Formula,
                                      df::AbstractDataFrame,
                                      dzero::UnivariateDistribution = Binomial(),
                                      dpos::UnivariateDistribution = PositivePoisson(),
                                      lzero::Link = canonicallink(dzero),
                                      lpos::Link = canonicallink(dpos);
                                      fpos::Formula = f,
                                      dofit::Bool = true,
                                      wts = ones(Float64,size(df,1)),
                                      offsetzero = Float64[],
                                      offsetpos = Float64[],
                                      offset = Float64[],
                                      verbose::Bool = false,
                                      fitargs...)

  mfzero = ModelFrame(f, df)
  mmzero = ModelMatrix(mfzero)
  y = model_response(mfzero)

  ixpos, Iy = setIy(y)

  offsetzero, offsetpos = setoffsets(y, ixpos, offset, offsetzero, offsetpos)

  # fit zero model to entire sample
  # TODO: should be wrapped in DataFrameRegressionModel but can't figure out right now why it complains it is not defined
  # mzero = DataFrameRegressionModel(fit(GeneralizedLinearModel, mmzero.m, Iy, dzero, lzero; wts=wts, offset=offsetzero, fitargs...), mfzero, mmzer)
  mzero, fittedzero = fitzero(M, mmzero.m, Iy, dzero, lzero, dofit, wts, offsetzero, verbose, fitargs...)

  mfpos = (f===fpos) ? mfzero : ModelFrame(fpos, df)
  mmpos = (f===fpos) ? mmzero : ModelMatrix(mfpos)
  mmpos.m = mmpos.m[ixpos,:]

  mpos, fittedpos = fitpos(M, mmpos.m, y[ixpos], dpos, lpos, dofit, wts[ixpos], offsetpos, verbose, fitargs...)

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

## Prediction function for GLMs
function StatsBase.predict{T<:AbstractFloat}(hurdle::Hurdle, X::AbstractMatrix{T};
  Xpos::AbstractMatrix{T} = X,
  offsetzero::AbstractVector = Array{T}(0),
  offsetpos::AbstractVector = Array{T}(0),
  offset::AbstractVector=Array{T}(0),
  kwargs...)

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
