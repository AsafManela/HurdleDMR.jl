#############################################
# hurdle
#############################################

using Reexport, GLM.FPVector, GLM.FP, Compat, StatsBase
@reexport using GLM, StatsBase
using Lasso

type Hurdle <: RegressionModel
  mzero::RegressionModel  # model for zeros
  mpos::RegressionModel   # model for positive counts
  # pzero::Int              # number of covariates
  # ppos::Int               # number of covariates
  fittedzero::Bool        # whether the model for zeros was fitted
  fittedpos::Bool         # whether the model for positives was fitted
end

function StatsBase.fit{M<:RegressionModel,T<:FP,V<:FPVector}(::Type{Hurdle},::Type{M},
  X::AbstractMatrix{T}, y::V,
  dzero::UnivariateDistribution = Binomial(),
  dpos::UnivariateDistribution = PositivePoisson(),
  lzero::Link = canonicallink(dzero),
  lpos::Link = canonicallink(dpos);
  Xpos::@compat(Union{AbstractMatrix{T},Void}) = nothing,
  dofit::Bool = true,
  wts::V = ones(y),
  offsetzero::AbstractVector = similar(y, 0),
  offsetpos::AbstractVector = similar(y, 0),
  offset::AbstractVector = similar(y, 0),
  verbose::Bool = false,
  fitargs...)

  # find positive y entries
  ixpos = find(y)

  # build positive indicators vector
  Iy = zeros(y)
  Iy[ixpos] = 1

  # fit zero model to entire sample
  # @bp M == GammaLassoPath
  verbose && info("fitting zero model: $dzero, $lzero")
  mzero = fit(M, X, Iy, dzero, lzero; dofit=dofit, wts=wts, offset=offsetzero, verbose=verbose, fitargs...)
  fittedzero = true

  if length(offset) != 0
    if length(offsetpos) == 0
      offsetpos = offset
    end
    if length(offsetzero) == 0
      offsetzero = offset
    end
  end

  if length(offsetpos) == length(y)
    offpos = offsetpos[ixpos]
  else
    offpos = offsetpos
  end

  # Xpos optional argument allows to specify a data matrix only for positive counts
  if Xpos == nothing
    # use X for Xpos too
    Xpos = X[ixpos,:]
  elseif size(Xpos,1) == length(y)
    # Xpos has same dimensions as X, take only positive y ones
    Xpos = Xpos[ixpos,:]
  end

  # fit truncated counts model to positive subsample
  mpos=nothing
  fittedpos = false
  ypos = y[ixpos]
  if sum(ypos.>1) > 1
    try
      mpos = fit(M, Xpos, ypos, dpos, lpos; dofit=dofit, wts=wts[ixpos], offset=offpos, verbose=verbose, fitargs...)
      fittedpos = dofit
    catch e
      verbose && warn("failed to fit truncated counts model to positive subsample.")
      verbose && warn("possibly not enough variation in ypos:")
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
    verbose && warn("ypos has no elements larger than 1! the data is fully described by a probability model")
  end

  if !fittedpos
    # create blank model without fitting
    if M <: RegularizationPath
      mpos = fit(M, Xpos, ypos, dpos, lpos; dofit=false, λ=[0.0], wts=wts[ixpos], offset=offpos, verbose=verbose, fitargs...)
    else
      mpos = fit(M, Xpos, ypos, dpos, lpos; dofit=false, wts=wts[ixpos], offset=offpos, verbose=verbose, fitargs...)
    end
  end

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
                                      wts = ones(Float64,size(df,1)),
                                      offsetzero = Float64[],
                                      offsetpos = Float64[],
                                      offset = Float64[],
                                      verbose::Bool = false,
                                      fitargs...)

  mfzero = ModelFrame(f, df)
  mmzero = ModelMatrix(mfzero)
  y = model_response(mfzero)

  # find positive y entries
  ixpos = find(y.>0)

  # build positive indicators vector
  Iy = zeros(y)
  Iy[ixpos] = 1

  if length(offset) != 0
    if length(offsetpos) == 0
      offsetpos = offset
    end
    if length(offsetzero) == 0
      offsetzero = offset
    end
  end

  if length(offsetpos) == length(y)
    offpos = offsetpos[ixpos]
  else
    offpos = offsetpos
  end

  # fit zero model to entire sample
  # TODO: should be wrapped in DataFrameRegressionModel but can't figure out right now why it complains it is not defined
  # mzero = DataFrameRegressionModel(fit(GeneralizedLinearModel, mmzero.m, Iy, dzero, lzero; wts=wts, offset=offsetzero, fitargs...), mfzero, mmzer)
  mzero = fit(M, mmzero.m, Iy, dzero, lzero; wts=wts, offset=offsetzero, fitargs...)
  fittedzero = true

  mfpos = (f===fpos) ? mfzero : ModelFrame(fpos, df)
  mmpos = (f===fpos) ? mmzero : ModelMatrix(mfpos)
  mmpos.m = mmpos.m[ixpos,:]

  # fit truncated counts model to positive subsample
  mpos=nothing
  fittedpos = false
  try
    # TODO: should be wrapped in DataFrameRegressionModel but can't figure out right now why it complains it is not defined
    # mpos = DataFrameRegressionModel(fit(GeneralizedLinearModel, mmpos.m, y[ixpos], dpos, lpos; wts=wts[ixpos], offset=offpos, fitargs...), mfpos, mmpos)
    mpos = fit(M, mmpos.m, y[ixpos], dpos, lpos; wts=wts[ixpos], offset=offpos, verbose=verbose, fitargs...)#, mfpos, mmpos)
    fittedpos = true
  catch e
    if typeof(e) <: ErrorException && (contains(e.msg,"step-halving") || contains(e.msg,"failure to converge") || contains(e.msg,"failed to converge")) ||
      typeof(e) == Base.LinAlg.PosDefException || typeof(e) == DomainError
      verbose && warn("failed to fit truncated counts model to positive subsample.")
      verbose && warn("possibly not enough variation in ypos:")
      verbose && warn("countmap(y)=$(countmap(y))")
      verbose && warn("X'=$(X')")
      verbose && warn("y=$y)")
      mpos = fit(M, mmpos.m, y[ixpos], dpos, lpos; dofit=false, wts=wts[ixpos], offset=offpos, verbose=verbose, fitargs...)#, mfpos, mmpos)
      fittedpos = false
    else
      rethrow(e)
    end
  end

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
