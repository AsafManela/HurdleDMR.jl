"""
Counts inverse regression (CIR) model supports both multinomial and hurdle inverse regressions
and holds both the inverse and forward regression model estimates
"""
struct CIR{BM<:DCR,FM<:RegressionModel} <: RegressionModel
  # covars::AbstractMatrix     # n×p covariates matrix
  # counts::AbstractMatrix     # n×d counts (document-term) matrix
  projdir::Int               # projection direction is the index of covariate representing the response
  inz::Vector{Int}           # indices of Z=srproj(counts) included in srprojX after dropping colinear ones on call to srprojX!
  bwdm::BM                   # inverse regression model: counts ~ X
  fwdm::FM                   # forward regression model: y ~ srproj(counts) + X
  fwdmnocounts::FM           # forward regression model w/o counts: y ~ X

  CIR{BM,FM}(projdir::Int, inz::Vector{Int}, bwdm::BM, fwdm::FM) where {BM<:DCR,FM<:RegressionModel} =
    new(projdir, inz, bwdm, fwdm)
  CIR{BM,FM}(projdir::Int, inz::Vector{Int}, bwdm::BM, fwdm::FM, fwdmnocounts::FM) where {BM<:DCR,FM<:RegressionModel} =
    new(projdir, inz, bwdm, fwdm, fwdmnocounts)
end

"""
    fit(::CIR{BM,FM},covars,counts,projdir[,fmargs...]; <keyword arguments>)

Fit a Counts Inverse Regression (CIR) of `covars[:,projdir] ~ counts + covars[:,~projdir]`.

CIR involves three steps:
  1. Fit a backward regression model BM<:DCR: counts ~ covars
  2. Calculate an sufficient reduction projection in direction projdir
  3. Fit a forward regression model FM<:RegressionModel:
    covars[:,projdir] ~ srproj(counts) + covars[:,~projdir]

# Example:
```julia
  m = fit(CIR{DMR,LinearModel}, covars, counts, 1; nocounts=true)
  yhat = predict(m, covars, counts)
  yhatnc = predict(m, covars, counts; nocounts=true)
```

# Arguments
- `covars` n-by-p matrix of covariates
- `counts` n-by-d matrix of counts (usually sparse)
- `projdir` index of covars column used as dependent variable in forward model
- `fmargs...` optional arguments passed along to the forward regression model

# Keywords
- `nocounts::Bool=false` whether to also fit a benchmark model without counts
- `bmkwargs...` keyword arguments passed along to the backward regression model
"""
function StatsBase.fit(::Type{C},covars::AbstractMatrix{T},counts::AbstractMatrix{V},projdir::Int, fmargs...;
  nocounts=false, bmkwargs...) where {T<:AbstractFloat,V,BM<:DCR,FM<:RegressionModel,C<:CIR{BM,FM}}

  # run inverse regression
  bwdm = fit(BM,covars,counts; bmkwargs...)

  # target variable
  y = covars[:,projdir]

  # calculate srproj design matrices for regressions
  X, X_nocounts, inz = srprojX(bwdm,counts,covars,projdir)

  # forward regression model with counts
  fwdm = fit(FM,X,y,fmargs...)

  if nocounts
    # forward model w/o counts
    fwdmnocounts = fit(FM,X_nocounts,y,fmargs...)

    # wrap in struct
    CIR{BM,FM}(projdir, inz, bwdm, fwdm, fwdmnocounts)
  else
    # wrap in struct
    CIR{BM,FM}(projdir, inz, bwdm, fwdm)
  end
end

"""
    fit(CIR{DMR,FM},m,df,counts,projdir[,fmargs...]; <keyword arguments>)

Version of fit(CIR{DMR,FM}...) that takes a @model() and a dataframe instead of a covars
matrix, and a projdir::Symbol specifies the dependent variable. See also fit(CIR...).

# Example:
```julia
  m = fit(CIR{DMR,LinearModel}, @model(c~x1+x2), df, counts, :x1; nocounts=true)
```
where `c~` is the model for counts.
`x1` (`projdir`) is the variable to predict.
We can then predict with a dataframe as well
```julia
  yhat = predict(m, df, counts)
  yhatnc = predict(m, df, counts; nocounts=true)
```
"""
function StatsBase.fit(::Type{C}, m::Model, df::AbstractDataFrame, counts::AbstractMatrix, sprojdir::Symbol, fmargs...;
  contrasts::Dict = Dict(), kwargs...) where {BM<:DMR,FM<:RegressionModel,C<:CIR{BM,FM}}
  # parse and merge rhs terms
  trms = getrhsterms(m, :c)

  # create model matrix
  mf, mm, counts = createmodelmatrix(trms, df, counts, contrasts)

  # resolve projdir
  projdir = ixprojdir(trms, sprojdir, mm)

  # fit and wrap in DataFrameRegressionModel
  StatsModels.DataFrameRegressionModel(fit(C, mm.m, counts, projdir, fmargs...; kwargs...), mf, mm)
end

"""
    fit(CIR{HDMR,FM},m,df,counts,projdir[,fmargs...]; <keyword arguments>)

Version of fit(CIR{HDMR,FM}...) that takes a @model() and a dataframe instead of a covars
matrix, and a projdir::Symbol specifies the dependent variable. See also fit(CIR...).

# Example:
```julia
  m = fit(CIR{HDMR,LinearModel}, @model(h~x1+x2, c~x1), df, counts, :x1; nocounts=true)
```
where `h~` is the model for zeros, `c~` is the model for positives.
`x1` (`projdir`) is the variable to predict.
We can then predict with a dataframe as well
```julia
  yhat = predict(m, df, counts)
  yhatnc = predict(m, df, counts; nocounts=true)
```
"""
function StatsBase.fit(::Type{C}, m::Model, df::AbstractDataFrame, counts::AbstractMatrix, sprojdir::Symbol, fmargs...;
  contrasts::Dict = Dict(), kwargs...) where {BM<:HDMR,FM<:RegressionModel,C<:CIR{BM,FM}}
  # parse and merge rhs terms
  trmszero = getrhsterms(m, :h)
  trmspos = getrhsterms(m, :c)
  trms, inzero, inpos = mergerhsterms(trmszero,trmspos)

  # create model matrix
  mf, mm, counts = createmodelmatrix(trms, df, counts, contrasts)

  # inzero and inpos may be different in mm with factor variables
  inzero, inpos = mapins(inzero, inpos, mm)

  # resolve projdir
  projdir = ixprojdir(trms, sprojdir, mm)

  # fit and wrap in DataFrameRegressionModel
  StatsModels.DataFrameRegressionModel(fit(C, mm.m, counts, projdir, fmargs...; inzero=inzero, inpos=inpos, kwargs...), mf, mm)
end

"Find column number of sprojdir"
function ixprojdir(trms::StatsModels.Terms, sprojdir::Symbol, mm)
  ix = something(findfirst(isequal(sprojdir), trms.terms), 0)
  @assert ix > 0 "$sprojdir not found in provided dataframe"
  mappedix = something(findfirst(isequal(ix), mm.assign), 0)
  @assert mappedix > 0 "$sprojdir not found in ModelMatrix (perhaps redundant?)"
  mappedix
end

StatsModels.@delegate StatsModels.DataFrameRegressionModel.model [coeffwd, coefbwd, srproj, srprojX]

"""
Predict using a fitted Counts inverse regression (CIR) given new covars and counts.

# Keywords
- Set `nocounts=true` to predict using a benchmark model without counts.
"""
function StatsBase.predict(m::CIR,covars::AbstractMatrix{T},counts::AbstractMatrix{V};
  nocounts=false) where {T<:AbstractFloat,V}

  # calculate srproj design matrices for regressions
  X, X_nocounts, inz = srprojX(m.bwdm,counts,covars,m.projdir;inz=m.inz,testrank=false)

  # use forward model to predict
  if nocounts
    if isdefined(m,:fwdmnocounts)
      predict(m.fwdmnocounts,X_nocounts)
    else
      error("To predict with benchmark model w/o counts the CIR model must be fit with nocounts=true")
    end
  else
    predict(m.fwdm,X)
  end
end

"""
Predict using a fitted Counts inverse regression (CIR) given new covars dataframe
and counts. See also [`predict(::CIR)`](@ref).
"""
function StatsBase.predict(mm::MM, df::AbstractDataFrame, counts::AbstractMatrix;
  kwargs...) where {T,M<:CIR,MM<:Union{CIR,StatsModels.DataFrameRegressionModel{M,T}}}
    # NOTE: this code is copied from StatsModels/statsmodel.jl's version of predict
    # copy terms, removing outcome if present (ModelFrame will complain if a
    # term is not found in the DataFrame and we don't want to remove elements with missing y)
    newTerms = StatsModels.dropresponse!(mm.mf.terms)
    # create new model frame/matrix
    newTerms.intercept = true
    mf = ModelFrame(newTerms, df; contrasts = mm.mf.contrasts)
    mf.terms.intercept = false

    newX = ModelMatrix(mf).m
    if !all(mf.nonmissing)
      counts = counts[mf.nonmissing,:]
    end
    yp = predict(mm, newX, counts; kwargs...)
    out = missings(eltype(yp), size(df, 1))
    out[mf.nonmissing] = yp
    return(out)
end
# # when the backward model is an HDMR we need to make sure we didn't drop a colinear zpos
# function StatsBase.predict(m::C,covars::AbstractMatrix{T},counts::AbstractMatrix{V};
#   nocounts=false) where {T<:AbstractFloat,V,BM<:HDMR,FM,C<:CIR{BM,FM}}
#
#   # calculate srproj design matrices for regressions
#   X, X_nocounts, includezpos = srprojX(m.bwdm,counts,covars,m.projdir)
#
#   # use forward model to predict
#   if nocounts
#     if isdefined(m,:fwdmnocounts)
#       predict(m.fwdmnocounts,X_nocounts)
#     else
#       error("To predict with benchmark model w/o counts the CIR model must be fit with nocounts=true")
#     end
#   else
#     predict(m.fwdm,X)
#   end
# end

"""
Returns coefficients of forward regression model.
Set nocounts=true to get coefficients for the benchmark model without counts.
"""
function coeffwd(m::CIR; nocounts=false)
  if nocounts
    if isdefined(m,:fwdmnocounts)
      coef(m.fwdmnocounts)
    else
      error("To get coef of benchmark model w/o counts the CIR model must be fit with nocounts=true")
    end
  else
    coef(m.fwdm)
  end
end
StatsBase.coef(m::CIR; kwargs...) = coeffwd(m; kwargs...)

"Returns coefficients for backward model for counts as function of covariates"
coefbwd(m::CIR; kwargs...) = coef(m.bwdm; kwargs...)

StatsBase.r2(m::CIR; nocounts=false) = (nocounts ? r2(m.fwdmnocounts) : r2(m.fwdm))
StatsBase.adjr2(m::CIR; nocounts=false) = (nocounts ? adjr2(m.fwdmnocounts) : adjr2(m.fwdm))
