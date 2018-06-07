"""
Counts inverse regression (CIR) model supports both multinomial and hurdle inverse regressions
and holds both the inverse and forward regression model estimates
"""
struct CIR{BM<:DCR,FM<:RegressionModel}
  covars::AbstractMatrix     # n×p covariates matrix
  counts::AbstractMatrix     # n×d counts (document-term) matrix
  projdir::Int               # projection direction is the index of covariate representing the response
  inz::Vector{Int}           # indices of Z=srproj(counts) included in srprojX after dropping colinear ones on call to srprojX!
  bwdm::BM                   # inverse regression model: counts ~ X
  fwdm::FM                   # forward regression model: y ~ srproj(counts) + X
  fwdmnocounts::FM           # forward regression model w/o counts: y ~ X

  CIR{BM,FM}(covars::AbstractMatrix{T}, counts::AbstractMatrix{V}, projdir::Int, inz::Vector{Int}, bwdm::BM, fwdm::FM) where {T<:AbstractFloat,V,BM<:DCR,FM<:RegressionModel} =
    new(covars, counts, projdir, inz, bwdm, fwdm)
  CIR{BM,FM}(covars::AbstractMatrix{T}, counts::AbstractMatrix{V}, projdir::Int, inz::Vector{Int}, bwdm::BM, fwdm::FM, fwdmnocounts::FM) where {T<:AbstractFloat,V,BM<:DCR,FM<:RegressionModel} =
    new(covars, counts, projdir, inz, bwdm, fwdm, fwdmnocounts)
end

"""
Fit a Counts inverse regression (CIR).
Set nocounts=true to also fit a benchmark model without counts
example:
  m = fit(CIR{DMR,LinearModel}, covars, counts, 1; nocounts=true)
  yhat = predict(m, covars, counts)
  yhatnc = predict(m, covars, counts; nocounts=true)
"""
function StatsBase.fit(::Type{C},covars::AbstractMatrix{T},counts::AbstractMatrix{V},projdir::Int;
  nocounts=false, dcrkwargs...) where {T<:AbstractFloat,V,BM<:DCR,FM<:RegressionModel,C<:CIR{BM,FM}}

  # run inverse regression
  bwdm = fit(BM,covars,counts; dcrkwargs...)

  # target variable
  y = covars[:,projdir]

  # calculate srproj design matrices for regressions
  X, X_nocounts, inz = srprojX(bwdm,counts,covars,projdir)

  # forward regression model with counts
  fwdm = fit(FM,X,y)

  if nocounts
    # forward model w/o counts
    fwdmnocounts = fit(FM,X_nocounts,y)

    # wrap in struct
    CIR{BM,FM}(covars, counts, projdir, inz, bwdm, fwdm, fwdmnocounts)
  else
    # wrap in struct
    CIR{BM,FM}(covars, counts, projdir, inz, bwdm, fwdm)
  end
end

"""
Predict using a fitter Counts inverse regression (CIR).
Set nocounts=true to predict using a benchmark model without counts.
"""
function StatsBase.predict(m::CIR,covars::AbstractMatrix{T},counts::AbstractMatrix{V};
  nocounts=false) where {T<:AbstractFloat,V}

  # calculate srproj design matrices for regressions
  X, X_nocounts = srprojX(m.bwdm,counts,covars,m.projdir)

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

# when the backward model is an HDMR we need to make sure we didn't drop a colinear zpos
function StatsBase.predict(m::C,covars::AbstractMatrix{T},counts::AbstractMatrix{V};
  nocounts=false) where {T<:AbstractFloat,V,BM<:HDMR,FM,C<:CIR{BM,FM}}

  # calculate srproj design matrices for regressions
  X, X_nocounts, includezpos = srprojX(m.bwdm,counts,covars,m.projdir)

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
