__precompile__()

module HurdleDMR

using Lasso, StatsBase, StatsModels, DataFrames, LambertW, MLBase, GLM.FPVector, GLM.FP

export dmr, dmrpaths, hdmr, hdmrpaths, fit, coef, srproj, srprojX, @~, mcdmr, posindic,
  DCR, DMR, HDMR, DMRCoefs, DMRPaths, HDMRCoefs, HDMRPaths, @model, Model,
  CIR, predict, coeffwd, coefbwd,
  hasintercept, ncategories, nobs, ncoefs, ncovars, ncovarszero, ncovarspos, ncoefszero, ncoefspos,
  Hurdle, PositivePoisson, LogProductLogLink, logpdf_exact, logpdf_approx,
  cv, cvstats, SerialKfold, LeaveOutSample,
  CVType, CVData, CVDataRow

##############################################
# hurdle glm model involves
#  1. a choice model for the 0 or positive choice (e.g. binomial with logit link)
#  2. a positive (truncated) count model for positive counts (e.g. poisson with logit link)
#############################################

include("positive_poisson.jl")
include("hurdle.jl")
include("sparserank.jl")
include("model.jl")
include("dmr.jl")
include("hdmr.jl")
include("srproj.jl")
include("invreg.jl")
include("multicounts.jl")
include("cross_validation.jl")

end
