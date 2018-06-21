__precompile__()

module HurdleDMR

using Lasso, GLM.FPVector, DataFrames, Reexport

@reexport using GLM, StatsBase, Distributions, StatsModels

export dmr, dmrpaths, hdmr, hdmrpaths, fit, coef, srproj, srprojX, @~, mcdmr, posindic
export DCR, DMR, HDMR, DMRCoefs, DMRPaths, HDMRCoefs, HDMRPaths, @model, Model
export CIR, predict, coeffwd, coefbwd
export hasintercept, ncategories, nobs, ncoefs, ncovars, ncovarszero, ncovarspos, ncoefszero, ncoefspos
export Hurdle, PositivePoisson, LogProductLogLink, logpdf_exact, logpdf_approx
export cv, SerialKfold, LeaveOutSample
export CVStats, CVType, CVData, CVDataRow


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
