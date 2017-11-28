__precompile__()

module HurdleDMR

using Lasso, GLM.FPVector, DataFrames, Reexport, Compat

@reexport using GLM, StatsBase, Distributions

export dmr, dmr2, dmrpath, hdmr, hdmr2, hdmrpath, DMR, collapse, fit, coef, srproj, @~, mcdmr
export PositivePoisson, LogProductLogLink, logpdf_exact, logpdf_approx
export Hurdle, cross_validate_dmr_srproj, cross_validate_dmr_srproj_for_different_specs, SerialKfold

##############################################
# hurdle glm model involves
#  1. a choice model for the 0 or positive choice (e.g. binomial with logit link)
#  2. a positive (truncated) count model for positive counts (e.g. poisson with logit link)
#############################################

# 1. Positive poisson regression:
include("positive_poisson.jl")
include("hurdle.jl")
include("sparserank.jl")
include("dmr.jl")
include("hdmr.jl")
include("multicounts.jl")
include("cross_validation.jl")

end
