################################################################################
# Squencial processing of multiple counts matrices
################################################################################

"""
Multiple Counts Distributed Multinomial Regression by running independent poisson gamma lasso regression to each column of counts,
picks the minimum AICc segement of each path, and returns a coefficient matrix (coefs) representing point estimates
for the entire multinomial (includes the intercept if one was included).
Unlike dmr(), mcdmr() takes a vector of counts matrices and sequencially
  1. regresses each counts matrix on the covars genrating an srproj Z (includes total counts m)
  2. accumulated srproj z into Z
  3. uses [covars Z] for subsequent regressions
Returns the accumulated Z matrix and a vector of coefficient matrices
"""
function mcdmr(covars::AbstractMatrix{T},multicounts::Vector,projdir::Int;
    verbose=true, kwargs...) where {T<:AbstractFloat}
  L = length(multicounts)
  n,p = size(covars)
  verbose && @info("fitting mcdmr to $L counts matrices")
  multicoefs = Vector{DMRCoefs}(undef,L)
  Z = Array{T}(undef,n,0)
	for l=1:L
		verbose && @info("fitting dmr to counts matrix #$l on $p covars + $(size(Z,2)) previous SR projections ...")
		# get l'th counts matrix
		counts = multicounts[l]
		# fit dmr
		multicoefs[l] = dmr([covars Z],counts; verbose=verbose, kwargs...)
		# collapse counts into low dimensional SR projection
		z = srproj(multicoefs[l],counts,projdir)
		# append z to Z
	  Z = [Z z]
	end
  Z, multicoefs
end
