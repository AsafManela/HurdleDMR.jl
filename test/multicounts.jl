include("testutils.jl")

using Base.Test, Distributions

include("addworkers.jl")

using CSV, GLM, DataFrames, LassoPlot

import HurdleDMR; @everywhere using HurdleDMR
# uncomment following for debugging and comment the previous @everywhere line. then use reload after making changes
# reload("HurdleDMR")

we8thereCounts = CSV.read(joinpath(testdir,"data","dmr_we8thereCounts.csv.gz"))
we8thereRatings = CSV.read(joinpath(testdir,"data","dmr_we8thereRatings.csv.gz"))
we8thereTerms = broadcast(string,names(we8thereCounts))

# covars = we8thereRatings[:,[:Overall]]
covars = we8thereRatings[:,:]
n,p = size(covars)
inzero = 1:p

inpos = [1,3]
covarspos = we8thereRatings[:,inpos]

T = Float64
# split counts matrix to 3 multicounts vector
d = size(we8thereCounts,2)

# to make sure m>0 in all of these, we sum the base test counts at different horizons
# counts1 = sparse(convert(Matrix{T},we8thereCounts[:,:]))
d = 100
# counts=sparse(convert(Matrix{Float64},we8thereCounts[:,end-d+1:end]))
srand(13)
counts1 = round.(10*sprand(n,d,0.3))

counts2 = counts1[:,:]
counts2[2:end,:] += counts1[1:end-1,:]
counts3 = counts2[:,:]
counts3[3:end,:] += counts1[1:end-2,:]
multicounts = [counts1, counts2, counts3]

covars=convert(Array{T,2},covars)
covarspos=convert(Array{T,2},covarspos)
# counts = hcat(multicounts...)
# typeof(counts)
# typeof(multicounts[1])
# typeof(hcat(multicounts[1:1]...))
# Z = Array(T,n,0)
# [covars Z] == covars

npos,ppos = size(covarspos)

γ=1.0

@testset "mcdmr" begin

@time dmrcoefs = dmr(covars, multicounts[1]; γ=γ, λminratio=0.01)

@time Z, multicoefs = mcdmr(covars, multicounts, 1; γ=γ, λminratio=0.01)

coefs = coef(dmrcoefs)
@test size(coefs) == (p+1, d)
@test coef(multicoefs[1]) == coefs
@test size(coef(multicoefs[2]),2) == size(coefs,2)
@test size(coef(multicoefs[2]),1) == size(coefs,1) + 2
@test size(coef(multicoefs[3]),2) == size(coefs,2)
@test size(coef(multicoefs[3]),1) == size(coefs,1) + 4

end

# rmprocs(workers())
