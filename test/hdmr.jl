include("testutils.jl")

using Base.Test, Gadfly, Distributions

include("addworkers.jl")

using CSV, GLM, DataFrames, LassoPlot

import HurdleDMR; @everywhere using HurdleDMR

# reload("HurdleDMR")

we8thereCounts = CSV.read(joinpath(testdir,"data","dmr_we8thereCounts.csv.gz"))
we8thereRatings = CSV.read(joinpath(testdir,"data","dmr_we8thereRatings.csv.gz"))
we8thereTerms = map(string,names(we8thereCounts))

# covars = we8thereRatings[:,[:Overall]]
covars = we8thereRatings[:,:]
n,p = size(covars)
inzero = 1:p

inpos = [1,3]
covarspos = we8thereRatings[:,inpos]

T = Float64
# counts=sparse(convert(Matrix{T},we8thereCounts))
d = 100
srand(13)
counts = round.(10*sprand(n,d,0.3))
covars=convert(Array{T,2},covars)
covarspos=convert(Array{T,2},covarspos)

npos,ppos = size(covarspos)
d = size(counts,2)
γ=1.0

###########################################################
# hurdle with covarspos == covarszero
###########################################################
@testset "hurdle-dmr with covarspos == covarszero" begin

# reload("HurdleDMR")

# hurdle dmr parallel local cluster
@time hdmrcoefs = hdmr(covars, counts; parallel=true, verbose=true)
coefsHppos, coefsHpzero = coef(hdmrcoefs)
@test size(coefsHppos) == (p+1, d)
@test size(coefsHpzero) == (p+1, d)

@time hdmrcoefsb = fit(HDMRCoefs, covars, counts; parallel=true, verbose=true)
@test coef(hdmrcoefsb)[1] == coefsHppos
@test coef(hdmrcoefsb)[2] == coefsHpzero
@test n == nobs(hdmrcoefs)
@test d == ncategories(hdmrcoefs)
@test p == ncovarspos(hdmrcoefs)
@test p == ncovarszero(hdmrcoefs)

# hurdle dmr parallel remote cluster
@time hdmrcoefs2 = hdmr(covars, counts; parallel=true, local_cluster=false)
coefsHppos2, coefsHpzero2 = coef(hdmrcoefs)
@test coefsHppos ≈ coefsHppos2
@test coefsHpzero ≈ coefsHpzero2
@time hdmrcoefs2 = fit(HDMRCoefs, covars, counts; parallel=true, local_cluster=false)
@test coef(hdmrcoefs2)[1] ≈ coefsHppos2
@test coef(hdmrcoefs2)[2] ≈ coefsHpzero2

# # # hurdle dmr serial
# @time hdmrcoefs3 = hdmr(covars, counts; parallel=false)
# coefsHspos, coefsHszero = coef(hdmrcoefs3)
# @test coefsHppos ≈ coefsHspos
# @test coefsHpzero ≈ coefsHszero
# @time hdmrcoefs3 = fit(HDMRCoefs, covars, counts; parallel=false)
# @test coef(hdmrcoefs3)[1] ≈ coefsHspos
# @test coef(hdmrcoefs3)[2] ≈ coefsHszero

# test posindic used by srproj
m = rand(Poisson(0.1),30,500)
ms = sparse(m)
Im = posindic(m)
Ims = posindic(ms)
@test Im == Ims

zHpos = srproj(coefsHppos, counts)
@test size(zHpos) == (n,p+1)

zHzero = srproj(coefsHpzero, posindic(counts))
@test size(zHzero) == (n,p+1)

# first half of coefs belongs to covarspos
z1pos = srproj(coefsHppos, counts, 1)
@test z1pos ≈ zHpos[:,[1,p+1]]

# second half of coefs belongs to covarszero
z1zero = srproj(coefsHpzero, posindic(counts), 1)
@test z1zero ≈ zHzero[:,[1,p+1]]

@time Z1 = srproj(coefsHppos, coefsHpzero, counts, 1, 1; intercept=true)
@test Z1 == [z1pos[:,1] z1zero[:,1] z1pos[:,2]]
@time Z1b = srproj(hdmrcoefs, counts, 1, 1; intercept=true)
@test Z1 == Z1b

Z3 = srproj(coefsHppos, coefsHpzero, counts, 3, 3; intercept=true)
z3pos = srproj(coefsHppos, counts, 3)
z3zero = srproj(coefsHpzero, posindic(counts), 3)
@test Z3 == [z3pos[:,1] z3zero[:,1] z3pos[:,2]]
@time Z3b = srproj(hdmrcoefs, counts, 3, 3; intercept=true)
@test Z3 == Z3b

regdata = DataFrame(y=covars[:,1], zw1=z1zero[:,1], zv1=z1pos[:,1], m=z1pos[:,2])
lmw1 = lm(@formula(y ~ zw1+m), regdata)
r2w1 = adjr2(lmw1)

lmv1 = lm(@formula(y ~ zv1+m), regdata)
r2v1 = adjr2(lmv1)

lmw1v1 = lm(@formula(y ~ zw1+zv1+m), regdata)
r2w1v1 = adjr2(lmw1v1)

X1, X1_nocounts, includezpos = srprojX(coefsHppos,coefsHpzero,counts,covars,1; includem=true)
@test X1_nocounts == [ones(n) covars[:,2:end]]
@test X1 == [X1_nocounts Z1]
X1b, X1_nocountsb, includezposb = srprojX(hdmrcoefs,counts,covars,1; includem=true)
@test X1 == X1b
@test X1_nocountsb == X1_nocountsb
@test includezposb == includezposb

X2, X2_nocounts, includezpos = srprojX(coefsHppos,coefsHpzero,counts,covars,1; includem=false)
@test X2_nocounts == X1_nocounts
@test X2 == X1[:,1:end-1]
X2b, X2_nocountsb, includezposb = srprojX(hdmrcoefs,counts,covars,1; includem=false)
@test X2 == X2b
@test X2_nocountsb == X2_nocountsb
@test includezposb == includezposb

X3, X3_nocounts, includezpos = srprojX(coefsHppos,coefsHpzero,counts,covars,3; includem=true)
@test X3_nocounts == [ones(n) covars[:,[1,2,4,5]]]
@test X3 == [X3_nocounts Z3]
X3b, X3_nocountsb, includezposb = srprojX(hdmrcoefs,counts,covars,3; includem=true)
@test X3 == X3b
@test X3_nocountsb == X3_nocountsb
@test includezposb == includezposb

@time cvstats13 = cross_validate_hdmr_srproj(covars,counts,1; k=2, gentype=MLBase.Kfold, γ=γ)
cvstats13b = cross_validate_hdmr_srproj(covars,counts,1; k=2, gentype=MLBase.Kfold, γ=γ)
@test isequal(cvstats13,cvstats13b)

@time cvstats14 = cross_validate_hdmr_srproj(covars,counts,1; k=2, gentype=MLBase.Kfold, γ=γ, seed=14)
@test !(isequal(cvstats13,cvstats14))

@time cvstatsSerialKfold = cross_validate_hdmr_srproj(covars,counts,1; k=3, gentype=SerialKfold, γ=γ)
@test !(isequal(cvstats13,cvstatsSerialKfold))

end

####################################################################
# hurdle with covarspos ≠ covarszero, both models includes projdir
####################################################################
@testset "hurdle-dmr with covarspos ≠ covarszero, both models includes projdir" begin

# hurdle dmr parallel local cluster
@time hdmrcoefs = hdmr(covars, counts; covarspos=covarspos, parallel=true)
coefsHppos, coefsHpzero = coef(hdmrcoefs)
@test size(coefsHppos) == (ppos+1, d)
@test size(coefsHpzero) == (p+1, d)

@time hdmrcoefsb = fit(HDMRCoefs, covars, counts; covarspos=covarspos, parallel=true, verbose=true)
@test coef(hdmrcoefsb)[1] == coefsHppos
@test coef(hdmrcoefsb)[2] == coefsHpzero
@test n == nobs(hdmrcoefs)
@test d == ncategories(hdmrcoefs)
@test ppos == ncovarspos(hdmrcoefs)
@test p == ncovarszero(hdmrcoefs)

# hurdle dmr parallel remote cluster
@time hdmrcoefs2 = hdmr(covars, counts; covarspos=covarspos, parallel=true, local_cluster=false)
coefsHppos2, coefsHpzero2 = coef(hdmrcoefs)
@test coefsHppos ≈ coefsHppos2
@test coefsHpzero ≈ coefsHpzero2
@time hdmrcoefs2 = fit(HDMRCoefs, covars, counts; covarspos=covarspos, parallel=true, local_cluster=false)
@test coef(hdmrcoefs2)[1] ≈ coefsHppos2
@test coef(hdmrcoefs2)[2] ≈ coefsHpzero2

# # # hurdle dmr serial
# @time hdmrcoefs3 = hdmr(covars, counts; covarspos=covarspos, parallel=false)
# coefsHspos, coefsHszero = coef(hdmrcoefs3)
# @test coefsHppos ≈ coefsHspos
# @test coefsHpzero ≈ coefsHszero
# @time hdmrcoefs3 = fit(HDMRCoefs, covars, counts; covarspos=covarspos, parallel=false)
# @test coef(hdmrcoefs3)[1] ≈ coefsHspos
# @test coef(hdmrcoefs3)[2] ≈ coefsHszero

zHpos = srproj(coefsHppos, counts)
@test size(zHpos) == (n,ppos+1)

zHzero = srproj(coefsHpzero, posindic(counts))
@test size(zHzero) == (n,p+1)

# first half of coefs belongs to covarspos
z1pos = srproj(coefsHppos, counts, 1)
@test z1pos ≈ zHpos[:,[1,ppos+1]]

# second half of coefs belongs to covarszero
z1zero = srproj(coefsHpzero, posindic(counts), 1)
@test z1zero ≈ zHzero[:,[1,p+1]]

Z1 = srproj(coefsHppos, coefsHpzero, counts, 1, 1; intercept=true)
@test Z1 == [z1pos[:,1] z1zero[:,1] z1pos[:,2]]
@time Z1b = srproj(hdmrcoefs, counts, 1, 1; intercept=true)
@test Z1 == Z1b

Z3 = srproj(coefsHppos, coefsHpzero, counts, 2, 3; intercept=true)
z3pos = srproj(coefsHppos, counts, 2)
z3zero = srproj(coefsHpzero, posindic(counts), 3)
@test Z3 == [z3pos[:,1] z3zero[:,1] z3pos[:,2]]
@time Z3b = srproj(hdmrcoefs, counts, 2, 3; intercept=true)
@test Z3 == Z3b

regdata = DataFrame(y=covars[:,1], zw1=z1zero[:,1], zv1=z1pos[:,1], m=z1pos[:,2], v1=covarspos[:,1], w1=covars[:,1])
lmw1 = lm(@formula(y ~ zw1+m), regdata)
r2w1 = adjr2(lmw1)

lmv1 = lm(@formula(y ~ zv1+m), regdata)
r2v1 = adjr2(lmv1)

lmw1v1 = lm(@formula(y ~ zw1+zv1+m), regdata)
r2w1v1 = adjr2(lmw1v1)

@time X1, X1_nocounts, includezpos = srprojX(coefsHppos,coefsHpzero,counts,covars,1; inpos=inpos, includem=true)
@test X1_nocounts == [ones(n) covars[:,2:end]]
@test X1 == [X1_nocounts Z1]
X1b, X1_nocountsb, includezposb = srprojX(hdmrcoefs,counts,covars,1; inpos=inpos, includem=true)
@test X1 == X1b
@test X1_nocountsb == X1_nocountsb
@test includezposb == includezposb

X2, X2_nocounts, includezpos = srprojX(coefsHppos,coefsHpzero,counts,covars,1; inpos=inpos, includem=false)
@test X2_nocounts == X1_nocounts
@test X2 == X1[:,1:end-1]
X2b, X2_nocountsb, includezposb = srprojX(hdmrcoefs,counts,covars,1; inpos=inpos, includem=false)
@test X2 == X2b
@test X2_nocountsb == X2_nocountsb
@test includezposb == includezposb

X3, X3_nocounts, includezpos = srprojX(coefsHppos,coefsHpzero,counts,covars,3; inpos=inpos, includem=true)
@test X3_nocounts == [ones(n) covars[:,[1,2,4,5]]]
@test X3 == [X3_nocounts Z3]
X3b, X3_nocountsb, includezposb = srprojX(hdmrcoefs,counts,covars,3; inpos=inpos, includem=true)
@test X3 == X3b
@test X3_nocountsb == X3_nocountsb
@test includezposb == includezposb

@time cvstats13 = cross_validate_hdmr_srproj(covars,counts,1; inpos=inpos, k=2, gentype=MLBase.Kfold, γ=γ)
@time cvstats13b = cross_validate_hdmr_srproj(covars,counts,1; inpos=inpos, k=2, gentype=MLBase.Kfold, γ=γ)
@test isequal(cvstats13,cvstats13b)

@time cvstats14 = cross_validate_hdmr_srproj(covars,counts,1; inpos=inpos, k=2, gentype=MLBase.Kfold, γ=γ, seed=14)
@test !(isequal(cvstats13,cvstats14))

@time cvstatsSerialKfold = cross_validate_hdmr_srproj(covars,counts,1; inpos=inpos, k=3, gentype=SerialKfold, γ=γ)
@test !(isequal(cvstats13,cvstatsSerialKfold))

end

########################################################################
# hurdle with covarspos ≠ covarszero, only pos model includes projdir
########################################################################
@testset "hurdle-dmr with covarspos ≠ covarszero, only pos model includes projdir" begin

covarszero = covars[:,2:end]
nzero,pzero = size(covarszero)
inzero = 2:p

# hurdle dmr parallel local cluster
@time hdmrcoefs = hdmr(covarszero, counts; covarspos=covarspos, parallel=true)
coefsHppos, coefsHpzero = coef(hdmrcoefs)
@test size(coefsHppos) == (ppos+1, d)
@test size(coefsHpzero) == (pzero+1, d)

@time hdmrcoefsb = fit(HDMRCoefs, covarszero, counts; covarspos=covarspos, parallel=true)
@test coef(hdmrcoefsb)[1] == coefsHppos
@test coef(hdmrcoefsb)[2] == coefsHpzero
@test n == nobs(hdmrcoefs)
@test d == ncategories(hdmrcoefs)
@test ppos == ncovarspos(hdmrcoefs)
@test pzero == ncovarszero(hdmrcoefs)

# hurdle dmr parallel remote cluster
@time hdmrcoefs2 = hdmr(covarszero, counts; covarspos=covarspos, parallel=true, local_cluster=false)
coefsHppos2, coefsHpzero2 = coef(hdmrcoefs)
@test coefsHppos ≈ coefsHppos2
@test coefsHpzero ≈ coefsHpzero2
@time hdmrcoefs2 = fit(HDMRCoefs, covarszero, counts; covarspos=covarspos, parallel=true, local_cluster=false)
@test coef(hdmrcoefs2)[1] ≈ coefsHppos2
@test coef(hdmrcoefs2)[2] ≈ coefsHpzero2

# # hurdle dmr serial
@time hdmrcoefs3 = hdmr(covarszero, counts; covarspos=covarspos, parallel=false)
coefsHspos, coefsHszero = coef(hdmrcoefs3)
@test coefsHppos ≈ coefsHspos
@test coefsHpzero ≈ coefsHszero
@time hdmrcoefs3 = fit(HDMRCoefs, covarszero, counts; covarspos=covarspos, parallel=false)
@test coef(hdmrcoefs3)[1] ≈ coefsHspos
@test coef(hdmrcoefs3)[2] ≈ coefsHszero

zHpos = srproj(coefsHppos, counts)
@test size(zHpos) == (n,ppos+1)

zHzero = srproj(coefsHpzero, posindic(counts))
@test size(zHzero) == (n,pzero+1)

z1pos = srproj(coefsHppos, counts, 1)
@test z1pos ≈ zHpos[:,[1,ppos+1]]

# projdir is not included in zero model
# z1zero = srproj(coefsHpzero, posindic(counts), 1)
# @test z1zero ≈ zHzero[:,[1,p+1]]

Z1 = srproj(coefsHppos, coefsHpzero, counts, 1, 0; intercept=true)
@test Z1 == z1pos
@time Z1b = srproj(hdmrcoefs, counts, 1, 0; intercept=true)
@test Z1 == Z1b

Z3 = srproj(coefsHppos, coefsHpzero, counts, 2, 2; intercept=true)
z3pos = srproj(coefsHppos, counts, 2)
z3zero = srproj(coefsHpzero, posindic(counts), 2)
@test Z3 == [z3pos[:,1] z3zero[:,1] z3pos[:,2]]
@time Z3b = srproj(hdmrcoefs, counts, 2, 2; intercept=true)
@test Z3 == Z3b

regdata = DataFrame(y=covars[:,1], zv1=z1pos[:,1], m=z1pos[:,2], v1=covarspos[:,1], w1=covarszero[:,1])

lmv1 = lm(@formula(y ~ zv1+m), regdata)
r2v1 = adjr2(lmv1)

@time X1, X1_nocounts, includezpos = srprojX(coefsHppos,coefsHpzero,counts,covars,1; inzero=inzero, inpos=inpos, includem=true)
@test X1_nocounts == [ones(n) covars[:,2:end]]
@test X1 == [X1_nocounts Z1]
X1b, X1_nocountsb, includezposb = srprojX(hdmrcoefs,counts,covars,1; inzero=inzero, inpos=inpos, includem=true)
@test X1 == X1b
@test X1_nocountsb == X1_nocountsb
@test includezposb == includezposb

X2, X2_nocounts, includezpos = srprojX(coefsHppos,coefsHpzero,counts,covars,1; inzero=inzero, inpos=inpos, includem=false)
@test X2_nocounts == X1_nocounts
@test X2 == X1[:,1:end-1]
X2b, X2_nocountsb, includezposb = srprojX(hdmrcoefs,counts,covars,1; inzero=inzero, inpos=inpos, includem=false)
@test X2 == X2b
@test X2_nocountsb == X2_nocountsb
@test includezposb == includezposb

X3, X3_nocounts, includezpos = srprojX(coefsHppos,coefsHpzero,counts,covars,3; inzero=inzero, inpos=inpos, includem=true)
@test X3_nocounts == [ones(n) covars[:,[2,4,5,1]]]
@test X3 == [X3_nocounts Z3]
X3b, X3_nocountsb, includezposb = srprojX(hdmrcoefs,counts,covars,3; inzero=inzero, inpos=inpos, includem=true)
@test X3 == X3b
@test X3_nocountsb == X3_nocountsb
@test includezposb == includezposb

@time cvstats13 = cross_validate_hdmr_srproj(covars,counts,1; inzero=inzero, inpos=inpos, k=2, gentype=MLBase.Kfold, γ=γ)
@time cvstats13b = cross_validate_hdmr_srproj(covars,counts,1; inzero=inzero, inpos=inpos, k=2, gentype=MLBase.Kfold, γ=γ)
@test isequal(cvstats13,cvstats13b)

@time cvstats14 = cross_validate_hdmr_srproj(covars,counts,1; inzero=inzero, inpos=inpos, k=2, gentype=MLBase.Kfold, γ=γ, seed=14)
@test !(isequal(cvstats13,cvstats14))

@time cvstatsSerialKfold = cross_validate_hdmr_srproj(covars,counts,1; inzero=inzero, inpos=inpos, k=3, gentype=SerialKfold, γ=γ)
@test !(isequal(cvstats13,cvstatsSerialKfold))

end

@testset "degenerate cases" begin

# column j is never zero, so hj=1 for all observations
zcounts = deepcopy(counts)
zcounts[:,2] = zeros(n)
zcounts[:,3] = ones(n)
# sum(iszero.(zcounts[:,nzj]))
# find(var(zcounts,1) .== 0)

# make sure we are not adding all zero obseravtions
m = sum(zcounts,2)
@test sum(m .== 0) == 0

# rows,cols = 40000,30000
# @time S = SharedMatrix{Float64}(rows,cols; init=nothing)
# @time A = convert(SharedMatrix{Float64},Matrix{Float64}(rows,cols))

# hurdle dmr parallel local cluster
# hdmr(covars, zcounts[:,2:3]; parallel=false, showwarnings=true, verbose=true)
@time hdmrcoefs = hdmr(covars, zcounts; parallel=true, showwarnings=true, verbose=false)
coefsHppos, coefsHpzero = coef(hdmrcoefs)
@test size(coefsHppos) == (p+1, d)
@test size(coefsHpzero) == (p+1, d)
@test coefsHppos[:,2] == zeros(p+1)
@test coefsHpzero[:,2] == zeros(p+1)
@test coefsHppos[:,3] == zeros(p+1)
@test coefsHpzero[:,3] == zeros(p+1)

# hurdle dmr parallel remote cluster
@time hdmrcoefs2 = hdmr(covars, zcounts; parallel=true, local_cluster=false)
coefsHppos2, coefsHpzero2 = coef(hdmrcoefs)
@test coefsHppos ≈ coefsHppos2
@test coefsHpzero ≈ coefsHpzero

end

@testset "degenerate case of no hurdle variation (all counts > 0)" begin

zcounts = full(deepcopy(counts))
srand(13)
for I = eachindex(zcounts)
    if iszero(zcounts[I])
        zcounts[I] = rand(1:10)
    end
end

# make sure we are not adding all zero obseravtions
m = sum(zcounts,2)
@test sum(m .== 0) == 0

# hurdle dmr parallel local cluster
@time hdmrcoefs = hdmr(covars, zcounts; parallel=true)
coefsHppos, coefsHpzero = coef(hdmrcoefs)
@test size(coefsHppos) == (p+1, d)
@test size(coefsHpzero) == (p+1, d)
@test coefsHppos[:,2] != zeros(p+1)
@test coefsHpzero[:,2] == zeros(p+1)
@test coefsHppos[:,3] != zeros(p+1)
@test coefsHpzero[:,3] == zeros(p+1)

# hurdle dmr parallel remote cluster
@time hdmrcoefs2 = hdmr(covars, zcounts; parallel=true, local_cluster=false)
coefsHppos2, coefsHpzero2 = coef(hdmrcoefs2)
@test coefsHppos ≈ coefsHppos2
@test coefsHpzero ≈ coefsHpzero

end

###########################################################
# Development
###########################################################
# reload("HurdleDMR")


###########################################################
# End Development
###########################################################

# # exception analysis code:
# (obj,objold,path,f,newcoef,oldcoef,b0,oldb0,b0diff,coefdiff,scratchmu,cd,r,α,curλ)=(0,0,[],0,[],[],0,0,0,[],[],[],[],0,0)
# ex=[]
# try
#   coefsH = dmr(covars, counts; hurdle=true, parallel=false, verbose=true)
# catch e
#   # if typeof(e) == Lasso.ConvergenceException
#   #   (obj,objold,path,f,newcoef,oldcoef,b0,oldb0,b0diff,coefdiff,scratchmu,cd,r,α,curλ) = e.debugvars
#   #   m = path.m
#   #   warn(e.msg)
#   #   nothing
#   if typeof(e) <: ErrorException
#     warn("caught exception $e")
#     ex=e
#   else
#     ex=e
#   end
# end
# typeof(ex) == Base.LinAlg.PosDefException
# coefsH
#
# path.m.rr
# ex.msg
# fieldnames(ex)
# rr = path.m.rr
# pp = path.m.pp

#   end
# end #do facts

# @time b0serial, coefsserial = dmr(covars, counts; γ=γ, nlocal_workers=1, irls_maxiter=2000)
# using ProfileView
# Profile.init(delay=0.001)
# Profile.clear()
# @profile b0, coefs = dmr(covars, counts; irls_maxiter=2000);
# ProfileView.view()
# # Profile.print()
#
# size(b0)
# size(coefs)
# [b0 coefs']'
#


#
# rmprocs(workers())
