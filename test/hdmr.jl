include("testutils.jl")

using Distributions

include("addworkers.jl")

using CSV, GLM, DataFrames, LassoPlot

import HurdleDMR; @everywhere using HurdleDMR

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
# counts=sparse(convert(Matrix{T},we8thereCounts))
d = 100
srand(13)
counts = round.(10*sprand(n,d,0.3))
covars=convert(Array{T,2},covars)
covarspos=convert(Array{T,2},covarspos)

newcovars = covars[1:10,:]

npos,ppos = size(covarspos)
d = size(counts,2)
γ=1.0

###########################################################
# hurdle with covarspos == covarszero
###########################################################
@testset "hurdle-dmr with covarspos == covarszero" begin

f = @model(h ~ Food + Service + Value + Atmosphere + Overall, c ~ Food + Service + Value + Atmosphere + Overall)

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
coefsHppos2, coefsHpzero2 = coef(hdmrcoefs2)
@test coefsHppos ≈ coefsHppos2
@test coefsHpzero ≈ coefsHpzero2
@time hdmrcoefs2 = fit(HDMRCoefs, covars, counts; parallel=true, local_cluster=false)
@test coef(hdmrcoefs2)[1] ≈ coefsHppos2
@test coef(hdmrcoefs2)[2] ≈ coefsHpzero2
@test_throws ErrorException predict(hdmrcoefs2,newcovars)

@time hdmrpaths3 = fit(HDMRPaths, covars, counts; parallel=true, verbose=true)
coefsHppos3, coefsHpzero3 = coef(hdmrpaths3)
@test coefsHppos3 ≈ coefsHppos
@test coefsHpzero3 ≈ coefsHpzero
@test n == nobs(hdmrpaths3)
@test d == ncategories(hdmrpaths3)
@test p == ncovarspos(hdmrpaths3)
@test p == ncovarszero(hdmrpaths3)
@test size(hdmrpaths3.nlpaths,1) == 100
η = predict(hdmrpaths3,newcovars)
@test sum(η,2) ≈ ones(size(η,1))

# # # hurdle dmr serial
# @time hdmrcoefs3 = hdmr(covars, counts; parallel=false)
# coefsHspos, coefsHszero = coef(hdmrcoefs3)
# @test coefsHppos ≈ coefsHspos
# @test coefsHpzero ≈ coefsHszero
# @time hdmrcoefs3 = fit(HDMRCoefs, covars, counts; parallel=false)
# @test coef(hdmrcoefs3)[1] ≈ coefsHspos
# @test coef(hdmrcoefs3)[2] ≈ coefsHszero

# using a dataframe and formula
@time hdmrcoefsdf = fit(HDMRCoefs, f, we8thereRatings, counts; parallel=true, verbose=true)
@test coef(hdmrcoefsdf)[1] == coefsHppos
@test coef(hdmrcoefsdf)[2] ≈ coefsHpzero
@test_throws ErrorException predict(hdmrcoefsdf,newcovars)

@time hdmrpathsdf = fit(HDMRPaths, f, we8thereRatings, counts; parallel=true, verbose=true)
@test coef(hdmrpathsdf)[1] == coefsHppos3
@test coef(hdmrpathsdf)[2] ≈ coefsHpzero3
@test predict(hdmrpathsdf,newcovars) ≈ η


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

# HIR
@time hir = fit(CIR{HDMR,LinearModel},covars,counts,1; nocounts=true)
@test coefbwd(hir)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hir)[2] ≈ coef(hdmrcoefs)[2]

@time hirglm = fit(CIR{HDMR,GeneralizedLinearModel},covars,counts,1,Poisson(); nocounts=true)
@test coefbwd(hirglm)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hirglm)[2] ≈ coef(hdmrcoefs)[2]
@test !(coeffwd(hirglm)[1] ≈ coeffwd(hir)[1])
@test !(coeffwd(hirglm)[2] ≈ coeffwd(hir)[2])

@time hirdf = fit(CIR{HDMR,LinearModel},f,we8thereRatings,counts,:Food; nocounts=true)
@test coefbwd(hirdf)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hirdf)[2] ≈ coef(hdmrcoefs)[2]
@test coeffwd(hirdf) ≈ coeffwd(hir)
@time hirglmdf = fit(CIR{HDMR,GeneralizedLinearModel},f,we8thereRatings,counts,:Food,Poisson(); nocounts=true)
@test coefbwd(hirglmdf)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hirglmdf)[2] ≈ coef(hdmrcoefs)[2]
@test !(coeffwd(hirglmdf)[2] ≈ coeffwd(hirdf)[2])
@test !(coeffwd(hirglmdf)[1] ≈ coeffwd(hirdf)[1])

zlm = lm(hcat(ones(n,1),Z1,covars[:,2:end]),covars[:,1])
@test r2(zlm) ≈ r2(hir)
@test adjr2(zlm) ≈ adjr2(hir)
predict(zlm,hcat(ones(10,1),Z1[1:10,:],covars[1:10,2:end]))
predict(hir,covars[1:10,:],counts[1:10,:])
@test predict(zlm,hcat(ones(10,1),Z1[1:10,:],covars[1:10,2:end])) ≈ predict(hir,covars[1:10,:],counts[1:10,:])

zlmnocounts = lm(hcat(ones(n,1),covars[:,2:end]),covars[:,1])
@test r2(zlmnocounts) ≈ r2(hir; nocounts=true)
@test adjr2(zlmnocounts) ≈ adjr2(hir; nocounts=true)
@test predict(zlmnocounts,hcat(ones(10,1),covars[1:10,2:end])) ≈ predict(hir,covars[1:10,:],counts[1:10,:]; nocounts=true)

# CV
srand(13)
@time cvstats13 = cv(CIR{HDMR,LinearModel},covars,counts,1; gen=MLBase.Kfold(size(covars,1),2), γ=γ)
@time cvstats13b = cv(CIR{HDMR,LinearModel},f,we8thereRatings,counts,:Food; k=2, gentype=MLBase.Kfold, γ=γ)
@test isapprox(cvstats13,cvstats13b)

@time cvstats14 = cv(CIR{HDMR,LinearModel},covars,counts,1; k=2, gentype=MLBase.Kfold, γ=γ, seed=14)
@test !(isapprox(cvstats13,cvstats14))

cvstats13glm = cv(CIR{HDMR,GeneralizedLinearModel},f,we8thereRatings,counts,:Food,Poisson(); k=2, gentype=MLBase.Kfold, γ=γ)
@test !(isapprox(cvstats13,cvstats13glm))

@time cvstatsSerialKfold = cv(CIR{HDMR,LinearModel},covars,counts,1; k=3, gentype=SerialKfold, γ=γ)
@test_throws DimensionMismatch isapprox(cvstats13,cvstatsSerialKfold)

end

####################################################################
# hurdle with covarspos ≠ covarszero, both models includes projdir
####################################################################
@testset "hurdle-dmr with covarspos ≠ covarszero, both models includes projdir" begin

f = @model(h ~ Food + Service + Value + Atmosphere + Overall, c ~ Food + Value)

# hurdle dmr parallel local cluster
@time hdmrcoefs = hdmr(covars, counts; inpos=inpos, parallel=true)
coefsHppos, coefsHpzero = coef(hdmrcoefs)
@test size(coefsHppos) == (ppos+1, d)
@test size(coefsHpzero) == (p+1, d)

@time hdmrcoefsb = fit(HDMRCoefs, covars, counts; inpos=inpos, parallel=true, verbose=true)
@test coef(hdmrcoefsb)[1] == coefsHppos
@test coef(hdmrcoefsb)[2] == coefsHpzero
@test n == nobs(hdmrcoefsb)
@test d == ncategories(hdmrcoefsb)
@test ppos == ncovarspos(hdmrcoefsb)
@test p == ncovarszero(hdmrcoefsb)

# hurdle dmr parallel remote cluster
@time hdmrcoefs2 = hdmr(covars, counts; inpos=inpos, parallel=true, local_cluster=false)
coefsHppos2, coefsHpzero2 = coef(hdmrcoefs2)
@test coefsHppos ≈ coefsHppos2
@test coefsHpzero ≈ coefsHpzero2
@time hdmrcoefs2 = fit(HDMRCoefs, covars, counts; inpos=inpos, parallel=true, local_cluster=false)
@test coef(hdmrcoefs2)[1] ≈ coefsHppos2
@test coef(hdmrcoefs2)[2] ≈ coefsHpzero2

@time hdmrpaths3 = fit(HDMRPaths, covars, counts; inpos=inpos, parallel=true, verbose=true)
coefsHppos3, coefsHpzero3 = coef(hdmrpaths3)
@test coefsHppos3 ≈ coefsHppos
@test coefsHpzero3 ≈ coefsHpzero
@test n == nobs(hdmrpaths3)
@test d == ncategories(hdmrpaths3)
@test ppos == ncovarspos(hdmrpaths3)
@test p == ncovarszero(hdmrpaths3)
@test size(hdmrpaths3.nlpaths,1) == 100
η = predict(hdmrpaths3,newcovars)
@test sum(η,2) ≈ ones(size(η,1))

# # # hurdle dmr serial
# @time hdmrcoefs3 = hdmr(covars, counts; covarspos=covarspos, parallel=false)
# coefsHspos, coefsHszero = coef(hdmrcoefs3)
# @test coefsHppos ≈ coefsHspos
# @test coefsHpzero ≈ coefsHszero
# @time hdmrcoefs3 = fit(HDMRCoefs, covars, counts; covarspos=covarspos, parallel=false)
# @test coef(hdmrcoefs3)[1] ≈ coefsHspos
# @test coef(hdmrcoefs3)[2] ≈ coefsHszero

# using a dataframe and formula
@time hdmrcoefsdf = fit(HDMRCoefs, f, we8thereRatings, counts; parallel=true, verbose=true)
@test coef(hdmrcoefsdf)[1] == coefsHppos
@test coef(hdmrcoefsdf)[2] ≈ coefsHpzero
@test_throws ErrorException predict(hdmrcoefsdf,newcovars)

@time hdmrpathsdf = fit(HDMRPaths, f, we8thereRatings, counts; parallel=true, verbose=true)
@test coef(hdmrpathsdf)[1] == coefsHppos3
@test coef(hdmrpathsdf)[2] ≈ coefsHpzero3
@test predict(hdmrpathsdf,newcovars) ≈ η

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

# HIR
@time hir = fit(CIR{HDMR,LinearModel},covars,counts,1; inpos=inpos, nocounts=true)
@test coefbwd(hir)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hir)[2] ≈ coef(hdmrcoefs)[2]

@time hirglm = fit(CIR{HDMR,GeneralizedLinearModel},covars,counts,1,Poisson(); inpos=inpos, nocounts=true)
@test coefbwd(hirglm)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hirglm)[2] ≈ coef(hdmrcoefs)[2]
@test !(coeffwd(hirglm)[1] ≈ coeffwd(hir)[1])
@test !(coeffwd(hirglm)[2] ≈ coeffwd(hir)[2])

@time hirdf = fit(CIR{HDMR,LinearModel},f,we8thereRatings,counts,:Food; nocounts=true)
@test coefbwd(hirdf)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hirdf)[2] ≈ coef(hdmrcoefs)[2]
@test coeffwd(hirdf) ≈ coeffwd(hir)
@time hirglmdf = fit(CIR{HDMR,GeneralizedLinearModel},f,we8thereRatings,counts,:Food,Poisson(); nocounts=true)
@test coefbwd(hirglmdf)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hirglmdf)[2] ≈ coef(hdmrcoefs)[2]
@test !(coeffwd(hirglmdf)[2] ≈ coeffwd(hirdf)[2])
@test !(coeffwd(hirglmdf)[1] ≈ coeffwd(hirdf)[1])

# CV
srand(13)
@time cvstats13 = cv(CIR{HDMR,LinearModel},covars,counts,1; inpos=inpos, gen=MLBase.Kfold(size(covars,1),2), γ=γ)
@time cvstats13b = cv(CIR{HDMR,LinearModel},f,we8thereRatings,counts,:Food; k=2, gentype=MLBase.Kfold, γ=γ)
@test isapprox(cvstats13,cvstats13b)

@time cvstats14 = cv(CIR{HDMR,LinearModel},covars,counts,1; inpos=inpos, k=2, gentype=MLBase.Kfold, γ=γ, seed=14)
@test !(isapprox(cvstats13,cvstats14))

cvstats13glm = cv(CIR{HDMR,GeneralizedLinearModel},f,we8thereRatings,counts,:Food,Poisson(); k=2, gentype=MLBase.Kfold, γ=γ)
@test !(isapprox(cvstats13,cvstats13glm))

@time cvstatsSerialKfold = cv(CIR{HDMR,LinearModel},covars,counts,1; inpos=inpos, k=3, gentype=SerialKfold, γ=γ)
@test_throws DimensionMismatch isapprox(cvstats13,cvstatsSerialKfold)

end

########################################################################
# hurdle with covarspos ≠ covarszero, only pos model includes projdir
########################################################################
@testset "hurdle-dmr with covarspos ≠ covarszero, only pos model includes projdir" begin

f = @model(c ~ Value + Food, h ~ Value + Service + Atmosphere + Overall)
# NOTE: the @model results in a different order of variables relative to the inzero/inpos interface
# which in turn yields slightly different estimates. Probably due to coordinate descent iterating differently.
# so for the tests we align the variables to give the same results.
trmszero = HurdleDMR.getrhsterms(f, :h)
trmspos = HurdleDMR.getrhsterms(f, :c)
trms, inzero, inpos = HurdleDMR.mergerhsterms(trmszero,trmspos)

covars = convert(Matrix{Float64},we8thereRatings[:,trms.terms])
pzero = length(inzero)
ppos = length(inpos)

projdir = HurdleDMR.ixprojdir(trms, :Food)

# hurdle dmr parallel local cluster
@time hdmrcoefs = hdmr(covars, counts; inpos=inpos, inzero=inzero, parallel=true)
coefsHppos, coefsHpzero = coef(hdmrcoefs)
@test size(coefsHppos) == (ppos+1, d)
@test size(coefsHpzero) == (pzero+1, d)

@time hdmrcoefsb = fit(HDMRCoefs, covars, counts; inpos=inpos, inzero=inzero, parallel=true)
@test coef(hdmrcoefsb)[1] == coefsHppos
@test coef(hdmrcoefsb)[2] == coefsHpzero
@test n == nobs(hdmrcoefs)
@test d == ncategories(hdmrcoefs)
@test ppos == ncovarspos(hdmrcoefs)
@test pzero == ncovarszero(hdmrcoefs)

# hurdle dmr parallel remote cluster
@time hdmrcoefs2 = hdmr(covars, counts; inpos=inpos, inzero=inzero, parallel=true, local_cluster=false)
coefsHppos2, coefsHpzero2 = coef(hdmrcoefs2)
@test coefsHppos ≈ coefsHppos2
@test coefsHpzero ≈ coefsHpzero2
@time hdmrcoefs2 = fit(HDMRCoefs, covars, counts; inpos=inpos, inzero=inzero, parallel=true, local_cluster=false)
@test coef(hdmrcoefs2)[1] ≈ coefsHppos2
@test coef(hdmrcoefs2)[2] ≈ coefsHpzero2

@time hdmrpaths3 = fit(HDMRPaths, covars, counts; inpos=inpos, inzero=inzero, parallel=true, verbose=true)
coefsHppos3, coefsHpzero3 = coef(hdmrpaths3)
@test coefsHppos3 ≈ coefsHppos
@test coefsHpzero3 ≈ coefsHpzero
@test n == nobs(hdmrpaths3)
@test d == ncategories(hdmrpaths3)
@test ppos == ncovarspos(hdmrpaths3)
@test pzero == ncovarszero(hdmrpaths3)
@test size(hdmrpaths3.nlpaths,1) == 100
η = predict(hdmrpaths3,newcovars)
@test sum(η,2) ≈ ones(size(η,1))

# # hurdle dmr serial
# @time hdmrcoefs3 = hdmr(covarszero, counts; covarspos=covarspos, parallel=false)
# coefsHspos, coefsHszero = coef(hdmrcoefs3)
# @test coefsHppos ≈ coefsHspos
# @test coefsHpzero ≈ coefsHszero
# @time hdmrcoefs3 = fit(HDMRCoefs, covarszero, counts; covarspos=covarspos, parallel=false)
# @test coef(hdmrcoefs3)[1] ≈ coefsHspos
# @test coef(hdmrcoefs3)[2] ≈ coefsHszero

# using a dataframe and formula
@time hdmrcoefsdf = fit(HDMRCoefs, f, we8thereRatings, counts; parallel=true, verbose=true)
hdmrcoefsdf.model.inpos
hdmrcoefsdf.model.inzero
@test coef(hdmrcoefsdf)[1] ≈ coefsHppos
@test coef(hdmrcoefsdf)[2] ≈ coefsHpzero
@test n == nobs(hdmrcoefsdf)
@test d == ncategories(hdmrcoefsdf)
@test ppos == ncovarspos(hdmrcoefsdf)
@test pzero == ncovarszero(hdmrcoefsdf)
@test_throws ErrorException predict(hdmrcoefsdf,newcovars)

@time hdmrpathsdf = fit(HDMRPaths, f, we8thereRatings, counts; parallel=true, verbose=true)
@test coef(hdmrpathsdf)[1] ≈ coefsHppos3
@test coef(hdmrpathsdf)[2] ≈ coefsHpzero3
@test n == nobs(hdmrpathsdf)
@test d == ncategories(hdmrpathsdf)
@test ppos == ncovarspos(hdmrpathsdf)
@test pzero == ncovarszero(hdmrpathsdf)
@test predict(hdmrpathsdf,newcovars) ≈ η

zHpos = srproj(coefsHppos, counts)
@test size(zHpos) == (n,ppos+1)

zHzero = srproj(coefsHpzero, posindic(counts))
@test size(zHzero) == (n,pzero+1)

z1pos = srproj(coefsHppos, counts, 2)
@test z1pos ≈ zHpos[:,[2,ppos+1]]

# projdir is not included in zero model
# z1zero = srproj(coefsHpzero, posindic(counts), 1)
# @test z1zero ≈ zHzero[:,[1,p+1]]

Z1 = srproj(coefsHppos, coefsHpzero, counts, 2, 0; intercept=true)
@test Z1 == z1pos
@time Z1b = srproj(hdmrcoefs, counts, 2, 0; intercept=true)
@test Z1 == Z1b

Z3 = srproj(coefsHppos, coefsHpzero, counts, 1, 1; intercept=true)
z3pos = srproj(coefsHppos, counts, 1)
z3zero = srproj(coefsHpzero, posindic(counts), 1)
@test Z3 == [z3pos[:,1] z3zero[:,1] z3pos[:,2]]
@time Z3b = srproj(hdmrcoefs, counts, 1, 1; intercept=true)
@test Z3 == Z3b

regdata = DataFrame(y=covars[:,1], zv1=z1pos[:,1], m=z1pos[:,2], v1=covarspos[:,1], w1=covars[:,inzero[1]])

lmv1 = lm(@formula(y ~ zv1+m), regdata)
r2v1 = adjr2(lmv1)

@time X1, X1_nocounts, includezpos = srprojX(coefsHppos,coefsHpzero,counts,covars,projdir; inzero=inzero, inpos=inpos, includem=true)
ix = filter!(x->x!=projdir,collect(1:5))
@test X1_nocounts ≈ [ones(n) covars[:,ix]] rtol=1e-8
@test X1 ≈ [X1_nocounts Z1] rtol=1e-8
X1b, X1_nocountsb, includezposb = srprojX(hdmrcoefs,counts,covars,projdir; inzero=inzero, inpos=inpos, includem=true)
@test X1 ≈ X1b rtol=1e-8
@test X1_nocountsb ≈ X1_nocountsb rtol=1e-8
@test includezposb == includezposb

X2, X2_nocounts, includezpos = srprojX(coefsHppos,coefsHpzero,counts,covars,projdir; inzero=inzero, inpos=inpos, includem=false)
@test X2_nocounts ≈ X1_nocounts rtol=1e-8
@test X2 ≈ X1[:,1:end-1] rtol=1e-8
X2b, X2_nocountsb, includezposb = srprojX(hdmrcoefs,counts,covars,projdir; inzero=inzero, inpos=inpos, includem=false)
@test X2 ≈ X2b rtol=1e-8
@test X2_nocountsb ≈ X2_nocountsb rtol=1e-8
@test includezposb == includezposb

X3, X3_nocounts, includezpos = srprojX(coefsHppos,coefsHpzero,counts,covars,1; inzero=inzero, inpos=inpos, includem=true)
ix = filter!(x->x!=1,collect(1:5))
@test X3_nocounts ≈ [ones(n) covars[:,ix]] rtol=1e-8
@test X3 ≈ [X3_nocounts Z3] rtol=1e-8
X3b, X3_nocountsb, includezposb = srprojX(hdmrcoefs,counts,covars,1; inzero=inzero, inpos=inpos, includem=true)
@test X3 ≈ X3b rtol=1e-8
@test X3_nocountsb ≈ X3_nocountsb rtol=1e-8
@test includezposb == includezposb

# HIR
@time hir = fit(CIR{HDMR,LinearModel},covars,counts,projdir; inzero=inzero, inpos=inpos, nocounts=true)
@test coefbwd(hir)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hir)[2] ≈ coef(hdmrcoefs)[2]

@time hirglm = fit(CIR{HDMR,GeneralizedLinearModel},covars,counts,projdir,Poisson(); inzero=inzero, inpos=inpos, nocounts=true)
@test coefbwd(hirglm)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hirglm)[2] ≈ coef(hdmrcoefs)[2]
@test !(coeffwd(hirglm)[1] ≈ coeffwd(hir)[1])
@test !(coeffwd(hirglm)[2] ≈ coeffwd(hir)[2])

@time hirdf = fit(CIR{HDMR,LinearModel},f,we8thereRatings,counts,:Food; nocounts=true)
@test coefbwd(hirdf)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hirdf)[2] ≈ coef(hdmrcoefs)[2]
@test coeffwd(hirdf) ≈ coeffwd(hir)
@time hirglmdf = fit(CIR{HDMR,GeneralizedLinearModel},f,we8thereRatings,counts,:Food,Poisson(); nocounts=true)
@test coefbwd(hirglmdf)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hirglmdf)[2] ≈ coef(hdmrcoefs)[2]
@test !(coeffwd(hirglmdf)[2] ≈ coeffwd(hirdf)[2])
@test !(coeffwd(hirglmdf)[1] ≈ coeffwd(hirdf)[1])

# CV
srand(13)
@time cvstats13 = cv(CIR{HDMR,LinearModel},covars,counts,projdir; inzero=inzero, inpos=inpos, gen=MLBase.Kfold(size(covars,1),2), γ=γ)
@time cvstats13b = cv(CIR{HDMR,LinearModel},f,we8thereRatings,counts,:Food; k=2, gentype=MLBase.Kfold, γ=γ)
@test cvstats13 ≈ cvstats13b

@time cvstats14 = cv(CIR{HDMR,LinearModel},covars,counts,projdir; inzero=inzero, inpos=inpos, k=2, gentype=MLBase.Kfold, γ=γ, seed=14)
@test !(isequal(cvstats13,cvstats14))

@time cvstatsSerialKfold = cv(CIR{HDMR,LinearModel},covars,counts,projdir; inzero=inzero, inpos=inpos, k=3, gentype=SerialKfold, γ=γ)
@test_throws DimensionMismatch isapprox(cvstats13,cvstatsSerialKfold)

end

@testset "degenerate cases" begin

# column j is never zero, so hj=1 for all observations
zcounts = deepcopy(counts)
zcounts[:,2] = zeros(n)
zcounts[:,3] = ones(n)

# make sure we are not adding all zero obseravtions
m = sum(zcounts,2)
@test sum(m .== 0) == 0

# hurdle dmr parallel local cluster
@time hdmrcoefs = fit(HDMR,covars, zcounts; parallel=true, showwarnings=true, verbose=false)
coefsHppos, coefsHpzero = coef(hdmrcoefs)
@test size(coefsHppos) == (p+1, d)
@test size(coefsHpzero) == (p+1, d)
@test coefsHppos[:,2] == zeros(p+1)
@test coefsHpzero[:,2] == zeros(p+1)
@test coefsHppos[:,3] == zeros(p+1)
@test coefsHpzero[:,3] == zeros(p+1)

# hurdle dmr parallel remote cluster
@time hdmrcoefs2 = fit(HDMRPaths,covars, zcounts; parallel=true, local_cluster=false)
coefsHppos2, coefsHpzero2 = coef(hdmrcoefs)
@test coefsHppos ≈ coefsHppos2
@test coefsHpzero ≈ coefsHpzero
η = predict(hdmrcoefs2,newcovars)
@test sum(η,2) ≈ ones(size(newcovars,1))
@test η[:,2] == zeros(size(newcovars,1))
@test η[:,3] ≈ ones(size(newcovars,1))*0.745 atol=0.001

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
@time hdmrcoefs = fit(HDMR,covars, zcounts; parallel=true)
coefsHppos, coefsHpzero = coef(hdmrcoefs)
@test size(coefsHppos) == (p+1, d)
@test size(coefsHpzero) == (p+1, d)
@test coefsHppos[:,2] != zeros(p+1)
@test coefsHpzero[:,2] == zeros(p+1)
@test coefsHppos[:,3] != zeros(p+1)
@test coefsHpzero[:,3] == zeros(p+1)

# hurdle dmr parallel remote cluster
@time hdmrcoefs2 = fit(HDMRPaths,covars, zcounts; parallel=true, local_cluster=false)
coefsHppos2, coefsHpzero2 = coef(hdmrcoefs2)
@test coefsHppos ≈ coefsHppos2
@test coefsHpzero ≈ coefsHpzero2

# just checking the fit(HDMRPaths...) ignore local_cluster
@time hdmrcoefs3 = fit(HDMRPaths,covars, zcounts; parallel=true, local_cluster=true)
coefsHppos3, coefsHpzero3 = coef(hdmrcoefs3)
@test coefsHppos2 == coefsHppos3
@test coefsHpzero2 == coefsHpzero3

end
