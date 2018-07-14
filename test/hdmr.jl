n = 100
p = 3
d = 4

srand(13)
m = 1+rand(Poisson(5),n)
covars = rand(n,p)
ηfn(vi) = exp.([0 + i*sum(vi) for i=1:d])
q = [ηfn(covars[i,:]) for i=1:n]
scale!.(q,ones(n)./sum.(q))
@assert sum.(q) ≈ ones(n)
c = broadcast((qi,mi)->rand(Multinomial(mi, qi)),q,m)
@assert sum.(c) == m
counts = convert(SparseMatrixCSC{Float64,Int},hcat(c...)')

# fit(GeneralizedLinearModel,covars,c,Multinomial(4,2))
newcovars = covars[1:10,:]

covarsdf = DataFrame(covars,[:v1, :v2, :vy])

# γ = 1.0

###########################################################
# hurdle with covarspos == covarszero
###########################################################
@testset "hurdle-dmr with covarspos == covarszero" begin

f = @model(h ~ v1 + v2 + vy, c ~ v1 + v2 + vy)
projdir = findfirst(names(covarsdf),:vy)
dirpos = 3
dirzero = 3

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
@test size(hdmrpaths3.nlpaths,1) == d
η = predict(hdmrpaths3,newcovars)
@test sum(η,2) ≈ ones(size(η,1))

# # # hurdle dmr serial
@time hdmrcoefs3 = hdmr(covars, counts; parallel=false)
coefsHspos, coefsHszero = coef(hdmrcoefs3)
@test coefsHppos ≈ coefsHspos
@test coefsHpzero ≈ coefsHszero
@time hdmrcoefs3 = fit(HDMRCoefs, covars, counts; parallel=false)
@test coef(hdmrcoefs3)[1] ≈ coefsHspos
@test coef(hdmrcoefs3)[2] ≈ coefsHszero

# using a dataframe and formula
@time hdmrcoefsdf = fit(HDMRCoefs, f, covarsdf, counts; parallel=true, verbose=true)
@test coef(hdmrcoefsdf)[1] == coefsHppos
@test coef(hdmrcoefsdf)[2] ≈ coefsHpzero
@test_throws ErrorException predict(hdmrcoefsdf,newcovars)

@time hdmrpathsdf = fit(HDMRPaths, f, covarsdf, counts; parallel=true, verbose=true)
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

z1pos = srproj(coefsHppos, counts, projdir)
@test z1pos ≈ zHpos[:,[projdir,p+1]]

# second half of coefs belongs to covarszero
z1zero = srproj(coefsHpzero, posindic(counts), projdir)
@test z1zero ≈ zHzero[:,[projdir,p+1]]

@time Z1 = srproj(coefsHppos, coefsHpzero, counts, projdir, projdir; intercept=true)
@test Z1 == [z1pos[:,1] z1zero[:,1] z1pos[:,2]]
@time Z1b = srproj(hdmrcoefs, counts, projdir, projdir; intercept=true)
@test Z1 == Z1b

Z3 = srproj(coefsHppos, coefsHpzero, counts, projdir, projdir; intercept=true)
z3pos = srproj(coefsHppos, counts, projdir)
z3zero = srproj(coefsHpzero, posindic(counts), projdir)
@test Z3 == [z3pos[:,1] z3zero[:,1] z3pos[:,2]]
@time Z3b = srproj(hdmrcoefs, counts, projdir, projdir; intercept=true)
@test Z3 == Z3b

regdata = DataFrame(y=covars[:,1], zw1=z1zero[:,1], zv1=z1pos[:,1], m=z1pos[:,2])
lmw1 = lm(@formula(y ~ zw1+m), regdata)
r2w1 = adjr2(lmw1)

lmv1 = lm(@formula(y ~ zv1+m), regdata)
r2v1 = adjr2(lmv1)

lmw1v1 = lm(@formula(y ~ zw1+zv1+m), regdata)
r2w1v1 = adjr2(lmw1v1)

X1, X1_nocounts, includezpos = srprojX(coefsHppos,coefsHpzero,counts,covars,projdir; includem=true)
@test X1_nocounts == [ones(n) covars[:,1:2]]
@test X1 == [X1_nocounts Z1]
X1b, X1_nocountsb, includezposb = srprojX(hdmrcoefs,counts,covars,projdir; includem=true)
@test X1 == X1b
@test X1_nocountsb == X1_nocountsb
@test includezposb == includezposb

X2, X2_nocounts, includezpos = srprojX(coefsHppos,coefsHpzero,counts,covars,projdir; includem=false)
@test X2_nocounts == X1_nocounts
@test X2 == X1[:,1:end-1]
X2b, X2_nocountsb, includezposb = srprojX(hdmrcoefs,counts,covars,projdir; includem=false)
@test X2 == X2b
@test X2_nocountsb == X2_nocountsb
@test includezposb == includezposb

X3, X3_nocounts, includezpos = srprojX(coefsHppos,coefsHpzero,counts,covars,projdir; includem=true)
@test X3_nocounts == [ones(n) covars[:,setdiff(1:p,[projdir])]]
@test X3 == [X3_nocounts Z3]
X3b, X3_nocountsb, includezposb = srprojX(hdmrcoefs,counts,covars,projdir; includem=true)
@test X3 == X3b
@test X3_nocountsb == X3_nocountsb
@test includezposb == includezposb

# HIR
@time hir = fit(CIR{HDMR,LinearModel},covars,counts,projdir; nocounts=true)
@test coefbwd(hir)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hir)[2] ≈ coef(hdmrcoefs)[2]

@time hirglm = fit(CIR{HDMR,GeneralizedLinearModel},covars,counts,projdir,Gamma(); nocounts=true)
@test coefbwd(hirglm)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hirglm)[2] ≈ coef(hdmrcoefs)[2]
@test !(coeffwd(hirglm)[1] ≈ coeffwd(hir)[1])
@test !(coeffwd(hirglm)[2] ≈ coeffwd(hir)[2])

@time hirdf = fit(CIR{HDMR,LinearModel},f,covarsdf,counts,:vy; nocounts=true)
@test coefbwd(hirdf)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hirdf)[2] ≈ coef(hdmrcoefs)[2]
@test coeffwd(hirdf) ≈ coeffwd(hir)
@time hirglmdf = fit(CIR{HDMR,GeneralizedLinearModel},f,covarsdf,counts,:vy,Gamma(); nocounts=true)
@test coefbwd(hirglmdf)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hirglmdf)[2] ≈ coef(hdmrcoefs)[2]
@test !(coeffwd(hirglmdf)[2] ≈ coeffwd(hirdf)[2])
@test !(coeffwd(hirglmdf)[1] ≈ coeffwd(hirdf)[1])

zlm = lm(hcat(ones(n,1),Z1,covars[:,1:2]),covars[:,projdir])
@test r2(zlm) ≈ r2(hir)
@test adjr2(zlm) ≈ adjr2(hir)
predict(zlm,hcat(ones(10,1),Z1[1:10,:],covars[1:10,1:2]))
predict(hir,covars[1:10,:],counts[1:10,:])
@test predict(zlm,hcat(ones(10,1),Z1[1:10,:],covars[1:10,1:2])) ≈ predict(hir,covars[1:10,:],counts[1:10,:])

zlmnocounts = lm(hcat(ones(n,1),covars[:,1:2]),covars[:,projdir])
@test r2(zlmnocounts) ≈ r2(hir; nocounts=true)
@test adjr2(zlmnocounts) ≈ adjr2(hir; nocounts=true)
@test predict(zlmnocounts,hcat(ones(10,1),covars[1:10,1:2])) ≈ predict(hir,covars[1:10,:],counts[1:10,:]; nocounts=true)

# CV
srand(13)
@time cvstats13 = cv(CIR{HDMR,LinearModel},covars,counts,projdir; gen=MLBase.Kfold(size(covars,1),2), γ=γ)
@time cvstats13b = cv(CIR{HDMR,LinearModel},f,covarsdf,counts,:vy; k=2, gentype=MLBase.Kfold, γ=γ)
@test isapprox(cvstats13,cvstats13b)

@time cvstats15 = cv(CIR{HDMR,LinearModel},covars,counts,projdir; k=2, gentype=MLBase.Kfold, γ=γ, seed=15)
@test !(isapprox(cvstats13,cvstats15))

cvstats13glm = cv(CIR{HDMR,GeneralizedLinearModel},f,covarsdf,counts,:vy,Gamma(); k=2, gentype=MLBase.Kfold, γ=γ)
@test !(isapprox(cvstats13,cvstats13glm))

@time cvstatsSerialKfold = cv(CIR{HDMR,LinearModel},covars,counts,projdir; k=3, gentype=SerialKfold, γ=γ)
@test_throws DimensionMismatch isapprox(cvstats13,cvstatsSerialKfold)

end

####################################################################
# hurdle with covarspos ≠ covarszero, both models includes projdir
####################################################################
@testset "hurdle-dmr with covarspos ≠ covarszero, both models includes projdir" begin

f = @model(h ~ v1 + v2 + vy, c ~ v2 + vy)
inzero = 1:p
inpos = 2:p
ppos = p-1
projdir = findfirst(names(covarsdf),:vy)
dirpos = 2
dirzero = 3

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
@test size(hdmrpaths3.nlpaths,1) == d
η = predict(hdmrpaths3,newcovars)
@test sum(η,2) ≈ ones(size(η,1))

# # hurdle dmr serial
@time hdmrcoefs3 = hdmr(covars, counts; inpos=inpos, parallel=false)
coefsHspos, coefsHszero = coef(hdmrcoefs3)
@test coefsHppos ≈ coefsHspos
@test coefsHpzero ≈ coefsHszero
@time hdmrcoefs3 = fit(HDMRCoefs, covars, counts; inpos=inpos, parallel=false)
@test coef(hdmrcoefs3)[1] ≈ coefsHspos
@test coef(hdmrcoefs3)[2] ≈ coefsHszero

# using a dataframe and formula
@time hdmrcoefsdf = fit(HDMRCoefs, f, covarsdf, counts; parallel=true, verbose=true)
@test coef(hdmrcoefsdf)[1] == coefsHppos
@test coef(hdmrcoefsdf)[2] ≈ coefsHpzero
@test_throws ErrorException predict(hdmrcoefsdf,newcovars)

@time hdmrpathsdf = fit(HDMRPaths, f, covarsdf, counts; parallel=true, verbose=true)
@test coef(hdmrpathsdf)[1] == coefsHppos3
@test coef(hdmrpathsdf)[2] ≈ coefsHpzero3
@test predict(hdmrpathsdf,newcovars) ≈ η

zHpos = srproj(coefsHppos, counts)
@test size(zHpos) == (n,ppos+1)

zHzero = srproj(coefsHpzero, posindic(counts))
@test size(zHzero) == (n,p+1)

# first half of coefs belongs to covarspos
z1pos = srproj(coefsHppos, counts, dirpos)
@test z1pos ≈ zHpos[:,[dirpos,ppos+1]]

# second half of coefs belongs to covarszero
z1zero = srproj(coefsHpzero, posindic(counts), dirzero)
@test z1zero ≈ zHzero[:,[dirzero,p+1]]

Z1 = srproj(coefsHppos, coefsHpzero, counts, dirpos, dirzero; intercept=true)
@test Z1 == [z1pos[:,1] z1zero[:,1] z1pos[:,2]]
@time Z1b = srproj(hdmrcoefs, counts, dirpos, dirzero; intercept=true)
@test Z1 == Z1b
#
# Z3 = srproj(coefsHppos, coefsHpzero, counts, 2, 3; intercept=true)
# z3pos = srproj(coefsHppos, counts, 2)
# z3zero = srproj(coefsHpzero, posindic(counts), 3)
# @test Z3 == [z3pos[:,1] z3zero[:,1] z3pos[:,2]]
# @time Z3b = srproj(hdmrcoefs, counts, 2, 3; intercept=true)
# @test Z3 == Z3b

regdata = DataFrame(y=covars[:,1], zw1=z1zero[:,1], zv1=z1pos[:,1], m=z1pos[:,2], v1=covars[:,last(inpos)], w1=covars[:,last(inzero)])
lmw1 = lm(@formula(y ~ zw1+m), regdata)
r2w1 = adjr2(lmw1)

lmv1 = lm(@formula(y ~ zv1+m), regdata)
r2v1 = adjr2(lmv1)

lmw1v1 = lm(@formula(y ~ zw1+zv1+m), regdata)
r2w1v1 = adjr2(lmw1v1)

@time X1, X1_nocounts, includezpos = srprojX(coefsHppos,coefsHpzero,counts,covars,projdir; inpos=inpos, includem=true)
@test X1_nocounts == [ones(n) covars[:,1:2]]
@test X1 == [X1_nocounts Z1]
X1b, X1_nocountsb, includezposb = srprojX(hdmrcoefs,counts,covars,projdir; inpos=inpos, includem=true)
@test X1 == X1b
@test X1_nocountsb == X1_nocountsb
@test includezposb == includezposb

X2, X2_nocounts, includezpos = srprojX(coefsHppos,coefsHpzero,counts,covars,projdir; inpos=inpos, includem=false)
@test X2_nocounts == X1_nocounts
@test X2 == X1[:,1:end-1]
X2b, X2_nocountsb, includezposb = srprojX(hdmrcoefs,counts,covars,projdir; inpos=inpos, includem=false)
@test X2 == X2b
@test X2_nocountsb == X2_nocountsb
@test includezposb == includezposb

# X3, X3_nocounts, includezpos = srprojX(coefsHppos,coefsHpzero,counts,covars,3; inpos=inpos, includem=true)
# @test X3_nocounts == [ones(n) covars[:,[1,2,4,5]]]
# @test X3 == [X3_nocounts Z3]
# X3b, X3_nocountsb, includezposb = srprojX(hdmrcoefs,counts,covars,3; inpos=inpos, includem=true)
# @test X3 == X3b
# @test X3_nocountsb == X3_nocountsb
# @test includezposb == includezposb

# HIR
@time hir = fit(CIR{HDMR,LinearModel},covars,counts,projdir; inpos=inpos, nocounts=true)
@test coefbwd(hir)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hir)[2] ≈ coef(hdmrcoefs)[2]

@time hirglm = fit(CIR{HDMR,GeneralizedLinearModel},covars,counts,projdir,Gamma(); inpos=inpos, nocounts=true)
@test coefbwd(hirglm)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hirglm)[2] ≈ coef(hdmrcoefs)[2]
@test !(coeffwd(hirglm)[1] ≈ coeffwd(hir)[1])
@test !(coeffwd(hirglm)[2] ≈ coeffwd(hir)[2])

@time hirdf = fit(CIR{HDMR,LinearModel},f,covarsdf,counts,:vy; nocounts=true)
@test coefbwd(hirdf)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hirdf)[2] ≈ coef(hdmrcoefs)[2]
@test coeffwd(hirdf) ≈ coeffwd(hir)
@time hirglmdf = fit(CIR{HDMR,GeneralizedLinearModel},f,covarsdf,counts,:vy,Gamma(); nocounts=true)
@test coefbwd(hirglmdf)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hirglmdf)[2] ≈ coef(hdmrcoefs)[2]
@test !(coeffwd(hirglmdf)[2] ≈ coeffwd(hirdf)[2])
@test !(coeffwd(hirglmdf)[1] ≈ coeffwd(hirdf)[1])

# CV
srand(13)
@time cvstats13 = cv(CIR{HDMR,LinearModel},covars,counts,projdir; inpos=inpos, gen=MLBase.Kfold(size(covars,1),2), γ=γ)
@time cvstats13b = cv(CIR{HDMR,LinearModel},f,covarsdf,counts,:vy; k=2, gentype=MLBase.Kfold, γ=γ)
@test isapprox(cvstats13,cvstats13b)

@time cvstats15 = cv(CIR{HDMR,LinearModel},covars,counts,projdir; inpos=inpos, k=2, gentype=MLBase.Kfold, γ=γ, seed=15)
@test !(isapprox(cvstats13,cvstats15))

cvstats13glm = cv(CIR{HDMR,GeneralizedLinearModel},f,covarsdf,counts,:vy,Gamma(); k=2, gentype=MLBase.Kfold, γ=γ)
@test !(isapprox(cvstats13,cvstats13glm))

@time cvstatsSerialKfold = cv(CIR{HDMR,LinearModel},covars,counts,projdir; inpos=inpos, k=3, gentype=SerialKfold, γ=γ)
@test_throws DimensionMismatch isapprox(cvstats13,cvstatsSerialKfold)

end

########################################################################
# hurdle with covarspos ≠ covarszero, only pos model includes projdir
########################################################################
@testset "hurdle-dmr with covarspos ≠ covarszero, only pos model includes projdir" begin

# f = @model(c ~ Value + Food, h ~ Value + Service + Atmosphere + Overall)
f = @model(h ~ v1 + v2, c ~ v2 + vy)
inzero = 1:2
inpos = 2:3
projdir = findfirst(names(covarsdf),:vy)

# NOTE: the @model results in a different order of variables relative to the inzero/inpos interface
# which in turn yields slightly different estimates. Probably due to coordinate descent iterating differently.
# so for the tests we align the variables to give the same results.
# trmszero = HurdleDMR.getrhsterms(f, :h)
# trmspos = HurdleDMR.getrhsterms(f, :c)
# trms, inzero, inpos = HurdleDMR.mergerhsterms(trmszero,trmspos)
#
# covars = convert(Matrix{Float64},covarsdf[:,trms.terms])
pzero = length(inzero)
ppos = length(inpos)
# trms
# projdir = HurdleDMR.ixprojdir(trms, :vy)
dirpos = 2
dirzero = 0

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
@test size(hdmrpaths3.nlpaths,1) == d
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
@time hdmrcoefsdf = fit(HDMRCoefs, f, covarsdf, counts; parallel=true, verbose=true)
@test coef(hdmrcoefsdf)[1] ≈ coefsHppos
@test coef(hdmrcoefsdf)[2] ≈ coefsHpzero
@test n == nobs(hdmrcoefsdf)
@test d == ncategories(hdmrcoefsdf)
@test ppos == ncovarspos(hdmrcoefsdf)
@test pzero == ncovarszero(hdmrcoefsdf)
@test_throws ErrorException predict(hdmrcoefsdf,newcovars)

@time hdmrpathsdf = fit(HDMRPaths, f, covarsdf, counts; parallel=true, verbose=true)
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

z1pos = srproj(coefsHppos, counts, dirpos)
@test z1pos ≈ zHpos[:,[dirpos,ppos+1]]

# projdir is not included in zero model
# z1zero = srproj(coefsHpzero, posindic(counts), 1)
# @test z1zero ≈ zHzero[:,[1,p+1]]

Z1 = srproj(coefsHppos, coefsHpzero, counts, dirpos, dirzero; intercept=true)
@test Z1 == z1pos
@time Z1b = srproj(hdmrcoefs, counts, dirpos, dirzero; intercept=true)
@test Z1 == Z1b

# Z3 = srproj(coefsHppos, coefsHpzero, counts, 1, 1; intercept=true)
# z3pos = srproj(coefsHppos, counts, 1)
# z3zero = srproj(coefsHpzero, posindic(counts), 1)
# @test Z3 == [z3pos[:,1] z3zero[:,1] z3pos[:,2]]
# @time Z3b = srproj(hdmrcoefs, counts, 1, 1; intercept=true)
# @test Z3 == Z3b

regdata = DataFrame(y=covars[:,1], zv1=z1pos[:,1], m=z1pos[:,2], v1=covars[:,last(inpos)], w1=covars[:,last(inzero)])

lmv1 = lm(@formula(y ~ zv1+m), regdata)
r2v1 = adjr2(lmv1)

@time X1, X1_nocounts, includezpos = srprojX(coefsHppos,coefsHpzero,counts,covars,projdir; inzero=inzero, inpos=inpos, includem=true)
ix = filter!(x->x!=projdir,collect(1:p))
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

# X3, X3_nocounts, includezpos = srprojX(coefsHppos,coefsHpzero,counts,covars,projdir; inzero=inzero, inpos=inpos, includem=true)
# ix = filter!(x->x!=projdir,collect(1:p))
# @test X3_nocounts ≈ [ones(n) covars[:,ix]] rtol=1e-8
# @test X3 ≈ [X3_nocounts Z3] rtol=1e-8
# X3b, X3_nocountsb, includezposb = srprojX(hdmrcoefs,counts,covars,1; inzero=inzero, inpos=inpos, includem=true)
# @test X3 ≈ X3b rtol=1e-8
# @test X3_nocountsb ≈ X3_nocountsb rtol=1e-8
# @test includezposb == includezposb

# HIR
@time hir = fit(CIR{HDMR,LinearModel},covars,counts,projdir; inzero=inzero, inpos=inpos, nocounts=true)
@test coefbwd(hir)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hir)[2] ≈ coef(hdmrcoefs)[2]

@time hirglm = fit(CIR{HDMR,GeneralizedLinearModel},covars,counts,projdir,Gamma(); inzero=inzero, inpos=inpos, nocounts=true)
@test coefbwd(hirglm)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hirglm)[2] ≈ coef(hdmrcoefs)[2]
@test !(coeffwd(hirglm)[1] ≈ coeffwd(hir)[1])
@test !(coeffwd(hirglm)[2] ≈ coeffwd(hir)[2])

@time hirdf = fit(CIR{HDMR,LinearModel},f,covarsdf,counts,:vy; nocounts=true)
@test coefbwd(hirdf)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hirdf)[2] ≈ coef(hdmrcoefs)[2]
@test coeffwd(hirdf) ≈ coeffwd(hir)
@time hirglmdf = fit(CIR{HDMR,GeneralizedLinearModel},f,covarsdf,counts,:vy,Gamma(); nocounts=true)
@test coefbwd(hirglmdf)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hirglmdf)[2] ≈ coef(hdmrcoefs)[2]
@test !(coeffwd(hirglmdf)[2] ≈ coeffwd(hirdf)[2])
@test !(coeffwd(hirglmdf)[1] ≈ coeffwd(hirdf)[1])

# CV
srand(13)
@time cvstats13 = cv(CIR{HDMR,LinearModel},covars,counts,projdir; inzero=inzero, inpos=inpos, gen=MLBase.Kfold(size(covars,1),2), γ=γ)
@time cvstats13b = cv(CIR{HDMR,LinearModel},f,covarsdf,counts,:vy; k=2, gentype=MLBase.Kfold, γ=γ)
@test cvstats13 ≈ cvstats13b

@time cvstats15 = cv(CIR{HDMR,LinearModel},covars,counts,projdir; inzero=inzero, inpos=inpos, k=2, gentype=MLBase.Kfold, γ=γ, seed=15)
@test !(isequal(cvstats13,cvstats15))

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
@test η[:,3] ≈ ones(size(newcovars,1))*0.36 atol=0.05
@test η[:,4] ≈ ones(size(newcovars,1))*0.6 atol=0.1

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
