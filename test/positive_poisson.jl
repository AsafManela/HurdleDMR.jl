@testset "PositivePoisson" begin
using GLM: linkinv, linkfun, mueta, inverselink

λ0=3.4
pp = PositivePoisson(λ0)
@test params(pp) == (λ0,)
@test rate(pp) == λ0
@test mean(pp) == λ0 / (1-exp(-λ0))

xs = 1:10
@test pdf.(pp,xs) ≈ exp.(logpdf.(pp,xs)) atol=1e-4

xs = 1:100:10000
lp1=[logpdf(Poisson(λ0),x)::Float64 for x=xs]
lp2=[HurdleDMR.logpdf_approx(Poisson(λ0),x)::Float64 for x=xs]
@test lp1 ≈ lp2

lp1=[logpdf(PositivePoisson(λ0),x)::Float64 for x=xs]
lp2=[logpdf_exact(PositivePoisson(λ0),x)::Float64 for x=xs]
@test lp1 ≈ lp2

le1 = HurdleDMR.logexpm1.(xs)
le1big = HurdleDMR.logexpm1.(big.(xs))
le1big2 = broadcast(x->log(exp(x)-one(x)),big.(xs))
@test le1 ≈ le1big
@test le1 ≈ le1big2

ηs=-10:1.0:10
μs=broadcast(η->linkinv(LogProductLogLink(),η),ηs)
ηscheck=broadcast(μ->linkfun(LogProductLogLink(),μ),μs)
@test ηs ≈ ηscheck

μs=broadcast(η->mueta(LogProductLogLink(),η),ηs)
μscheck=broadcast(η->inverselink(LogProductLogLink(),η)[2],ηs)
@test μs ≈ μscheck

loglik(y, μ) = GLM.loglik_obs(PositivePoisson(λ0), y, μ, one(y), 0)

μs=1.01:10.0:1000.0
ηs=broadcast(μ->linkfun(LogProductLogLink(),μ),μs)
μscheck=broadcast(η->linkinv(LogProductLogLink(),η),ηs)
@test μs ≈ μscheck
#verify works for large μ
ys = round.(Int,μs) .+ 1.0
devresids = devresid.(PositivePoisson(λ0), ys, μs)
@test all(isfinite,devresids)
logliks = loglik.(ys, μs)
@test all(isfinite,logliks)

@test isinf(devresid(PositivePoisson(λ0), 3.0, 0.1))
@test iszero(devresid(PositivePoisson(λ0), 0.3, 0.1))

ys = fill(1.0, size(μs))
devresids1 = devresid.(PositivePoisson(λ0), ys, μs)
@test all(isfinite,devresids1)
logliks1 = loglik.(ys, μs)
@test all(isfinite,logliks1)

μsbig = big.(μs)
ηs=broadcast(μ->linkfun(LogProductLogLink(),μ),μsbig)
μscheck=broadcast(η->linkinv(LogProductLogLink(),η),ηs)
@test μsbig ≈ μscheck
#verify works for large μ
ysbig = round.(BigInt,μsbig) .+ 1.0
devresidsbig = devresid.(PositivePoisson(λ0), ysbig, μsbig)
@test all(isfinite,devresidsbig)
@test devresids ≈ Float64.(devresidsbig)

logliksbig = loglik.(ysbig, μsbig)
@test all(isfinite,logliksbig)
@test logliks ≈ Float64.(logliksbig)

ysbig = fill(big"1.0", size(μsbig))
devresidsbig1 = devresid.(PositivePoisson(λ0), ysbig, μsbig)
@test all(isfinite,devresidsbig1)
@test devresids1 ≈ Float64.(devresidsbig1)
logliksbig1 = loglik.(ysbig, μsbig)
@test all(isfinite,logliksbig1)

# μs close to 1.0
μs=big.(collect(1.0+1e-10:1e-10:1.0+1000*1e-10))
ηs=broadcast(μ->linkfun(LogProductLogLink(),μ),μs)
μscheck=broadcast(η->linkinv(LogProductLogLink(),η),ηs)
@test μs ≈ μscheck
#verify works for large μ
ys = round.(BigInt,μs) .+ big"1.0"
devresidsbig = devresid.(PositivePoisson(λ0), ys, μs)
@test all(isfinite,devresidsbig)
logliksbig = loglik.(ys, μs)
@test all(isfinite,logliksbig)

ys = fill(big"1.0", size(μs))
devresidsbig = devresid.(PositivePoisson(λ0), ys, μs)
@test all(isfinite,devresidsbig)
logliks1 = loglik.(ys, μs)
@test all(isfinite,logliks1)

# R benchmark
seed=12
nn=1000
b0=1.0
b1=-2.0
coefs0=[b0,b1]

# # uncomment code to generate R benchmark and save it to csv files
# using RCall
# R"if(!require(VGAM)){install.packages(\"VGAM\");library(VGAM)}"
# R"library(VGAM)"
# R"set.seed($seed)"
# R"pdata <- data.frame(x2 = runif(nn <- $nn))"
# R"pdata <- transform(pdata, lambda = exp($b0 + $b1 * x2))"
# R"pdata <- transform(pdata, y1 = rpospois(nn, lambda))"
# R"with(pdata, table(y1))"
# R"fit <- vglm(y1 ~ x2, pospoisson, data = pdata, trace = TRUE, crit = \"coef\")"
# print(R"summary(fit)")
# pdata=rcopy("pdata")
# coefsR=vec(rcopy(R"coef(fit, matrix = TRUE)"))
# coefsRdf = DataFrame(intercept=[coefsR[1]],x2=[coefsR[2]])
# writetable(joinpath(testdir,"data","positive_poisson_pdata.csv"),pdata)
# writetable(joinpath(testdir,"data","positive_poisson_coefsR.csv"),coefsRdf)

# load saved R benchmark
import CSV
pdata=CSV.read(joinpath(testdir,"data","positive_poisson_pdata.csv"), DataFrame)
coefsR=vec(Matrix{Float64}(CSV.read(joinpath(testdir,"data","positive_poisson_coefsR.csv"), DataFrame)))

X=Matrix{Float64}(pdata[!,[:x2]])
Xwconst=[ones(size(X,1)) X]
y=convert(Array{Float64,1},pdata.y1)

glmfit = fit(GeneralizedLinearModel,Xwconst,y,PositivePoisson(),LogProductLogLink())

# glmfit = fit(GeneralizedLinearModel,Xwconst,y,PositivePoisson(),LogProductLogLink();convTol=1e-2)
coefsGLM = coef(glmfit)
@test coefsGLM ≈ coefsR rtol=1e-7
@test coefsGLM ≈ coefs0 rtol=0.05

# GammaLassoPath without actualy regularization
glpfit = fit(GammaLassoPath,X,y,PositivePoisson(),LogProductLogLink(); λ=[0.0])
coefsGLP = vec(coef(glpfit))
@test coefsGLP ≈ coefsGLM

# GammaLassoPath doing Lasso
lassofit = fit(GammaLassoPath,X,y,PositivePoisson(),LogProductLogLink(); γ=0.0)
coefsLasso = vec(coef(lassofit;select=MinAICc()))
@test coefsLasso ≈ coefsGLM rtol=0.02
@test coefsLasso ≈ coefs0 rtol=0.05

# GammaLassoPath doing concave regularization
glpfit = fit(GammaLassoPath,X,y,PositivePoisson(),LogProductLogLink(); γ=10.0)
coefsGLP = vec(coef(glpfit;select=MinAICc()))
@test coefsGLP ≈ coefsLasso rtol=0.0002
@test coefsGLP ≈ coefs0 rtol=0.05

# # problematic case 1
# X = [6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 6.693264063357201 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 5.354611250685761 4.015958438014321 4.015958438014321 4.015958438014321 4.015958438014321 4.015958438014321 4.015958438014321 4.015958438014321 4.015958438014321 4.015958438014321 2.6773056253428806 2.6773056253428806 2.6773056253428806 2.6773056253428806 1.3386528126714403 1.3386528126714403 1.3386528126714403 1.3386528126714403]'
# y = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
# yalt = [3.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
# y = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
# y=ones(size(X,1))
# y[end]=1
# problem1fit = fit(LassoPath,X,y,Binomial();verbose=true,naivealgorithm=true)
# problem1fit = fit(GeneralizedLinearModel,X,y,Poisson();verbose=true)
#
# problem1fit = fit(GammaLassoPath,X,y,Poisson();γ=0.0,verbose=true,naivealgorithm=false)
# problem1fit.m.pp
# m=[]
# problem1fit=[]
# (obj,objold,path,f,newcoef,oldcoef,b0,oldb0,b0diff,coefdiff,scratchmu,cd,r,α,curλ)=(0,0,[],0,[],[],0,0,0,[],[],[],[],0,0)
# try
#   problem1fit = fit(GammaLassoPath,X,y,PositivePoisson();γ=0.0,verbose=true,naivealgorithm=false)
# catch e
#   if typeof(e) == Lasso.ConvergenceException
#     (obj,objold,path,f,newcoef,oldcoef,b0,oldb0,b0diff,coefdiff,scratchmu,cd,r,α,curλ) = e.debugvars
#     m = path.m
#     error(e.msg)
#     nothing
#   else
#     e
#   end
# end
# coef(problem1fit)
# newcoef==oldcoef
# b0diff
# coefdiff
# newcoef
# oldcoef
# newcoef-oldcoef
# scratchmu
# m.rr.devresid
# sum(m.rr.devresid)
# println(m.rr.var)
# cd
# m.rr.mu
# m.pp.oldy
# deviance(m)
# fieldnames(path.m.pp)
# curλ
# b0=-3.67186085318965
# newcoef.coef[1]=-0.08566741748972534
#
# rnew = deepcopy(m.rr)
# pnew = deepcopy(m.pp)
# newmu = Lasso.linpred!(scratchmu, pnew, newcoef, b0)
# updateμ!(rnew, newmu)
# devnew = deviance(rnew)
# curλ*Lasso.P(α, newcoef, pnew.ω)
# objnew = devnew/2 + curλ*Lasso.P(α, newcoef, pnew.ω)
#
# rold = deepcopy(m.rr)
# pold = deepcopy(m.pp)
# oldmu = Lasso.linpred!(scratchmu, pold, oldcoef, oldb0)
# updateμ!(rold, oldmu)
# devold = deviance(rold)
# objold = devold/2 + curλ*Lasso.P(α, oldcoef, pold.ω)
#
# devnew==devold
# curλ*Lasso.P(α, oldcoef, pold.ω) > curλ*Lasso.P(α, newcoef, pnew.ω)
# objnew > objold
# # step-halving failure zoomin
# function devf(b0,newcoef,m,cd,f)
#   T = eltype(y)
#   m=deepcopy(m)
#   newcoef=deepcopy(newcoef)
#   b0=deepcopy(b0)
#   cd=deepcopy(cd)
#   r=m.rr
#   p=m.pp
#   for icoef = 1:nnz(newcoef)
#       oldcoefval = icoef > nnz(oldcoef) ? zero(T) : oldcoef.coef[icoef]
#       newcoef.coef[icoef] = oldcoefval+f*(coefdiff.coef[icoef])
#   end
#   b0 = oldb0+f*b0diff
#   updateμ!(r, Lasso.linpred!(scratchmu, cd, newcoef, b0))
#   dev = deviance(r)
#   curλ*Lasso.P(α, newcoef, cd.ω)
#   obj = dev/2 + curλ*Lasso.P(α, newcoef, cd.ω)
# end
# using Gadfly
# Δf=f*1
# fs=-100f:Δf:100f
# devfs=broadcast(f->devf(b0,newcoef,m,cd,f),fs)
# plot(layer(x=fs,y=devfs,Geom.line),layer(x=[0],y=[objold],Geom.point))
# r.var
# #
#
# fieldnames(newcoef)
# size(newcoef)
#
# # problematic case 2
#
# using Lasso, StatsBase
# X=[3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 3.461994189597978 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.7695953516783822 2.0771965137587864 2.0771965137587864 2.0771965137587864 2.0771965137587864 2.0771965137587864 2.0771965137587864 2.0771965137587864 1.3847976758391911 1.3847976758391911 1.3847976758391911 1.3847976758391911 1.3847976758391911 1.3847976758391911 1.3847976758391911 1.3847976758391911 1.3847976758391911 1.3847976758391911 1.3847976758391911 1.3847976758391911 1.3847976758391911 1.3847976758391911 1.3847976758391911 1.3847976758391911 1.3847976758391911 1.3847976758391911 0.6923988379195956 0.6923988379195956 0.6923988379195956 0.6923988379195956 0.6923988379195956 0.6923988379195956 0.6923988379195956 0.6923988379195956 0.6923988379195956 0.6923988379195956 0.6923988379195956 0.6923988379195956 0.6923988379195956 0.6923988379195956 0.6923988379195956 0.6923988379195956 0.6923988379195956 0.6923988379195956 0.6923988379195956 0.6923988379195956 0.6923988379195956 0.6923988379195956 0.6923988379195956 0.6923988379195956 0.6923988379195956]'
# Xwconst=[ones(X) X]
# y=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
# sum(y.!=1.0)
# summarystats(y)
# std(y)
# problem2fit = fit(GeneralizedLinearModel,Xwconst,y,Poisson();verbose=true)
# problem2fit = fit(GeneralizedLinearModel,ones(X),y,Poisson();verbose=true)
# problem2fit = fit(LassoPath,X,y,Poisson();verbose=true,λ=[0.0])
# problem2fit = fit(LassoPath,X,y,Poisson();verbose=true,λ=[0.0],dofit=false)
# problem2fit = fit(LassoPath,X,y,Poisson();verbose=true)
# intercept=true
# n=length(y)
# T=eltype(X)
# d=Poisson()
# l=canonicallink(d)
# wts=ones(T, length(y))
# wts .*= convert(T, 1/sum(wts))
# offset=similar(y, 0)
# irls_tol =1e-7
# nullmodel = fit(GeneralizedLinearModel, ones(T, n, ifelse(intercept, 1, 0)), y, d, l;
#                     wts=wts, offset=offset, convTol=irls_tol)
# problem2fit = fit(GeneralizedLinearModel,ones(X),y,Poisson();verbose=true,convTol=irls_tol)
# problem2fit = fit(GeneralizedLinearModel,ones(X),y,Poisson();verbose=true,convTol=irls_tol,dofit=false)
#
# problem2fit = fit(GammaLassoPath,X,y,Poisson();γ=0.0,verbose=true)
# problem2fit = fit(LassoPath,ones(X),y,Poisson();verbose=true,intercept=false)


end
