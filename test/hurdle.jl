include("testutils.jl")

using CSV, Base.Test, Lasso, DataFrames

using HurdleDMR

# code to generate R benchmark
# using RCall
# R"if(!require(pscl)){install.packages(\"pscl\");library(pscl)}"
# R"library(pscl)"
# R"data(\"bioChemists\", package = \"pscl\")"
# R"n <- dim(bioChemists)[1]"
# R"set.seed(13)"
# R"bioChemists$offpos = runif(n, min=0, max=3)"
# R"bioChemists$offzero = runif(n, min=0, max=3)"
# bioChemists=rcopy(R"bioChemists")
# writetable(joinpath(testdir,"data","bioChemists.csv"),bioChemists)

bioChemists=CSV.read(joinpath(testdir,"data","bioChemists.csv"))
bioChemists[:marMarried]=bioChemists[:mar] .== "Married"
bioChemists[:femWomen]=bioChemists[:fem] .== "Women"
bioChemists[:art] = convert(Array{Union{Float64, Missings.Missing},1}, bioChemists[:art])

X=convert(Array{Float64,2},bioChemists[:,[:femWomen,:marMarried,:kid5,:phd,:ment]])
Xwconst=[ones(size(X,1)) X]
y=convert(Array{Float64,1},bioChemists[:art])
const ixpartial = 50:60

###########################################################
# Xpos == Xzero
###########################################################
@testset "hurdle with Xpos == Xzero" begin

## logit-poisson
# R"fm_hp1 <- hurdle(art ~ fem + mar + kid5 + phd + ment, data = bioChemists)"
# print(R"summary(fm_hp1)")
# coefsR1=vec(rcopy(R"coef(fm_hp1, matrix = TRUE)"))
# yhatR1=vec(rcopy(R"predict(fm_hp1)"))
# yhatR1partial=vec(rcopy(R"predict(fm_hp1, newdata = bioChemists[ixpartial,])"))
# writetable(joinpath(testdir,"data","hurdle_coefsR1.csv"),DataFrame(coefsR=coefsR1))
# writetable(joinpath(testdir,"data","hurdle_yhatR1.csv"),DataFrame(yhatR1=yhatR1))
# writetable(joinpath(testdir,"data","hurdle_yhatR1partial.csv"),DataFrame(yhatR1partial=yhatR1partial))

coefsR1=vec(convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","hurdle_coefsR1.csv"))))
yhatR1=vec(convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","hurdle_yhatR1.csv"))))
yhatR1partial=vec(convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","hurdle_yhatR1partial.csv"))))

# simple hurdle with GLM underlying
hurdlefit = fit(Hurdle,GeneralizedLinearModel,Xwconst,y)

coefsJ=vcat(coef(hurdlefit)...)
@test coefsJ ≈ coefsR1 rtol=1e-6
yhatJ=predict(hurdlefit, Xwconst)
@test yhatJ ≈ yhatR1 rtol=1e-6
yhatJpartial=predict(hurdlefit, Xwconst[ixpartial,:])
@test yhatJpartial ≈ yhatR1partial rtol=1e-6

# same simple hurdle through UNREGULATED lasso path
hurdleglrfit = fit(Hurdle,GammaLassoPath,X,y;λ=[0.0, 0.01])
coefsJ=vcat(coef(hurdleglrfit;select=:AICc)...)
@test coefsJ ≈ coefsR1 rtol=1e-4
# rdist(coefsJ,coefsR1)
yhatJ = predict(hurdleglrfit, X; select=:AICc)
@test yhatJ ≈ yhatR1 rtol=1e-4
yhatJpartial=predict(hurdleglrfit, X[ixpartial,:]; select=:AICc)
@test yhatJpartial ≈ yhatR1partial rtol=1e-4
yhatJ = predict(hurdleglrfit, X; select=:all)
@test size(yhatJ) == (size(X,1),2)

# regulated lasso path
hurdleglrfit = fit(Hurdle,GammaLassoPath,X,y; γ=0.0)
coefsJ=vcat(coef(hurdleglrfit;select=:AICc)...)
@test coefsJ ≈ coefsR1 rtol=0.25
# rdist(coefsJ,coefsR1)
yhatJ = predict(hurdleglrfit, X; select=:AICc)
@test yhatJ ≈ yhatR1 rtol=0.05
yhatJpartial=predict(hurdleglrfit, X[ixpartial,:]; select=:AICc)
@test yhatJpartial ≈ yhatR1partial rtol=0.05

coefsJpos, coefsJzero = coef(hurdleglrfit;select=:all)
@test size(coefsJpos,1) == 6
@test size(coefsJzero,1) == 6

# this one throws an error because we did not specify the same λ vector for both submodels so they have different lengths
@test_throws AssertionError predict(hurdleglrfit, X; select=:all)

coefsJCVmin=vcat(coef(hurdleglrfit;select=:CVmin)...)
@test coefsJCVmin ≈ coefsR1 rtol=0.30
# rdist(coefsJCVmin,coefsR1)

# try with SharedArray
Xs = convert(SharedArray,X)
hurdleglrfitShared = fit(Hurdle,GammaLassoPath,Xs,y; γ=0.0)
coefsJShared=vcat(coef(hurdleglrfitShared;select=:AICc)...)
@test coefsJShared ≈ coefsJ
yhatJ = predict(hurdleglrfitShared, X; select=:AICc)
@test yhatJ ≈ yhatR1 rtol=0.05
yhatJpartial=predict(hurdleglrfitShared, X[ixpartial,:]; select=:AICc)
@test yhatJpartial ≈ yhatR1partial rtol=0.05

# regulated gamma lasso path
hurdleglrfit = fit(Hurdle,GammaLassoPath,X,y; γ=10.0)
coefsJ=vcat(coef(hurdleglrfit;select=:AICc)...)
@test coefsJ ≈ coefsR1 rtol=0.11
# rdist(coefsJ,coefsR1)
yhatJ = predict(hurdleglrfit, X; select=:AICc)
@test yhatJ ≈ yhatR1 rtol=0.05
yhatJpartial=predict(hurdleglrfit, X[ixpartial,:]; select=:AICc)
@test yhatJpartial ≈ yhatR1partial rtol=0.05

# using ProfileView
# Profile.init(delay=0.001)
# Profile.clear()
# @profile hurdleglrfit2 = fit2(Hurdle,GammaLassoPath,X,y; γ=10.0);
# ProfileView.view()
# Profile.print()

# using DataFrames formula interface
hurdlefit = fit(Hurdle,GeneralizedLinearModel,@formula(art ~ femWomen + marMarried + kid5 + phd + ment), bioChemists)
coefsJ=vcat(coef(hurdlefit)...)
@test coefsJ ≈ coefsR1 rtol=1e-6
yhatJ=predict(hurdlefit, Xwconst)
@test yhatJ ≈ yhatR1 rtol=1e-6
yhatJpartial=predict(hurdlefit, Xwconst[ixpartial,:])
@test yhatJpartial ≈ yhatR1partial rtol=1e-6

end

###########################################################
# Xpos ≠ Xzero
###########################################################
@testset "hurdle with Xpos ≠ Xzero" begin

# regulated gamma lasso path with different Xpos and Xzero
# R"fm_hp2 <- hurdle(art ~ fem + mar + kid5 | phd + ment, data = bioChemists)"
# print(R"summary(fm_hp2)")
# coefsR2=vec(rcopy(R"coef(fm_hp2, matrix = TRUE)"))
# yhatR2=vec(rcopy(R"predict(fm_hp2)"))
# yhatR2partial=vec(rcopy(R"predict(fm_hp2, newdata = bioChemists[ixpartial,])"))
# writetable(joinpath(testdir,"data","hurdle_coefsR2.csv"),DataFrame(coefsR=coefsR2))
# writetable(joinpath(testdir,"data","hurdle_yhatR2.csv"),DataFrame(yhatR2=yhatR2))
# writetable(joinpath(testdir,"data","hurdle_yhatR2partial.csv"),DataFrame(yhatR2partial=yhatR2partial))

coefsR2=vec(convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","hurdle_coefsR2.csv"))))
yhatR2=vec(convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","hurdle_yhatR2.csv"))))
yhatR2partial=vec(convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","hurdle_yhatR2partial.csv"))))

Xpos = X[:,1:3]
Xzero = X[:,4:5]
Xzerowconst=[ones(size(X,1)) Xzero]
Xposwconst=[ones(size(X,1)) Xpos]
# ixpos = y.>0
# ypos = y[ixpos]
# countmap(ypos)

# simple hurdle with GLM underlying
hurdlefit = fit(Hurdle,GeneralizedLinearModel,Xzerowconst,y; Xpos=Xposwconst)
coefsJ=vcat(coef(hurdlefit)...)
@test coefsJ ≈ coefsR2 rtol=1e-5
yhatJ=predict(hurdlefit, Xzerowconst; Xpos=Xposwconst)
@test yhatJ ≈ yhatR2 rtol=1e-6
yhatJpartial=predict(hurdlefit, Xzerowconst[ixpartial,:]; Xpos=Xposwconst[ixpartial,:])
@test yhatJpartial ≈ yhatR2partial rtol=1e-6

# same simple hurdle through UNREGULATED lasso path
hurdleglrfit = fit(Hurdle,GammaLassoPath,Xzero,y; Xpos=Xpos, λ=[0.0,0.0001])
coefsJ=vcat(coef(hurdleglrfit;select=:AICc)...)
@test coefsJ ≈ coefsR2 rtol=1e-4
# rdist(coefsJ,coefsR2)
yhatJ=predict(hurdleglrfit, Xzero; Xpos=Xpos, select=:AICc)
@test yhatJ ≈ yhatR2 rtol=1e-4
yhatJpartial=predict(hurdleglrfit, Xzero[ixpartial,:]; Xpos=Xpos[ixpartial,:], select=:AICc)
@test yhatJpartial ≈ yhatR2partial rtol=1e-4
yhatJ = predict(hurdleglrfit, Xzero; Xpos=Xpos, select=:all)
@test size(yhatJ) == (size(X,1),2)

# regulated lasso path
hurdleglrfit = fit(Hurdle,GammaLassoPath,Xzero,y; Xpos=Xpos, γ=0.0)
coefsJ=vcat(coef(hurdleglrfit;select=:AICc)...)
@test coefsJ ≈ coefsR2 rtol=0.10
# rdist(coefsJ,coefsR2)
yhatJ = predict(hurdleglrfit, Xzero; Xpos=Xpos, select=:AICc)
@test yhatJ ≈ yhatR2 rtol=0.05
yhatJpartial=predict(hurdleglrfit, Xzero[ixpartial,:]; Xpos=Xpos[ixpartial,:], select=:AICc)
@test yhatJpartial ≈ yhatR2partial rtol=0.05

coefsJpos, coefsJzero = coef(hurdleglrfit;select=:all)
@test size(coefsJpos,1) == 4
@test size(coefsJzero,1) == 3

# try with SharedArray
Xzeros = convert(SharedArray,Xzero)
Xposs = convert(SharedArray,Xpos)
hurdleglrfitShared = fit(Hurdle,GammaLassoPath,Xzeros,y; Xpos=Xposs, γ=0.0)
coefsJShared=vcat(coef(hurdleglrfitShared;select=:AICc)...)
@test coefsJShared ≈ coefsJ
yhatJ = predict(hurdleglrfitShared, Xzeros; Xpos=Xposs, select=:AICc)
@test yhatJ ≈ yhatR2 rtol=0.05
yhatJpartial=predict(hurdleglrfitShared, Xzeros[ixpartial,:]; Xpos=Xposs[ixpartial,:], select=:AICc)
@test yhatJpartial ≈ yhatR2partial rtol=0.05

# regulated gamma lasso path
hurdleglrfit = fit(Hurdle,GammaLassoPath,Xzero,y; Xpos=Xpos, γ=10.0)
coefsJ=vcat(coef(hurdleglrfit;select=:AICc)...)
@test coefsJ ≈ coefsR2 rtol=0.15
# rdist(coefsJ,coefsR2)
yhatJ = predict(hurdleglrfit, Xzero; Xpos=Xpos, select=:AICc)
@test yhatJ ≈ yhatR2 rtol=0.05
yhatJpartial=predict(hurdleglrfit, Xzero[ixpartial,:]; Xpos=Xpos[ixpartial,:], select=:AICc)
@test yhatJpartial ≈ yhatR2partial rtol=0.05

# using DataFrames formula interface
hurdlefit = fit(Hurdle,GeneralizedLinearModel,@formula(art ~ phd + ment), bioChemists; fpos = @formula(art ~ femWomen + marMarried + kid5))
coefsJ=vcat(coef(hurdlefit)...)
@test coefsJ ≈ coefsR2 rtol=1e-5
# rdist(coefsJ,coefsR2)
yhatJ=predict(hurdlefit, Xzerowconst; Xpos=Xposwconst)
@test yhatJ ≈ yhatR2 rtol=1e-6
yhatJpartial=predict(hurdlefit, Xzerowconst[ixpartial,:]; Xpos=Xposwconst[ixpartial,:])
@test yhatJpartial ≈ yhatR2partial rtol=1e-6

end

###########################################################
# Xpos ≠ Xzero
###########################################################
@testset "hurdle with Xpos ≠ Xzero AND offset specified" begin

# regulated gamma lasso path with different Xpos and Xzero and an offset
# R"fm_hp3 <- hurdle(art ~ fem + mar + kid5 + offset(offpos) | phd + ment + offset(offzero), data = bioChemists)"
# print(R"summary(fm_hp3)")
# coefsR3=vec(rcopy(R"coef(fm_hp3, matrix = TRUE)"))
# yhatR3=vec(rcopy(R"predict(fm_hp3)"))
# yhatR3partial=vec(rcopy(R"predict(fm_hp3, newdata = bioChemists[ixpartial,])"))
# writetable(joinpath(testdir,"data","hurdle_coefsR3.csv"),DataFrame(coefsR=coefsR3))
# writetable(joinpath(testdir,"data","hurdle_yhatR3.csv"),DataFrame(yhatR3=yhatR3))
# writetable(joinpath(testdir,"data","hurdle_yhatR3partial.csv"),DataFrame(yhatR3partial=yhatR3partial))

coefsR3=vec(convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","hurdle_coefsR3.csv"))))
yhatR3=vec(convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","hurdle_yhatR3.csv"))))
yhatR3partial=vec(convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","hurdle_yhatR3partial.csv"))))

offpos = convert(Vector{Float64},bioChemists[:offpos])
offzero = convert(Vector{Float64},bioChemists[:offzero])
Xpos = X[:,1:3]
Xzero = X[:,4:5]
Xzerowconst=[ones(size(X,1)) Xzero]
Xposwconst=[ones(size(X,1)) Xpos]
# ixpos = y.>0
# ypos = y[ixpos]
# countmap(ypos)

# simple hurdle with GLM underlying
hurdlefit = fit(Hurdle,GeneralizedLinearModel,Xzerowconst,y; Xpos=Xposwconst, offsetzero=offzero, offsetpos=offpos)
coefsJ=vcat(coef(hurdlefit)...)
@test coefsJ ≈ coefsR3 rtol=1e-5
yhatJ=predict(hurdlefit, Xzerowconst; Xpos=Xposwconst, offsetzero=offzero, offsetpos=offpos)
@test yhatJ ≈ yhatR3 rtol=1e-5
yhatJpartial=predict(hurdlefit, Xzerowconst[ixpartial,:]; Xpos=Xposwconst[ixpartial,:], offsetzero=offzero[ixpartial], offsetpos=offpos[ixpartial])
@test yhatJpartial ≈ yhatR3partial rtol=1e-6

# same simple hurdle through UNREGULATED lasso path
hurdleglrfit = fit(Hurdle,GammaLassoPath,Xzero,y; Xpos=Xpos, offsetzero=offzero, offsetpos=offpos, λ=[0.0,0.0001])
coefsJ=vcat(coef(hurdleglrfit;select=:AICc)...)
@test coefsJ ≈ coefsR3 rtol=1e-4
# rdist(coefsJ,coefsR3)
yhatJ=predict(hurdleglrfit, Xzero; Xpos=Xpos, offsetzero=offzero, offsetpos=offpos, select=:AICc)
@test yhatJ ≈ yhatR3 rtol=1e-4
yhatJpartial=predict(hurdleglrfit, Xzero[ixpartial,:]; Xpos=Xpos[ixpartial,:], offsetzero=offzero[ixpartial], offsetpos=offpos[ixpartial], select=:AICc)
@test yhatJpartial ≈ yhatR3partial rtol=1e-4
yhatJ = predict(hurdleglrfit, Xzero; Xpos=Xpos, offsetzero=offzero, offsetpos=offpos, select=:all)
@test size(yhatJ) == (size(X,1),2)

# regulated lasso path
hurdleglrfit = fit(Hurdle,GammaLassoPath,Xzero,y; Xpos=Xpos, offsetzero=offzero, offsetpos=offpos, γ=0.0)
coefsJ=vcat(coef(hurdleglrfit;select=:AICc)...)
@test coefsJ ≈ coefsR3 rtol=0.10
# rdist(coefsJ,coefsR3)
yhatJ = predict(hurdleglrfit, Xzero; Xpos=Xpos, offsetzero=offzero, offsetpos=offpos, select=:AICc)
@test yhatJ ≈ yhatR3 rtol=0.05
yhatJpartial=predict(hurdleglrfit, Xzero[ixpartial,:]; Xpos=Xpos[ixpartial,:], offsetzero=offzero[ixpartial], offsetpos=offpos[ixpartial], select=:AICc)
@test yhatJpartial ≈ yhatR3partial rtol=0.05

coefsJpos, coefsJzero = coef(hurdleglrfit;select=:all)
@test size(coefsJpos,1) == 4
@test size(coefsJzero,1) == 3

# try with SharedArray
Xzeros = convert(SharedArray,Xzero)
Xposs = convert(SharedArray,Xpos)
offzeros = convert(SharedArray,offzero)
offposs = convert(SharedArray,offpos)
hurdleglrfitShared = fit(Hurdle,GammaLassoPath,Xzeros,y; Xpos=Xposs, offsetzero=offzeros, offsetpos=offposs, γ=0.0)
coefsJShared=vcat(coef(hurdleglrfitShared;select=:AICc)...)
@test coefsJShared ≈ coefsJ
yhatJ = predict(hurdleglrfitShared, Xzeros; Xpos=Xposs, offsetzero=offzeros, offsetpos=offposs, select=:AICc)
@test yhatJ ≈ yhatR3 rtol=0.05
yhatJpartial=predict(hurdleglrfitShared, Xzeros[ixpartial,:]; Xpos=Xposs[ixpartial,:], offsetzero=offzeros[ixpartial], offsetpos=offposs[ixpartial], select=:AICc)
@test yhatJpartial ≈ yhatR3partial rtol=0.05

# regulated gamma lasso path
hurdleglrfit = fit(Hurdle,GammaLassoPath,Xzero,y; Xpos=Xpos, offsetzero=offzero, offsetpos=offpos, γ=10.0)
coefsJ=vcat(coef(hurdleglrfit;select=:AICc)...)
@test coefsJ ≈ coefsR3 rtol=0.15
# rdist(coefsJ,coefsR3)
yhatJ = predict(hurdleglrfit, Xzero; Xpos=Xpos, offsetzero=offzero, offsetpos=offpos, select=:AICc)
@test yhatJ ≈ yhatR3 rtol=0.05
yhatJpartial=predict(hurdleglrfit, Xzero[ixpartial,:]; Xpos=Xpos[ixpartial,:], offsetzero=offzero[ixpartial], offsetpos=offpos[ixpartial], select=:AICc)
@test yhatJpartial ≈ yhatR3partial rtol=0.05

# using DataFrames formula interface
hurdlefit = fit(Hurdle,GeneralizedLinearModel,@formula(art ~ phd + ment), bioChemists; fpos = @formula(art ~ femWomen + marMarried + kid5), offsetzero=offzero, offsetpos=offpos)
coefsJ=vcat(coef(hurdlefit)...)
@test coefsJ ≈ coefsR3 rtol=1e-5
# rdist(coefsJ,coefsR3)
yhatJ=predict(hurdlefit, Xzerowconst; Xpos=Xposwconst, offsetzero=offzero, offsetpos=offpos)
@test yhatJ ≈ yhatR3 rtol=1e-5
yhatJpartial=predict(hurdlefit, Xzerowconst[ixpartial,:]; Xpos=Xposwconst[ixpartial,:], offsetzero=offzero[ixpartial], offsetpos=offpos[ixpartial])
@test yhatJpartial ≈ yhatR3partial rtol=1e-5

end

###########################################################
# degenrate cases
###########################################################
@testset "hurdle degenerate cases" begin

# degenerate positive counts data case 1
include(joinpath(testdir,"data","degenerate_hurdle_1.jl"))
hurdle = fit(Hurdle,GammaLassoPath,X,y)
coefsJ=vcat(coef(hurdle;select=:AICc)...)
@test vec(coefsJ) ≈ [0.0, 0.0, -6.04112, 0.675767] rtol=1e-4
coefsJpos, coefsJzero = coef(hurdle;select=:all)
@test size(coefsJpos,1) == 2
@test coefsJpos == zeros(coefsJpos)
@test size(coefsJzero,1) == 2

# degenerate positive counts data case 1 without >1
y0or1=deepcopy(y)
y0or1[y.>1]=1
hurdle = fit(Hurdle,GammaLassoPath,X,y0or1)
coefs0or1=vcat(coef(hurdle;select=:AICc)...)
@test coefs0or1 == coefsJ
coefs0or1pos, coefs0or1zero = coef(hurdle;select=:all)
@test coefs0or1pos == coefsJpos
@test coefs0or1zero == coefsJzero

# degenerate positive counts data case 1 without zeros
y0or1 = deepcopy(y)
y0or1[y.==0]=1
hurdle = fit(Hurdle,GammaLassoPath,X,y0or1)
coef(hurdle;select=:AICc)
coefs0or1=vcat(coef(hurdle;select=:AICc)...)
@test vec(coefs0or1) ≈ [-7.34062, 0.0, 0.0, 0.0] rtol=1e-4
coefs0or1pos, coefs0or1zero = coef(hurdle;select=:all)
@test all(iszero,coefs0or1zero)

# degenerate positive counts data case 1 without only zeros
y0or1 = zeros(y)
@test_throws ErrorException fit(Hurdle,GammaLassoPath,X,y0or1)

# degenerate positive counts data case 2
include(joinpath(testdir,"data","degenerate_hurdle_2.jl"))
hurdle = fit(Hurdle,GammaLassoPath,X,y)
coefsJ=vcat(coef(hurdle;select=:AICc)...)
@test vec(coefsJ) ≈ [0.0,0.0,-5.30128195796556,0.1854148891565171]'' rtol=1e-4
coefsJpos, coefsJzero = coef(hurdle;select=:all)
@test size(coefsJpos,1) == 2
@test coefsJpos == zeros(coefsJpos)
@test size(coefsJzero,1) == 2

# degenerate positive counts data case 3
include(joinpath(testdir,"data","degenerate_hurdle_3.jl"))
hurdle = fit(Hurdle,GammaLassoPath,X,y)
coefsJ=vcat(coef(hurdle;select=:AICc)...)
@test vec(coefsJ) ≈ [0.0,0.0,-4.541820686620407,0.0]'' rtol=1e-4
coefsJpos, coefsJzero = coef(hurdle;select=:all)
@test size(coefsJpos,1) == 2
@test coefsJpos == zeros(coefsJpos)
@test size(coefsJzero,1) == 2

# this test case used to give numerical headaches to devresid(PositivePoisson(),...)
include(joinpath(testdir,"data","degenerate_hurdle_5.jl"))
# X = load(joinpath(testdir,"data","degenerate_hurdle5.jld"),"X")
# Xpos = load(joinpath(testdir,"data","degenerate_hurdle5.jld"),"Xpos")
# y = load(joinpath(testdir,"data","degenerate_hurdle5.jld"),"y")
# offset = load(joinpath(testdir,"data","degenerate_hurdle5.jld"),"offset")
# show(offset)
hurdle = fit(Hurdle,GammaLassoPath,X,y; Xpos=Xpos, offset=offset)
@test !(isnull(hurdle.mpos))
@test !(isnull(hurdle.mzero))
@test typeof(hurdle.mpos) <: GammaLassoPath
@test typeof(hurdle.mzero) <: GammaLassoPath

# M = GammaLassoPath
# dzero = Binomial()
# lzero = canonicallink(dzero)
# dpos = PositivePoisson()
# lpos = canonicallink(dpos)
# dofit = true
# wts = ones(length(y))
# offsetzero = Vector{Float64}()
# offsetpos = Vector{Float64}()
# verbose = true
# fitargs = ()
#
# ixpos, Iy = HurdleDMR.setIy(y)
#
# offsetzero, offsetpos = HurdleDMR.setoffsets(y, ixpos, offset, offsetzero, offsetpos)
#
# mzero, fittedzero = HurdleDMR.fitzero(M, X, Iy, dzero, lzero, dofit, wts, offsetzero, verbose, fitargs...)
#
# # Xpos optional argument allows to specify a data matrix only for positive counts
# if Xpos == nothing
#   # use X for Xpos too
#   Xpos = X[ixpos,:]
# elseif size(Xpos,1) == length(y)
#   # Xpos has same dimensions as X, take only positive y ones
#   Xpos = Xpos[ixpos,:]
# end
#
# mpos, fittedpos = HurdleDMR.fitpos(M, Xpos, y[ixpos], dpos, lpos, dofit, wts[ixpos], offsetpos, verbose, fitargs...)
#
# T = eltype(y)
# intercept = true
# d = dpos
# l = lpos
# @enter fit(GeneralizedLinearModel, ones(T, length(ypos), ifelse(intercept, 1, 0)), ypos, d, l;
#   wts=wts[ixpos], offset=offsetpos, convTol=1e-7, dofit=dofit)
#
# hurdle = fit(Hurdle,GammaLassoPath,X,y; Xpos=Xpos, offset=offset)
# show(y)
# coefsJ=vcat(coef(hurdle;select=:AICc)...)


end

# TODO never got this to work:
# bioChemists[:art]=convert(Vector{Float64},bioChemists[:art])
# hurdlefit = fit(Hurdle,GammaLassoPath,@formula(art ~ fem + mar + kid5 + phd + ment), bioChemists;intercept=false)
# coefsJ=coef(hurdlefit)
# @test coefsJ ≈ coefsR1 rtol=1e-6
# rdist(coefsR1,coefsJ)
