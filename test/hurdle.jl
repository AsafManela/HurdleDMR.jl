rtol=0.05
rdist(x::Number,y::Number) = abs(x-y)/max(abs(x),abs(y))
rdist{T<:Number,S<:Number}(x::AbstractArray{T}, y::AbstractArray{S}; norm::Function=vecnorm) = norm(x - y) / max(norm(x), norm(y))

testfolder = dirname(@__FILE__)
# push!(LOAD_PATH, joinpath(testfolder,".."))
# push!(LOAD_PATH, joinpath(testfolder,"..","src"))

using FactCheck, Lasso, DataFrames

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
# writetable(joinpath(testfolder,"data","bioChemists.csv"),bioChemists)

bioChemists=readtable(joinpath(testfolder,"data","bioChemists.csv"))
bioChemists[:marMarried]=bioChemists[:mar] .== "Married"
bioChemists[:femWomen]=bioChemists[:fem] .== "Women"
bioChemists[:art] = convert(DataVector{Float64}, bioChemists[:art])

X=convert(Array{Float64,2},bioChemists[:,[:femWomen,:marMarried,:kid5,:phd,:ment]])
Xwconst=[ones(size(X,1)) X]
y=convert(Array{Float64,1},bioChemists[:art])

###########################################################
# Xpos == Xzero
###########################################################
facts("hurdle with Xpos == Xzero") do

## logit-poisson
# R"fm_hp1 <- hurdle(art ~ fem + mar + kid5 + phd + ment, data = bioChemists)"
# print(R"summary(fm_hp1)")
# coefsR1=vec(rcopy(R"coef(fm_hp1, matrix = TRUE)"))
# yhatR1=vec(rcopy(R"predict(fm_hp1)"))
# yhatR1partial=vec(rcopy(R"predict(fm_hp1, newdata = bioChemists[50:60,])"))
# writetable(joinpath(testfolder,"data","hurdle_coefsR1.csv"),DataFrame(coefsR=coefsR1))
# writetable(joinpath(testfolder,"data","hurdle_yhatR1.csv"),DataFrame(yhatR1=yhatR1))
# writetable(joinpath(testfolder,"data","hurdle_yhatR1partial.csv"),DataFrame(yhatR1partial=yhatR1partial))

coefsR1=vec(convert(Matrix{Float64},readtable(joinpath(testfolder,"data","hurdle_coefsR1.csv"))))
yhatR1=vec(convert(Matrix{Float64},readtable(joinpath(testfolder,"data","hurdle_yhatR1.csv"))))
yhatR1partial=vec(convert(Matrix{Float64},readtable(joinpath(testfolder,"data","hurdle_yhatR1partial.csv"))))

# simple hurdle with GLM underlying
hurdlefit = fit(Hurdle,GeneralizedLinearModel,Xwconst,y)

coefsJ=vcat(coef(hurdlefit)...)
@fact coefsJ --> roughly(coefsR1;rtol=1e-6)
yhatJ=vcat(predict(hurdlefit, Xwconst)...)
@fact yhatJ --> roughly(yhatR1;rtol=1e-6)
yhatJpartial=vcat(predict(hurdlefit, Xwconst[50:60,:])...)
@fact yhatJpartial --> roughly(yhatR1partial;rtol=1e-6)

# same simple hurdle through UNREGULATED lasso path
hurdleglrfit = fit(Hurdle,GammaLassoPath,X,y;λ=[0.0, 0.01])
coefsJ=vcat(coef(hurdleglrfit;select=:AICc)...)
@fact coefsJ --> roughly(coefsR1;rtol=1e-4)
# rdist(coefsJ,coefsR1)
yhatJ = vcat(predict(hurdleglrfit, X; select=:AICc)...)
@fact yhatJ --> roughly(yhatR1;rtol=1e-4)
yhatJpartial=vcat(predict(hurdleglrfit, X[50:60,:]; select=:AICc)...)
@fact yhatJpartial --> roughly(yhatR1partial;rtol=1e-4)
yhatJ = predict(hurdleglrfit, X; select=:all)
@fact size(yhatJ) --> (size(X,1),2)

# regulated lasso path
hurdleglrfit = fit(Hurdle,GammaLassoPath,X,y; γ=0.0)
coefsJ=vcat(coef(hurdleglrfit;select=:AICc)...)
@fact coefsJ --> roughly(coefsR1;rtol=0.25)
# rdist(coefsJ,coefsR1)
yhatJ = vcat(predict(hurdleglrfit, X; select=:AICc)...)
@fact yhatJ --> roughly(yhatR1;rtol=0.05)
yhatJpartial=vcat(predict(hurdleglrfit, X[50:60,:]; select=:AICc)...)
@fact yhatJpartial --> roughly(yhatR1partial;rtol=0.05)

coefsJpos, coefsJzero = coef(hurdleglrfit;select=:all)
@fact size(coefsJpos,1) --> 6
@fact size(coefsJzero,1) --> 6

# this one throws an error because we did not specify the same λ vector for both submodels so they have different lengths
@fact_throws predict(hurdleglrfit, X; select=:all)

coefsJCVmin=vcat(coef(hurdleglrfit;select=:CVmin)...)
@fact coefsJCVmin --> roughly(coefsR1;rtol=0.30)
# rdist(coefsJCVmin,coefsR1)

# try with SharedArray
Xs = convert(SharedArray,X)
hurdleglrfitShared = fit(Hurdle,GammaLassoPath,Xs,y; γ=0.0)
coefsJShared=vcat(coef(hurdleglrfitShared;select=:AICc)...)
@fact coefsJShared --> roughly(coefsJ)
yhatJ = vcat(predict(hurdleglrfitShared, X; select=:AICc)...)
@fact yhatJ --> roughly(yhatR1;rtol=0.05)
yhatJpartial=vcat(predict(hurdleglrfitShared, X[50:60,:]; select=:AICc)...)
@fact yhatJpartial --> roughly(yhatR1partial;rtol=0.05)

# regulated gamma lasso path
hurdleglrfit = fit(Hurdle,GammaLassoPath,X,y; γ=10.0)
coefsJ=vcat(coef(hurdleglrfit;select=:AICc)...)
@fact coefsJ --> roughly(coefsR1;rtol=0.11)
# rdist(coefsJ,coefsR1)
yhatJ = vcat(predict(hurdleglrfit, X; select=:AICc)...)
@fact yhatJ --> roughly(yhatR1;rtol=0.05)
yhatJpartial=vcat(predict(hurdleglrfit, X[50:60,:]; select=:AICc)...)
@fact yhatJpartial --> roughly(yhatR1partial;rtol=0.05)

# using ProfileView
# Profile.init(delay=0.001)
# Profile.clear()
# @profile hurdleglrfit2 = fit2(Hurdle,GammaLassoPath,X,y; γ=10.0);
# ProfileView.view()
# Profile.print()

# using DataFrames formula interface
hurdlefit = fit(Hurdle,GeneralizedLinearModel,@formula(art ~ femWomen + marMarried + kid5 + phd + ment), bioChemists)
coefsJ=vcat(coef(hurdlefit)...)
@fact coefsJ --> roughly(coefsR1;rtol=1e-6)
yhatJ=vcat(predict(hurdlefit, Xwconst)...)
@fact yhatJ --> roughly(yhatR1;rtol=1e-6)
yhatJpartial=vcat(predict(hurdlefit, Xwconst[50:60,:])...)
@fact yhatJpartial --> roughly(yhatR1partial;rtol=1e-6)

end

###########################################################
# Xpos ≠ Xzero
###########################################################
facts("hurdle with Xpos ≠ Xzero") do

# regulated gamma lasso path with different Xpos and Xzero
# R"fm_hp2 <- hurdle(art ~ fem + mar + kid5 | phd + ment, data = bioChemists)"
# print(R"summary(fm_hp2)")
# coefsR2=vec(rcopy(R"coef(fm_hp2, matrix = TRUE)"))
# yhatR2=vec(rcopy(R"predict(fm_hp2)"))
# yhatR2partial=vec(rcopy(R"predict(fm_hp2, newdata = bioChemists[50:60,])"))
# writetable(joinpath(testfolder,"data","hurdle_coefsR2.csv"),DataFrame(coefsR=coefsR2))
# writetable(joinpath(testfolder,"data","hurdle_yhatR2.csv"),DataFrame(yhatR2=yhatR2))
# writetable(joinpath(testfolder,"data","hurdle_yhatR2partial.csv"),DataFrame(yhatR2partial=yhatR2partial))

coefsR2=vec(convert(Matrix{Float64},readtable(joinpath(testfolder,"data","hurdle_coefsR2.csv"))))
yhatR2=vec(convert(Matrix{Float64},readtable(joinpath(testfolder,"data","hurdle_yhatR2.csv"))))
yhatR2partial=vec(convert(Matrix{Float64},readtable(joinpath(testfolder,"data","hurdle_yhatR2partial.csv"))))

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
@fact coefsJ --> roughly(coefsR2;rtol=1e-5)
yhatJ=vcat(predict(hurdlefit, Xzerowconst; Xpos=Xposwconst)...)
@fact yhatJ --> roughly(yhatR2;rtol=1e-6)
yhatJpartial=vcat(predict(hurdlefit, Xzerowconst[50:60,:]; Xpos=Xposwconst[50:60,:])...)
@fact yhatJpartial --> roughly(yhatR2partial;rtol=1e-6)

# same simple hurdle through UNREGULATED lasso path
hurdleglrfit = fit(Hurdle,GammaLassoPath,Xzero,y; Xpos=Xpos, λ=[0.0,0.0001])
coefsJ=vcat(coef(hurdleglrfit;select=:AICc)...)
@fact coefsJ --> roughly(coefsR2;rtol=1e-4)
# rdist(coefsJ,coefsR2)
yhatJ=vcat(predict(hurdleglrfit, Xzero; Xpos=Xpos, select=:AICc)...)
@fact yhatJ --> roughly(yhatR2;rtol=1e-4)
yhatJpartial=vcat(predict(hurdleglrfit, Xzero[50:60,:]; Xpos=Xpos[50:60,:], select=:AICc)...)
@fact yhatJpartial --> roughly(yhatR2partial;rtol=1e-4)
yhatJ = predict(hurdleglrfit, Xzero; Xpos=Xpos, select=:all)
@fact size(yhatJ) --> (size(X,1),2)

# regulated lasso path
hurdleglrfit = fit(Hurdle,GammaLassoPath,Xzero,y; Xpos=Xpos, γ=0.0)
coefsJ=vcat(coef(hurdleglrfit;select=:AICc)...)
@fact coefsJ --> roughly(coefsR2;rtol=0.10)
# rdist(coefsJ,coefsR2)
yhatJ = vcat(predict(hurdleglrfit, Xzero; Xpos=Xpos, select=:AICc)...)
@fact yhatJ --> roughly(yhatR2;rtol=0.05)
yhatJpartial=vcat(predict(hurdleglrfit, Xzero[50:60,:]; Xpos=Xpos[50:60,:], select=:AICc)...)
@fact yhatJpartial --> roughly(yhatR2partial;rtol=0.05)

coefsJpos, coefsJzero = coef(hurdleglrfit;select=:all)
@fact size(coefsJpos,1) --> 4
@fact size(coefsJzero,1) --> 3

# try with SharedArray
Xzeros = convert(SharedArray,Xzero)
Xposs = convert(SharedArray,Xpos)
hurdleglrfitShared = fit(Hurdle,GammaLassoPath,Xzeros,y; Xpos=Xposs, γ=0.0)
coefsJShared=vcat(coef(hurdleglrfitShared;select=:AICc)...)
@fact coefsJShared --> roughly(coefsJ)
yhatJ = vcat(predict(hurdleglrfitShared, Xzeros; Xpos=Xposs, select=:AICc)...)
@fact yhatJ --> roughly(yhatR2;rtol=0.05)
yhatJpartial=vcat(predict(hurdleglrfitShared, Xzeros[50:60,:]; Xpos=Xposs[50:60,:], select=:AICc)...)
@fact yhatJpartial --> roughly(yhatR2partial;rtol=0.05)

# regulated gamma lasso path
hurdleglrfit = fit(Hurdle,GammaLassoPath,Xzero,y; Xpos=Xpos, γ=10.0)
coefsJ=vcat(coef(hurdleglrfit;select=:AICc)...)
@fact coefsJ --> roughly(coefsR2;rtol=0.15)
# rdist(coefsJ,coefsR2)
yhatJ = vcat(predict(hurdleglrfit, Xzero; Xpos=Xpos, select=:AICc)...)
@fact yhatJ --> roughly(yhatR2;rtol=0.05)
yhatJpartial=vcat(predict(hurdleglrfit, Xzero[50:60,:]; Xpos=Xpos[50:60,:], select=:AICc)...)
@fact yhatJpartial --> roughly(yhatR2partial;rtol=0.05)

# using DataFrames formula interface
hurdlefit = fit(Hurdle,GeneralizedLinearModel,@formula(art ~ phd + ment), bioChemists; fpos = @formula(art ~ femWomen + marMarried + kid5))
coefsJ=vcat(coef(hurdlefit)...)
@fact coefsJ --> roughly(coefsR2;rtol=1e-5)
# rdist(coefsJ,coefsR2)
yhatJ=vcat(predict(hurdlefit, Xzerowconst; Xpos=Xposwconst)...)
@fact yhatJ --> roughly(yhatR2;rtol=1e-6)
yhatJpartial=vcat(predict(hurdlefit, Xzerowconst[50:60,:]; Xpos=Xposwconst[50:60,:])...)
@fact yhatJpartial --> roughly(yhatR2partial;rtol=1e-6)

end

###########################################################
# Xpos ≠ Xzero
###########################################################
facts("hurdle with Xpos ≠ Xzero AND offset specified") do

# regulated gamma lasso path with different Xpos and Xzero and an offset
# R"fm_hp3 <- hurdle(art ~ fem + mar + kid5 + offset(offpos) | phd + ment + offset(offzero), data = bioChemists)"
# print(R"summary(fm_hp3)")
# coefsR3=vec(rcopy(R"coef(fm_hp3, matrix = TRUE)"))
# yhatR3=vec(rcopy(R"predict(fm_hp3)"))
# yhatR3partial=vec(rcopy(R"predict(fm_hp3, newdata = bioChemists[50:60,])"))
# writetable(joinpath(testfolder,"data","hurdle_coefsR3.csv"),DataFrame(coefsR=coefsR3))
# writetable(joinpath(testfolder,"data","hurdle_yhatR3.csv"),DataFrame(yhatR3=yhatR3))
# writetable(joinpath(testfolder,"data","hurdle_yhatR3partial.csv"),DataFrame(yhatR3partial=yhatR3partial))

coefsR3=vec(convert(Matrix{Float64},readtable(joinpath(testfolder,"data","hurdle_coefsR3.csv"))))
yhatR3=vec(convert(Matrix{Float64},readtable(joinpath(testfolder,"data","hurdle_yhatR3.csv"))))
yhatR3partial=vec(convert(Matrix{Float64},readtable(joinpath(testfolder,"data","hurdle_yhatR3partial.csv"))))

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
@fact coefsJ --> roughly(coefsR3;rtol=1e-5)
yhatJ=vcat(predict(hurdlefit, Xzerowconst; Xpos=Xposwconst, offsetzero=offzero, offsetpos=offpos)...)
@fact yhatJ --> roughly(yhatR3;rtol=1e-5)
yhatJpartial=vcat(predict(hurdlefit, Xzerowconst[50:60,:]; Xpos=Xposwconst[50:60,:], offsetzero=offzero[50:60], offsetpos=offpos[50:60])...)
@fact yhatJpartial --> roughly(yhatR3partial;rtol=1e-6)

# same simple hurdle through UNREGULATED lasso path
hurdleglrfit = fit(Hurdle,GammaLassoPath,Xzero,y; Xpos=Xpos, offsetzero=offzero, offsetpos=offpos, λ=[0.0,0.0001])
coefsJ=vcat(coef(hurdleglrfit;select=:AICc)...)
@fact coefsJ --> roughly(coefsR3;rtol=1e-4)
# rdist(coefsJ,coefsR3)
yhatJ=vcat(predict(hurdleglrfit, Xzero; Xpos=Xpos, offsetzero=offzero, offsetpos=offpos, select=:AICc)...)
@fact yhatJ --> roughly(yhatR3;rtol=1e-4)
yhatJpartial=vcat(predict(hurdleglrfit, Xzero[50:60,:]; Xpos=Xpos[50:60,:], offsetzero=offzero[50:60], offsetpos=offpos[50:60], select=:AICc)...)
@fact yhatJpartial --> roughly(yhatR3partial;rtol=1e-4)
yhatJ = predict(hurdleglrfit, Xzero; Xpos=Xpos, offsetzero=offzero, offsetpos=offpos, select=:all)
@fact size(yhatJ) --> (size(X,1),2)

# regulated lasso path
hurdleglrfit = fit(Hurdle,GammaLassoPath,Xzero,y; Xpos=Xpos, offsetzero=offzero, offsetpos=offpos, γ=0.0)
coefsJ=vcat(coef(hurdleglrfit;select=:AICc)...)
@fact coefsJ --> roughly(coefsR3;rtol=0.10)
# rdist(coefsJ,coefsR3)
yhatJ = vcat(predict(hurdleglrfit, Xzero; Xpos=Xpos, offsetzero=offzero, offsetpos=offpos, select=:AICc)...)
@fact yhatJ --> roughly(yhatR3;rtol=0.05)
yhatJpartial=vcat(predict(hurdleglrfit, Xzero[50:60,:]; Xpos=Xpos[50:60,:], offsetzero=offzero[50:60], offsetpos=offpos[50:60], select=:AICc)...)
@fact yhatJpartial --> roughly(yhatR3partial;rtol=0.05)

coefsJpos, coefsJzero = coef(hurdleglrfit;select=:all)
@fact size(coefsJpos,1) --> 4
@fact size(coefsJzero,1) --> 3

# try with SharedArray
Xzeros = convert(SharedArray,Xzero)
Xposs = convert(SharedArray,Xpos)
offzeros = convert(SharedArray,offzero)
offposs = convert(SharedArray,offpos)
hurdleglrfitShared = fit(Hurdle,GammaLassoPath,Xzeros,y; Xpos=Xposs, offsetzero=offzeros, offsetpos=offposs, γ=0.0)
coefsJShared=vcat(coef(hurdleglrfitShared;select=:AICc)...)
@fact coefsJShared --> roughly(coefsJ)
yhatJ = vcat(predict(hurdleglrfitShared, Xzeros; Xpos=Xposs, offsetzero=offzeros, offsetpos=offposs, select=:AICc)...)
@fact yhatJ --> roughly(yhatR3;rtol=0.05)
yhatJpartial=vcat(predict(hurdleglrfitShared, Xzeros[50:60,:]; Xpos=Xposs[50:60,:], offsetzero=offzeros[50:60], offsetpos=offposs[50:60], select=:AICc)...)
@fact yhatJpartial --> roughly(yhatR3partial;rtol=0.05)

# regulated gamma lasso path
hurdleglrfit = fit(Hurdle,GammaLassoPath,Xzero,y; Xpos=Xpos, offsetzero=offzero, offsetpos=offpos, γ=10.0)
coefsJ=vcat(coef(hurdleglrfit;select=:AICc)...)
@fact coefsJ --> roughly(coefsR3;rtol=0.15)
# rdist(coefsJ,coefsR3)
yhatJ = vcat(predict(hurdleglrfit, Xzero; Xpos=Xpos, offsetzero=offzero, offsetpos=offpos, select=:AICc)...)
@fact yhatJ --> roughly(yhatR3;rtol=0.05)
yhatJpartial=vcat(predict(hurdleglrfit, Xzero[50:60,:]; Xpos=Xpos[50:60,:], offsetzero=offzero[50:60], offsetpos=offpos[50:60], select=:AICc)...)
@fact yhatJpartial --> roughly(yhatR3partial;rtol=0.05)

# using DataFrames formula interface
hurdlefit = fit(Hurdle,GeneralizedLinearModel,@formula(art ~ phd + ment), bioChemists; fpos = @formula(art ~ femWomen + marMarried + kid5), offsetzero=offzero, offsetpos=offpos)
coefsJ=vcat(coef(hurdlefit)...)
@fact coefsJ --> roughly(coefsR3;rtol=1e-5)
# rdist(coefsJ,coefsR3)
yhatJ=vcat(predict(hurdlefit, Xzerowconst; Xpos=Xposwconst, offsetzero=offzero, offsetpos=offpos)...)
@fact yhatJ --> roughly(yhatR3;rtol=1e-5)
yhatJpartial=vcat(predict(hurdlefit, Xzerowconst[50:60,:]; Xpos=Xposwconst[50:60,:], offsetzero=offzero[50:60], offsetpos=offpos[50:60])...)
@fact yhatJpartial --> roughly(yhatR3partial;rtol=1e-5)

end

###########################################################
# degenrate cases
###########################################################
facts("hurdle degenerate cases") do

# degenerate positive counts data case 1
include(joinpath(testfolder,"data","degenerate_hurdle_1.jl"))
hurdle = fit(Hurdle,GammaLassoPath,X,y)
coefsJ=vcat(coef(hurdle;select=:AICc)...)
@fact vec(coefsJ) --> roughly([0.0, 0.0, -6.04112, 0.675767],1e-4)
coefsJpos, coefsJzero = coef(hurdle;select=:all)
@fact size(coefsJpos,1) --> 2
@fact coefsJpos --> zeros(coefsJpos)
@fact size(coefsJzero,1) --> 2

# degenerate positive counts data case 1 without >1
y0or1=deepcopy(y)
y0or1[y.>1]=1
hurdle = fit(Hurdle,GammaLassoPath,X,y0or1)
coefs0or1=vcat(coef(hurdle;select=:AICc)...)
@fact coefs0or1 --> coefsJ
coefs0or1pos, coefs0or1zero = coef(hurdle;select=:all)
@fact coefs0or1pos --> coefsJpos
@fact coefs0or1zero --> coefsJzero

# degenerate positive counts data case 1 without zeros
y0or1 = deepcopy(y)
y0or1[y.==0]=1
hurdle = fit(Hurdle,GammaLassoPath,X,y0or1)
coef(hurdle;select=:AICc)
coefs0or1=vcat(coef(hurdle;select=:AICc)...)
@fact vec(coefs0or1) --> roughly([-7.34062, 0.0, 0.0, 0.0],1e-4)
coefs0or1pos, coefs0or1zero = coef(hurdle;select=:all)
@fact all(iszero,coefs0or1zero) --> true

# degenerate positive counts data case 1 without only zeros
y0or1 = zeros(y)
@fact_throws fit(Hurdle,GammaLassoPath,X,y0or1)

# degenerate positive counts data case 2
include(joinpath(testfolder,"data","degenerate_hurdle_2.jl"))
hurdle = fit(Hurdle,GammaLassoPath,X,y)
coefsJ=vcat(coef(hurdle;select=:AICc)...)
@fact vec(coefsJ) --> roughly([0.0,0.0,-5.30128195796556,0.1854148891565171]'',1e-4)
coefsJpos, coefsJzero = coef(hurdle;select=:all)
@fact size(coefsJpos,1) --> 2
@fact coefsJpos --> zeros(coefsJpos)
@fact size(coefsJzero,1) --> 2

# degenerate positive counts data case 3
include(joinpath(testfolder,"data","degenerate_hurdle_3.jl"))
hurdle = fit(Hurdle,GammaLassoPath,X,y)
coefsJ=vcat(coef(hurdle;select=:AICc)...)
@fact vec(coefsJ) --> roughly([0.0,0.0,-4.541820686620407,0.0]'',1e-4)
coefsJpos, coefsJzero = coef(hurdle;select=:all)
@fact size(coefsJpos,1) --> 2
@fact coefsJpos --> zeros(coefsJpos)
@fact size(coefsJzero,1) --> 2

end

# TODO never got this to work:
# bioChemists[:art]=convert(Vector{Float64},bioChemists[:art])
# hurdlefit = fit(Hurdle,GammaLassoPath,@formula(art ~ fem + mar + kid5 + phd + ment), bioChemists;intercept=false)
# coefsJ=coef(hurdlefit)
# @fact coefsJ --> roughly(coefsR1;rtol=1e-6)
# rdist(coefsR1,coefsJ)
