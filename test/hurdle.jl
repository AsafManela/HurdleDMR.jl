testfolder = dirname(@__FILE__)
push!(LOAD_PATH, joinpath(testfolder,".."))
push!(LOAD_PATH, joinpath(testfolder,"..","src"))

using FactCheck, Lasso, DataFrames

rdist(x::Number,y::Number) = abs(x-y)/max(abs(x),abs(y))
rdist{T<:Number,S<:Number}(x::AbstractArray{T}, y::AbstractArray{S}; norm::Function=vecnorm) = norm(x - y) / max(norm(x), norm(y))

using HurdleDMR

# code to generate R benchmark
# using RCall
# R"if(!require(pscl)){install.packages(\"pscl\");library(pscl)}"
# R"library(pscl)"
# R"data(\"bioChemists\", package = \"pscl\")"
# bioChemists=rcopy(R"bioChemists")
# writetable(joinpath(testfolder,"data","bioChemists.csv"),bioChemists)

bioChemists=readtable(joinpath(testfolder,"data","bioChemists.csv"))
bioChemists[:marMarried]=bioChemists[:mar] .== "Married"
bioChemists[:femWomen]=bioChemists[:fem] .== "Women"

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
# writetable(joinpath(testfolder,"data","hurdle_coefsR1.csv"),DataFrame(coefsR=coefsR1))
coefsR1=vec(convert(Matrix{Float64},readtable(joinpath(testfolder,"data","hurdle_coefsR1.csv"))))

# simple hurdle with GLM underlying
hurdlefit = fit(Hurdle,GeneralizedLinearModel,Xwconst,y)
coefsJ=vcat(coef(hurdlefit)...)
@fact coefsJ --> roughly(coefsR1;rtol=1e-6)

# same simple hurdle with through UNREGULATED lasso path
hurdleglrfit = fit(Hurdle,GammaLassoPath,X,y;λ=[0.0],verbose=true)
@time coefsJ=vcat(coef(hurdleglrfit;select=:AICc)...)
@fact coefsJ --> roughly(coefsR1;rtol=1e-4)
# rdist(coefsJ,coefsR1)

# regulated lasso path
hurdleglrfit = fit(Hurdle,GammaLassoPath,X,y; γ=0.0)
coefsJ=vcat(coef(hurdleglrfit;select=:AICc)...)
@fact coefsJ --> roughly(coefsR1;rtol=0.25)
rdist(coefsJ,coefsR1)

coefsJpos, coefsJzero = coef(hurdleglrfit;select=:all)
@fact size(coefsJpos,1) --> 6
@fact size(coefsJzero,1) --> 6

coefsJCVmin=vcat(coef(hurdleglrfit;select=:CVmin)...)
@fact coefsJCVmin --> roughly(coefsR1;rtol=0.30)
rdist(coefsJCVmin,coefsR1)

# try with SharedArray
Xs = convert(SharedArray,X)
hurdleglrfitShared = fit(Hurdle,GammaLassoPath,Xs,y; γ=0.0)
coefsJShared=vcat(coef(hurdleglrfitShared;select=:AICc)...)
@fact coefsJShared --> roughly(coefsJ)

# regulated gamma lasso path
@time hurdleglrfit = fit(Hurdle,GammaLassoPath,X,y; γ=10.0)
coefsJ=vcat(coef(hurdleglrfit;select=:AICc)...)
@fact coefsJ --> roughly(coefsR1;rtol=0.11)
rdist(coefsJ,coefsR1)

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

end

###########################################################
# Xpos ≠ Xzero
###########################################################
facts("hurdle with Xpos ≠ Xzero") do

# regulated gamma lasso path with different Xpos and Xzero
# R"fm_hp2 <- hurdle(art ~ fem + mar + kid5 | phd + ment, data = bioChemists)"
# print(R"summary(fm_hp2)")
# coefsR2=vec(rcopy(R"coef(fm_hp2, matrix = TRUE)"))
# writetable(joinpath(testfolder,"data","hurdle_coefsR2.csv"),DataFrame(coefsR=coefsR2))
coefsR2=vec(convert(Matrix{Float64},readtable(joinpath(testfolder,"data","hurdle_coefsR2.csv"))))

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

# same simple hurdle with through UNREGULATED lasso path
hurdleglrfit = fit(Hurdle,GammaLassoPath,Xzero,y; Xpos=Xpos, λ=[0.0])
coefsJ=vcat(coef(hurdleglrfit;select=:AICc)...)
@fact coefsJ --> roughly(coefsR2;rtol=1e-4)
# rdist(coefsJ,coefsR2)

# regulated lasso path
hurdleglrfit = fit(Hurdle,GammaLassoPath,Xzero,y; Xpos=Xpos, γ=0.0)
coefsJ=vcat(coef(hurdleglrfit;select=:AICc)...)
@fact coefsJ --> roughly(coefsR2;rtol=0.10)
# rdist(coefsJ,coefsR2)

coefsJpos, coefsJzero = coef(hurdleglrfit;select=:all)
@fact size(coefsJpos,1) --> 4
@fact size(coefsJzero,1) --> 3

# try with SharedArray
Xzeros = convert(SharedArray,Xzero)
Xposs = convert(SharedArray,Xpos)
hurdleglrfitShared = fit(Hurdle,GammaLassoPath,Xzeros,y; Xpos=Xposs, γ=0.0)
coefsJShared=vcat(coef(hurdleglrfitShared;select=:AICc)...)
@fact coefsJShared --> roughly(coefsJ)

# regulated gamma lasso path
hurdleglrfit = fit(Hurdle,GammaLassoPath,Xzero,y; Xpos=Xpos, γ=10.0)
coefsJ=vcat(coef(hurdleglrfit;select=:AICc)...)
@fact coefsJ --> roughly(coefsR2;rtol=0.15)
# rdist(coefsJ,coefsR2)

# using DataFrames formula interface
hurdlefit = fit(Hurdle,GeneralizedLinearModel,@formula(art ~ phd + ment), bioChemists; fpos = @formula(art ~ femWomen + marMarried + kid5))
coefsJ=vcat(coef(hurdlefit)...)
@fact coefsJ --> roughly(coefsR2;rtol=1e-5)
# rdist(coefsJ,coefsR2)

end

###########################################################
# degenrate cases
###########################################################
facts("hurdle degenerate cases") do

# degenerate positive counts data case 1
include(joinpath(testfolder,"data","degenerate_hurdle_1.jl"))
hurdle = fit(Hurdle,GammaLassoPath,X,y)
coefsJ=vcat(coef(hurdle;select=:AICc)...)
@fact coefsJ --> roughly([0.0; 0.0; -6.04112; 0.675767]'',1e-4)
coefsJpos, coefsJzero = coef(hurdle;select=:all)
@fact size(coefsJpos,1) --> 2
@fact coefsJpos --> zeros(coefsJpos)
@fact size(coefsJzero,1) --> 2

# degenerate positive counts data case 1 without >1
y0or1=y
y0or1[y.>1]=1
hurdle = fit(Hurdle,GammaLassoPath,X,y0or1)
coefs0or1=vcat(coef(hurdle;select=:AICc)...)
@fact coefs0or1 --> coefsJ
coefs0or1pos, coefs0or1zero = coef(hurdle;select=:all)
@fact coefs0or1pos --> coefsJpos
@fact coefs0or1zero --> coefsJzero

# degenerate positive counts data case 2
include(joinpath(testfolder,"data","degenerate_hurdle_2.jl"))
hurdle = fit(Hurdle,GammaLassoPath,X,y)
coefsJ=vcat(coef(hurdle;select=:AICc)...)
@fact coefsJ --> roughly([0.0,0.0,-5.30128195796556,0.1854148891565171]'',1e-4)
coefsJpos, coefsJzero = coef(hurdle;select=:all)
@fact size(coefsJpos,1) --> 2
@fact coefsJpos --> zeros(coefsJpos)
@fact size(coefsJzero,1) --> 2

# degenerate positive counts data case 3
include(joinpath(testfolder,"data","degenerate_hurdle_3.jl"))
hurdle = fit(Hurdle,GammaLassoPath,X,y)
coefsJ=vcat(coef(hurdle;select=:AICc)...)
@fact coefsJ --> roughly([0.0,0.0,-4.541820686620407,0.0]'',1e-4)
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
