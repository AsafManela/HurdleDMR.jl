# reusing data loaded by test/hurdle.jl
X=convert(Array{Float64,2},bioChemists[[:femWomen,:marMarried,:kid5,:phd,:ment]])
Xwconst=[ones(size(X,1)) X]
y=convert(Array{Float64,1},bioChemists[:art])
p = size(Xwconst,2)

###########################################################
# Xpos == Xzero
###########################################################
@testset "inrep with Xpos == Xzero" begin

coefsR1=vec(convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","hurdle_coefsR1.csv"))))
yhatR1=vec(convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","hurdle_yhatR1.csv"))))
yhatR1partial=vec(convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","hurdle_yhatR1partial.csv"))))

# simple inrep with GLM underlying
increpfit = fit(InclusionRepetition,GeneralizedLinearModel,Xwconst,y)
showres = IOBuffer()
show(showres, increpfit)
showstr = String(take!(copy(showres)))
@test occursin("InclusionRepetition regression",showstr)
@test occursin("Positive part coefficients",showstr)
@test occursin("Zero part coefficients",showstr)

coefsJ=vcat(coef(increpfit)...)
@test coefsJ[1+p:end] ≈ coefsR1[1+p:end] rtol=1e-6
yhatJ=predict(increpfit, Xwconst)
@test yhatJ ≈ yhatR1 rtol=1e-2
yhatJpartial=predict(increpfit, Xwconst[ixpartial,:])
@test yhatJpartial ≈ yhatR1partial rtol=1e-2

# same simple inrep through UNREGULATED lasso path
increpglrfit = fit(InclusionRepetition,GammaLassoPath,X,y;λ=[0.0, 0.01])
coefsJ=vcat(coef(increpglrfit;select=MinAICc())...)
@test coefsJ[1+p:end] ≈ coefsR1[1+p:end] rtol=1e-4
# rdist(coefsJ,coefsR1)
yhatJ = predict(increpglrfit, X; select=MinAICc())
@test yhatJ ≈ yhatR1 rtol=1e-2
yhatJpartial=predict(increpglrfit, X[ixpartial,:]; select=MinAICc())
@test yhatJpartial ≈ yhatR1partial rtol=1e-2
yhatJ = predict(increpglrfit, X; select=AllSeg())
@test size(yhatJ) == (size(X,1),2)

# regulated lasso path
increpglrfit = fit(InclusionRepetition,GammaLassoPath,X,y; γ=0.0)
coefsJ=vcat(coef(increpglrfit;select=MinAICc())...)
@test coefsJ[1+p:end] ≈ coefsR1[1+p:end] rtol=0.4
# rdist(coefsJ[1+p:end],coefsR1[1+p:end])
yhatJ = predict(increpglrfit, X; select=MinAICc())
@test yhatJ ≈ yhatR1 rtol=0.05
yhatJpartial=predict(increpglrfit, X[ixpartial,:]; select=MinAICc())
@test yhatJpartial ≈ yhatR1partial rtol=0.05

coefsJpos, coefsJzero = coef(increpglrfit;select=AllSeg())
@test size(coefsJpos,1) == 6
@test size(coefsJzero,1) == 6

showres = IOBuffer()
show(showres, increpglrfit)
showstr = String(take!(copy(showres)))
@test occursin("InclusionRepetition regression",showstr)
@test occursin("Positive part regularization path",showstr)
@test occursin("Zero part regularization path",showstr)

# this one throws an error because we did not specify the same λ vector for both submodels so they have different lengths
@test_throws AssertionError predict(increpglrfit, X; select=AllSeg())

Random.seed!(1)
coefsJCVmin=vcat(coef(increpglrfit, MinCVKfold{MinCVmse}(5))...)
@test coefsJCVmin[1+p:end] ≈ coefsR1[1+p:end] rtol=0.30
# rdist(coefsJCVmin,coefsR1)

# try with SharedArray
Xs = convert(SharedArray,X)
increpglrfitShared = fit(InclusionRepetition,GammaLassoPath,Xs,y; γ=0.0)
coefsJShared=vcat(coef(increpglrfitShared;select=MinAICc())...)
@test coefsJShared ≈ coefsJ
yhatJ = predict(increpglrfitShared, X; select=MinAICc())
@test yhatJ ≈ yhatR1 rtol=0.05
yhatJpartial=predict(increpglrfitShared, X[ixpartial,:]; select=MinAICc())
@test yhatJpartial ≈ yhatR1partial rtol=0.05

# regulated gamma lasso path
increpglrfit = fit(InclusionRepetition,GammaLassoPath,X,y; γ=10.0)
coefsJ=vcat(coef(increpglrfit;select=MinAICc())...)
@test coefsJ[1+p:end] ≈ coefsR1[1+p:end] rtol=0.2
# rdist(coefsJ[1+p:end],coefsR1[1+p:end])
yhatJ = predict(increpglrfit, X; select=MinAICc())
@test yhatJ ≈ yhatR1 rtol=0.05
yhatJpartial=predict(increpglrfit, X[ixpartial,:]; select=MinAICc())
@test yhatJpartial ≈ yhatR1partial rtol=0.05

# using ProfileView
# Profile.init(delay=0.001)
# Profile.clear()
# @profile increpglrfit2 = fit2(InclusionRepetition,GammaLassoPath,X,y; γ=10.0);
# ProfileView.view()
# Profile.print()

# using DataFrames formula interface
increpfit = fit(InclusionRepetition,GeneralizedLinearModel,
    @formula(art ~ femWomen + marMarried + kid5 + phd + ment), bioChemists)
coefsJ=vcat(coef(increpfit)...)
@test coefsJ[1+p:end] ≈ coefsR1[1+p:end] rtol=1e-6
yhatJ=predict(increpfit, Xwconst)
@test yhatJ ≈ yhatR1 rtol=1e-2
yhatJpartial=predict(increpfit, Xwconst[ixpartial,:])
@test yhatJpartial ≈ yhatR1partial rtol=1e-2

end

###########################################################
# Xpos ≠ Xzero
###########################################################
@testset "inrep with Xpos ≠ Xzero" begin

coefsR2=vec(convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","hurdle_coefsR2.csv"))))
yhatR2=vec(convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","hurdle_yhatR2.csv"))))
yhatR2partial=vec(convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","hurdle_yhatR2partial.csv"))))

Xpos = X[:,1:3]
Xzero = X[:,4:5]
Xzerowconst=[ones(size(X,1)) Xzero]
Xposwconst=[ones(size(X,1)) Xpos]
ppos = size(Xposwconst,2)
# ixpos = y.>0
# ypos = y[ixpos]
# countmap(ypos)

# simple inrep with GLM underlying
increpfit = fit(InclusionRepetition,GeneralizedLinearModel,Xzerowconst,y; Xpos=Xposwconst)
coefsJ=vcat(coef(increpfit)...)
@test coefsJ[1+ppos:end] ≈ coefsR2[1+ppos:end] rtol=1e-5
yhatJ=predict(increpfit, Xzerowconst; Xpos=Xposwconst)
@test yhatJ ≈ yhatR2 rtol=1e-2
yhatJpartial=predict(increpfit, Xzerowconst[ixpartial,:]; Xpos=Xposwconst[ixpartial,:])
@test yhatJpartial ≈ yhatR2partial rtol=1e-2

# same simple inrep through UNREGULATED lasso path
increpglrfit = fit(InclusionRepetition,GammaLassoPath,Xzero,y; Xpos=Xpos, λ=[0.0,0.0001])
coefsJ=vcat(coef(increpglrfit;select=MinAICc())...)
@test coefsJ[1+ppos:end] ≈ coefsR2[1+ppos:end] rtol=1e-4
# rdist(coefsJ,coefsR2)
yhatJ=predict(increpglrfit, Xzero; Xpos=Xpos, select=MinAICc())
@test yhatJ ≈ yhatR2 rtol=1e-2
yhatJpartial=predict(increpglrfit, Xzero[ixpartial,:]; Xpos=Xpos[ixpartial,:], select=MinAICc())
@test yhatJpartial ≈ yhatR2partial rtol=1e-2
yhatJ = predict(increpglrfit, Xzero; Xpos=Xpos, select=AllSeg())
@test size(yhatJ) == (size(X,1),2)

# regulated lasso path
increpglrfit = fit(InclusionRepetition,GammaLassoPath,Xzero,y; Xpos=Xpos, γ=0.0)
coefsJ=vcat(coef(increpglrfit;select=MinAICc())...)
@test coefsJ[1+ppos:end] ≈ coefsR2[1+ppos:end] rtol=0.3
# rdist(coefsJ,coefsR2)
yhatJ = predict(increpglrfit, Xzero; Xpos=Xpos, select=MinAICc())
@test yhatJ ≈ yhatR2 rtol=0.05
yhatJpartial=predict(increpglrfit, Xzero[ixpartial,:]; Xpos=Xpos[ixpartial,:], select=MinAICc())
@test yhatJpartial ≈ yhatR2partial rtol=0.05

coefsJpos, coefsJzero = coef(increpglrfit;select=AllSeg())
@test size(coefsJpos,1) == 4
@test size(coefsJzero,1) == 3

# try with SharedArray
Xzeros = convert(SharedArray,Xzero)
Xposs = convert(SharedArray,Xpos)
increpglrfitShared = fit(InclusionRepetition,GammaLassoPath,Xzeros,y; Xpos=Xposs, γ=0.0)
coefsJShared=vcat(coef(increpglrfitShared;select=MinAICc())...)
@test coefsJShared ≈ coefsJ
yhatJ = predict(increpglrfitShared, Xzeros; Xpos=Xposs, select=MinAICc())
@test yhatJ ≈ yhatR2 rtol=0.05
yhatJpartial=predict(increpglrfitShared, Xzeros[ixpartial,:]; Xpos=Xposs[ixpartial,:], select=MinAICc())
@test yhatJpartial ≈ yhatR2partial rtol=0.05

# regulated gamma lasso path
increpglrfit = fit(InclusionRepetition,GammaLassoPath,Xzero,y; Xpos=Xpos, γ=10.0)
coefsJ=vcat(coef(increpglrfit;select=MinAICc())...)
@test coefsJ[1+ppos:end] ≈ coefsR2[1+ppos:end] rtol=0.3
# rdist(coefsJ,coefsR2)
yhatJ = predict(increpglrfit, Xzero; Xpos=Xpos, select=MinAICc())
@test yhatJ ≈ yhatR2 rtol=0.05
yhatJpartial=predict(increpglrfit, Xzero[ixpartial,:]; Xpos=Xpos[ixpartial,:], select=MinAICc())
@test yhatJpartial ≈ yhatR2partial rtol=0.05

# using DataFrames formula interface
increpfit = fit(InclusionRepetition,GeneralizedLinearModel,@formula(art ~ phd + ment), bioChemists; fpos = @formula(art ~ femWomen + marMarried + kid5))
coefsJ=vcat(coef(increpfit)...)
@test coefsJ[1+ppos:end] ≈ coefsR2[1+ppos:end] rtol=1e-5
# rdist(coefsJ,coefsR2)
yhatJ=predict(increpfit, Xzerowconst; Xpos=Xposwconst)
@test yhatJ ≈ yhatR2 rtol=1e-3
yhatJpartial=predict(increpfit, Xzerowconst[ixpartial,:]; Xpos=Xposwconst[ixpartial,:])
@test yhatJpartial ≈ yhatR2partial rtol=1e-3

end

###########################################################
# Xpos ≠ Xzero
###########################################################
@testset "inrep with Xpos ≠ Xzero AND offset specified" begin

coefsR3=vec(convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","hurdle_coefsR3.csv"))))
yhatR3=vec(convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","hurdle_yhatR3.csv"))))
yhatR3partial=vec(convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","hurdle_yhatR3partial.csv"))))

offpos = convert(Vector{Float64},bioChemists[:offpos])
offzero = convert(Vector{Float64},bioChemists[:offzero])
Xpos = X[:,1:3]
Xzero = X[:,4:5]
Xzerowconst=[ones(size(X,1)) Xzero]
Xposwconst=[ones(size(X,1)) Xpos]
ppos = size(Xposwconst, 2)
# ixpos = y.>0
# ypos = y[ixpos]
# countmap(ypos)

# simple inrep with GLM underlying
increpfit = fit(InclusionRepetition,GeneralizedLinearModel,Xzerowconst,y; Xpos=Xposwconst, offsetzero=offzero, offsetpos=offpos)
coefsJ=vcat(coef(increpfit)...)
@test coefsJ[1+ppos:end] ≈ coefsR3[1+ppos:end] rtol=1e-5
yhatJ=predict(increpfit, Xzerowconst; Xpos=Xposwconst, offsetzero=offzero, offsetpos=offpos)
@test yhatJ ≈ yhatR3 rtol=0.1
yhatJpartial=predict(increpfit, Xzerowconst[ixpartial,:]; Xpos=Xposwconst[ixpartial,:], offsetzero=offzero[ixpartial], offsetpos=offpos[ixpartial])
@test yhatJpartial ≈ yhatR3partial rtol=0.2

# same simple inrep through UNREGULATED lasso path
increpglrfit = fit(InclusionRepetition,GammaLassoPath,Xzero,y; Xpos=Xpos, offsetzero=offzero, offsetpos=offpos, λ=[0.0,0.0001])
coefsJ=vcat(coef(increpglrfit;select=MinAICc())...)
@test coefsJ[1+ppos:end] ≈ coefsR3[1+ppos:end] rtol=1e-4
# rdist(coefsJ,coefsR3)
yhatJ=predict(increpglrfit, Xzero; Xpos=Xpos, offsetzero=offzero, offsetpos=offpos, select=MinAICc())
@test yhatJ ≈ yhatR3 rtol=0.1
yhatJpartial=predict(increpglrfit, Xzero[ixpartial,:]; Xpos=Xpos[ixpartial,:], offsetzero=offzero[ixpartial], offsetpos=offpos[ixpartial], select=MinAICc())
@test yhatJpartial ≈ yhatR3partial rtol=0.2
yhatJ = predict(increpglrfit, Xzero; Xpos=Xpos, offsetzero=offzero, offsetpos=offpos, select=AllSeg())
@test size(yhatJ) == (size(X,1),2)

# regulated lasso path
increpglrfit = fit(InclusionRepetition,GammaLassoPath,Xzero,y; Xpos=Xpos, offsetzero=offzero, offsetpos=offpos, γ=0.0)
coefsJ=vcat(coef(increpglrfit;select=MinAICc())...)
@test coefsJ[1+ppos:end] ≈ coefsR3[1+ppos:end] rtol=0.10
# rdist(coefsJ,coefsR3)
yhatJ = predict(increpglrfit, Xzero; Xpos=Xpos, offsetzero=offzero, offsetpos=offpos, select=MinAICc())
@test yhatJ ≈ yhatR3 rtol=0.1
yhatJpartial=predict(increpglrfit, Xzero[ixpartial,:]; Xpos=Xpos[ixpartial,:], offsetzero=offzero[ixpartial], offsetpos=offpos[ixpartial], select=MinAICc())
@test yhatJpartial ≈ yhatR3partial rtol=0.2

coefsJpos, coefsJzero = coef(increpglrfit;select=AllSeg())
@test size(coefsJpos,1) == 4
@test size(coefsJzero,1) == 3

# try with SharedArray
Xzeros = convert(SharedArray,Xzero)
Xposs = convert(SharedArray,Xpos)
offzeros = convert(SharedArray,offzero)
offposs = convert(SharedArray,offpos)
increpglrfitShared = fit(InclusionRepetition,GammaLassoPath,Xzeros,y; Xpos=Xposs, offsetzero=offzeros, offsetpos=offposs, γ=0.0)
coefsJShared=vcat(coef(increpglrfitShared;select=MinAICc())...)
@test coefsJShared ≈ coefsJ
yhatJShared = predict(increpglrfitShared, Xzeros; Xpos=Xposs, offsetzero=offzeros, offsetpos=offposs, select=MinAICc())
@test yhatJShared ≈ yhatJ
yhatJSharedpartial=predict(increpglrfitShared, Xzeros[ixpartial,:]; Xpos=Xposs[ixpartial,:], offsetzero=offzeros[ixpartial], offsetpos=offposs[ixpartial], select=MinAICc())
@test yhatJSharedpartial ≈ yhatJpartial

# regulated gamma lasso path
increpglrfit = fit(InclusionRepetition,GammaLassoPath,Xzero,y; Xpos=Xpos, offsetzero=offzero, offsetpos=offpos, γ=10.0)
coefsJ=vcat(coef(increpglrfit;select=MinAICc())...)
@test coefsJ[1+ppos:end] ≈ coefsR3[1+ppos:end] rtol=0.10
# rdist(coefsJ,coefsR3)
yhatJ = predict(increpglrfit, Xzero; Xpos=Xpos, offsetzero=offzero, offsetpos=offpos, select=MinAICc())
@test yhatJ ≈ yhatR3 rtol=0.1
yhatJpartial=predict(increpglrfit, Xzero[ixpartial,:]; Xpos=Xpos[ixpartial,:], offsetzero=offzero[ixpartial], offsetpos=offpos[ixpartial], select=MinAICc())
@test yhatJpartial ≈ yhatR3partial rtol=0.2

# using DataFrames formula interface
increpfit = fit(InclusionRepetition,GeneralizedLinearModel,@formula(art ~ phd + ment), bioChemists; fpos = @formula(art ~ femWomen + marMarried + kid5), offsetzero=offzero, offsetpos=offpos)
coefsJ=vcat(coef(increpfit)...)
@test coefsJ[1+ppos:end] ≈ coefsR3[1+ppos:end] rtol=1e-5
# rdist(coefsJ,coefsR3)
yhatJ=predict(increpfit, Xzerowconst; Xpos=Xposwconst, offsetzero=offzero, offsetpos=offpos)
@test yhatJ ≈ yhatR3 rtol=0.1
yhatJpartial=predict(increpfit, Xzerowconst[ixpartial,:]; Xpos=Xposwconst[ixpartial,:], offsetzero=offzero[ixpartial], offsetpos=offpos[ixpartial])
@test yhatJpartial ≈ yhatR3partial rtol=0.2

end

###########################################################
# degenrate cases
###########################################################
@testset "inrep degenerate cases" begin

# degenerate positive counts data case 1
include(joinpath(testdir,"data","degenerate_hurdle_1.jl"))
inrep = fit(InclusionRepetition,GammaLassoPath,X,y; showwarnings=true)
coefsJ=vcat(coef(inrep;select=MinAICc())...)
@test vec(coefsJ) ≈ [-4.8941, 0.0, -6.04112, 0.675767] rtol=1e-4
coefsJpos, coefsJzero = coef(inrep;select=AllSeg())
@test size(coefsJpos,1) == 2
@test size(coefsJzero,1) == 2

ydeg = zero(y)
ydeg[1] = 2
ydeg[2] = 3
increpglm = @test_logs (:warn, r"failed to fit truncated counts model to positive") fit(InclusionRepetition,GeneralizedLinearModel,[ones(size(X,1)) X],ydeg; showwarnings=true)


# degenerate positive counts data case 1 without >1
y0or1 = deepcopy(y)
y0or1[y.>1] .= 1
inrep = @test_logs (:warn, r"ypos has no elements larger than 0") fit(InclusionRepetition,GammaLassoPath,X,y0or1; showwarnings=true)
coefs0or1=vcat(coef(inrep;select=MinAICc())...)
@test coefs0or1 ≈ [0.0, 0.0, -6.04112, 0.675767] rtol=1e-4
coefs0or1pos, coefs0or1zero = coef(inrep;select=AllSeg())
@test all(iszero,coefs0or1pos)
@test coefs0or1zero == coefsJzero

# degenerate positive counts data case 1 without zeros
y0or1 = deepcopy(y)
y0or1[y.==0] .= 1
y0or1[1] = 3.0

inrep = @test_logs (:warn, r"I\(y\) is all ones") fit(InclusionRepetition,GammaLassoPath,X,y0or1; verbose=true, showwarnings=true)
# coefs0or1=vcat(coef(inrep;select=MinAICc())...)
coefs0or1pos, coefs0or1zero = coef(inrep;select=MinAICc())
@test coefs0or1pos ≈ [-10.2903, 0.820934] rtol=0.1
@test all(iszero,coefs0or1zero)
coefs0or1pos, coefs0or1zero = coef(inrep;select=AllSeg())
@test all(iszero,coefs0or1zero)

increpglm = @test_logs (:warn, r"I\(y\) is all ones") fit(InclusionRepetition,GeneralizedLinearModel,[ones(size(X,1)) X],y0or1; verbose=true, showwarnings=true)

# degenerate positive counts data case 1 with only zeros
y0or1 = zero(y)
@test_throws ErrorException fit(InclusionRepetition,GammaLassoPath,X,y0or1)
@test_logs (:warn, r"I\(y\) is all zeros") @test_throws ErrorException fit(InclusionRepetition,GammaLassoPath,X,y0or1; verbose=true, showwarnings=true)

# degenerate positive counts data case 2
include(joinpath(testdir,"data","degenerate_hurdle_2.jl"))
inrep = fit(InclusionRepetition,GammaLassoPath,X,y; showwarnings=true)
coefsJ=vcat(coef(inrep;select=MinAICc())...)
@test vec(coefsJ) ≈ [-3.26059, -0.220401, -5.3013, 0.185418] rtol=1e-4
coefsJpos, coefsJzero = coef(inrep;select=AllSeg())
@test size(coefsJpos,1) == 2
@test size(coefsJzero,1) == 2

# degenerate positive counts data case 3
include(joinpath(testdir,"data","degenerate_hurdle_3.jl"))
inrep = @test_logs (:warn, r"ypos has no elements larger than 0") fit(InclusionRepetition,GammaLassoPath,X,y; showwarnings=true)
coefsJ=vcat(coef(inrep;select=MinAICc())...)
# @test vec(coefsJ) ≈ [0.0, 0.0, -4.54363, 0.000458273] rtol=1e-4
@test length(coefsJ) == 4
coefsJpos, coefsJzero = coef(inrep;select=AllSeg())
Matrix( coefsJzero)
@test size(coefsJpos,1) == 2
@test coefsJpos == zero(coefsJpos)
@test size(coefsJzero,1) == 2

# this test case used to give numerical headaches to devresid(PositivePoisson(),...)
include(joinpath(testdir,"data","degenerate_hurdle_5.jl"))
inrep = fit(InclusionRepetition,GammaLassoPath,X,y; Xpos=Xpos, offset=offset)
@test !(ismissing(inrep.mpos))
@test !(ismissing(inrep.mzero))
@test typeof(inrep.mpos) <: GammaLassoPath
@test typeof(inrep.mzero) <: GammaLassoPath

end
