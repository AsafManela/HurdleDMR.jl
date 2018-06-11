include("testutils.jl")

using Base.Test, Gadfly, Distributions

include("addworkers.jl")

using CSV, GLM, DataFrames, LassoPlot

import HurdleDMR; @everywhere using HurdleDMR

# uncomment following for debugging and comment the previous @everywhere line. then use reload after making changes
# reload("HurdleDMR")

γ = 1.0

# # uncomment to generate R benchmark
# using RCall
# R"library(textir)"
# R"library(Matrix)"
# R"data(we8there)"
# R"we8thereCounts <- we8thereCounts[,1:100]" # comment this line to use entire dataset
# we8thereCounts = DataFrame(rcopy(R"as.matrix(we8thereCounts)"))
# we8thereRatings = rcopy(R"we8thereRatings")
# we8thereTerms = rcopy(R"we8thereCounts@Dimnames$Terms")
# names!(we8thereCounts,map(Symbol,we8thereTerms))
#
# R"cl <- makeCluster(2,type=\"FORK\")"
# R"fits <- dmr(cl, we8thereRatings, we8thereCounts, gamma=$γ, verb=0)"
# R"stopCluster(cl)"
# coefsRdistrom = rcopy(R"as.matrix(coef(fits))")
# zRdistrom = rcopy(R"as.matrix(srproj(fits,we8thereCounts))")
# z1Rdistrom = rcopy(R"as.matrix(srproj(fits,we8thereCounts,1))")
#
# CSV.write(joinpath(testdir,"data","dmr_we8thereCounts.csv.gz"),we8thereCounts)
# CSV.write(joinpath(testdir,"data","dmr_we8thereRatings.csv.gz"),we8thereRatings)
# CSV.write(joinpath(testdir,"data","dmr_coefsRdistrom.csv.gz"),DataFrame(full(coefsRdistrom)))
# CSV.write(joinpath(testdir,"data","dmr_zRdistrom.csv.gz"),DataFrame(zRdistrom))
# CSV.write(joinpath(testdir,"data","dmr_z1Rdistrom.csv.gz"),DataFrame(z1Rdistrom))

we8thereCounts = CSV.read(joinpath(testdir,"data","dmr_we8thereCounts.csv.gz"))
we8thereRatings = CSV.read(joinpath(testdir,"data","dmr_we8thereRatings.csv.gz"))
we8thereTerms = broadcast(string,names(we8thereCounts))
coefsRdistrom = sparse(convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","dmr_coefsRdistrom.csv.gz"))))
zRdistrom = convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","dmr_zRdistrom.csv.gz")))
z1Rdistrom = convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","dmr_z1Rdistrom.csv.gz")))

# covars = we8thereRatings[:,[:Overall]]
covars = we8thereRatings[:,:]
n,p = size(covars)
# inzero = 1:p
#
# inpos = [1,3]
# covarspos = we8thereRatings[:,inpos]

T = Float64
counts=sparse(convert(Matrix{Float64},we8thereCounts))

# use smaller counts matrix when not comparing with distrom
smalld = 100
srand(13)
smallcounts = round.(10*sprand(n,smalld,0.3))

covars=convert(Array{T,2},covars)
# covarspos=convert(Array{T,2},covarspos)
#
# npos,ppos = size(covarspos)
d = size(counts,2)

@testset "dmr" begin

# to get exactly Rdistrom's run use λminratio=0.01 because our gamlr's have different defaults
@time dmrcoefs = dmr(covars, counts; γ=γ, λminratio=0.01)
coefs = coef(dmrcoefs)
@test size(coefs) == (p+1, d)

@time dmrcoefsb = fit(DMRCoefs, covars, counts; γ=γ, λminratio=0.01)
@test coef(dmrcoefs) == coefs
@time dmrb = fit(DMR, covars, counts; γ=γ, λminratio=0.01)
@test coef(dmrb) == coefs
@test n > nobs(dmrcoefs)
@test d == ncategories(dmrcoefs)
@test p == ncovars(dmrcoefs)

@time dmrcoefs2 = dmr(covars, counts; local_cluster=false, γ=γ, λminratio=0.01)
coefs2 = coef(dmrcoefs2)
@test coefs ≈ coefs2
@time dmrcoefs2 = fit(DMRCoefs, covars, counts; local_cluster=false, γ=γ, λminratio=0.01)
@test coef(dmrcoefs2) == coefs2

@time dmrPaths = dmrpaths(covars, counts; γ=γ, λminratio=0.01, verbose=false)
@time dmrPaths2 = fit(DMRPaths, covars, counts; γ=γ, λminratio=0.01, verbose=false)
@test coef(dmrPaths) == coef(dmrPaths2)
@test n > nobs(dmrPaths)
@test d == ncategories(dmrPaths)
@test p == ncovars(dmrPaths)

paths = dmrPaths.nlpaths
# these terms require the full counts data, but we use only first 100 for quick testing
# plotterms=["first_date","chicken_wing","ate_here", "good_food","food_fabul","terribl_servic"]
plotterms=we8thereTerms[1:6]
plotix=[find(we8thereTerms.==term)[1]::Int64 for term=plotterms]
plotterms==we8thereTerms[plotix]
plots=permutedims(convert(Matrix{Gadfly.Plot},reshape([plot(paths[plotix[i]].value,Guide.title(plotterms[i]);select=:AICc,x=:logλ) for i=1:length(plotterms)],2,3)), [2,1])
filename = joinpath(testdir,"plots","we8there.svg")
# # TODO: uncomment after Gadfly get's its get_stroke_vector bug fixed
# draw(SVG(filename,9inch,11inch),Gadfly.gridstack(plots))
# @test isfile(filename)

@time z = srproj(coefs, counts)
@time zb = srproj(dmrcoefs, counts)
@time zc = srproj(dmrPaths, counts)
@test z == zb
@test z ≈ zc
@test size(z) == (size(counts,1),p+1)

regdata = DataFrame(y=covars[:,1], z=z[:,1], m=z[:,2])
lm1 = lm(@formula(y ~ z+m), regdata)
r21 = adjr2(lm1)

@test coefs ≈ full(coefsRdistrom) rtol=rtol
# println("rdist(coefs,coefsRdistrom)=$(rdist(coefs,coefsRdistrom))")

@test z ≈ zRdistrom rtol=rtol
# println("rdist(z,zRdistrom)=$(rdist(z,zRdistrom))")

regdata3 = DataFrame(y=covars[:,1], z=zRdistrom[:,1], m=zRdistrom[:,2])
lm3 = lm(@formula(y ~ z+m), regdata3)
r23 = adjr2(lm3)

@test r21 ≈ r23 rtol=rtol
# println("rdist(r21,r23)=$(rdist(r21,r23))")

# Rdistrom.dmrplots(fits.gamlrs[plotix],we8thereTerms[plotix])

# project in a single direction
@time z1 = srproj(coefs, counts, 1)
@time z1b = srproj(dmrcoefs, counts, 1)
@time z1c = srproj(coefs, counts, 1)
@test z1 ≈ z[:,[1,end]]

@time z1dense = srproj(coefs, full(counts), 1)
@time z1denseb = srproj(dmrcoefs, full(counts), 1)
@time z1densec = srproj(dmrPaths, full(counts), 1)
@test z1dense == z1denseb
@test z1dense ≈ z1densec
@test z1dense ≈ z1

@test z1 ≈ z1Rdistrom rtol=rtol

X1, X1_nocounts, inz = srprojX(coefs,counts,covars,1; includem=true)
@test X1_nocounts == [ones(n,1) covars[:,2:end]]
@test X1 == [X1_nocounts z1]
@test inz == [1]

X1b, X1_nocountsb, inz = srprojX(dmrcoefs,counts,covars,1; includem=true)
@test X1 == X1b
@test X1_nocounts == X1_nocountsb
@test inz == [1]

X2, X2_nocounts, inz = srprojX(coefs,counts,covars,1; includem=false)
@test X2_nocounts == X1_nocounts
@test X2 == X1[:,1:end-1]
@test inz == [1]

X2b, X2_nocountsb, inz = srprojX(dmrcoefs,counts,covars,1; includem=false)
@test X2 == X2b
@test X2_nocounts == X2_nocountsb
@test inz == [1]

# project in a single direction, focusing only on cj
focusj = 3
@time z1j = srproj(coefs, counts, 1; focusj=focusj)
@time z1jdense = srproj(coefs, full(counts), 1; focusj=focusj)
@test z1jdense == z1j

X1j, X1j_nocounts, inz = srprojX(coefs,counts,covars,1; includem=true, focusj=focusj)
@test X1j_nocounts == [ones(n,1) covars[:,2:end]]
@test X1j == [X1_nocounts z1j]
@test inz == [1]

X2j, X2j_nocounts, inz = srprojX(coefs,counts,covars,1; includem=false, focusj=focusj)
@test X2j_nocounts == X1_nocounts
@test X2j == X1j[:,1:end-1]
@test inz == [1]

X1jfull, X1jfull_nocounts, inz = srprojX(coefs,full(counts),covars,1; includem=true, focusj=focusj)
@test X1jfull ≈ X1j
@test X1jfull_nocounts ≈ X1jfull_nocounts
@test inz == [1]

X2jfull, X2jfull_nocounts, inz = srprojX(coefs,full(counts),covars,1; includem=false, focusj=focusj)
@test X2jfull ≈ X2j
@test X2jfull_nocounts ≈ X2jfull_nocounts
@test inz == [1]

# MNIR
# using Juno
# Juno.@enter fit(CIR{DMR,LinearModel},covars,counts,1)
@time mnir = fit(CIR{DMR,LinearModel},covars,counts,1; γ=γ, λminratio=0.01, nocounts=true)
@test coefbwd(mnir) ≈ coef(dmrcoefs)

zlm = lm(hcat(ones(n,1),z1,covars[:,2:end]),covars[:,1])
@test r2(zlm) ≈ r2(mnir)
@test adjr2(zlm) ≈ adjr2(mnir)
@test predict(zlm,hcat(ones(10,1),z1[1:10,:],covars[1:10,2:end])) ≈ predict(mnir,covars[1:10,:],counts[1:10,:])

zlmnocounts = lm(hcat(ones(n,1),covars[:,2:end]),covars[:,1])
@test r2(zlmnocounts) ≈ r2(mnir; nocounts=true)
@test adjr2(zlmnocounts) ≈ adjr2(mnir; nocounts=true)
@test predict(zlmnocounts,hcat(ones(10,1),covars[1:10,2:end])) ≈ predict(mnir,covars[1:10,:],counts[1:10,:]; nocounts=true)

# CV
@time cvstats13 = cv(CIR{DMR,LinearModel},covars,smallcounts,1; k=2, gentype=MLBase.Kfold, γ=γ)
@time cvstats13b = cv(CIR{DMR,LinearModel},covars,smallcounts,1; k=2, gentype=MLBase.Kfold, γ=γ)
@test isequal(cvstats13,cvstats13b)

cvstats14 = cv(CIR{DMR,LinearModel},covars,smallcounts,1; k=2, gentype=MLBase.Kfold, γ=γ, seed=14)
@test !(isequal(cvstats13,cvstats14))

@time cvstatsSerialKfold = cv(CIR{DMR,LinearModel},covars,smallcounts,1; k=5, gentype=SerialKfold, γ=γ)

end

#########################################################################3
# degenerate cases
#########################################################################3

@testset "dmr degenerate cases" begin

# always one (zero var) counts columns
zcounts = deepcopy(smallcounts)
zcounts[:,2] = zeros(n)
zcounts[:,3] = ones(n)
find(var(zcounts,1) .== 0)

# make sure we are not adding all zero obseravtions
m = sum(zcounts,2)
@test sum(m .== 0) == 0

# this one should warn on dimension 2
@time dmrzcoefs = dmr(covars, zcounts; γ=γ, λminratio=0.01, showwarnings=true)
zcoefs = coef(dmrzcoefs)
@test size(zcoefs) == (p+1, smalld)
@test zcoefs[:,2] ≈ zeros(p+1)

@time dmrzcoefs2 = dmr(covars, zcounts; local_cluster=false, γ=γ, λminratio=0.01)
zcoefs2 = coef(dmrzcoefs2)
@test zcoefs2 ≈ zcoefs2

end

#########################################################################3
# profile cv
#########################################################################3

# # NOTE: to run this and get the profile, do not add any parallel workers
# # this makes it slow (about 350 secs) and may miss some serialization costs
# # but I don't know how to profile all the workers too ...
# @time cvstats13 = cv(CIR{DMR,LinearModel},covars,counts,1; k=2, gentype=MLBase.Kfold, γ=γ)
# using ProfileView
# Profile.init(delay=0.001)
# Profile.clear()
# @profile cvstats13 = cv(CIR{DMR,LinearModel},covars,counts,1; k=2, gentype=MLBase.Kfold, γ=γ);
# ProfileView.view()
# # ProfileView.svgwrite(joinpath(tempdir(),"profileview.svg"))
# # Profile.print()

#########################################################################3
# chicken wing focus
#########################################################################3
# j = plotix[2]
# path = paths[j]
# pathseg=collect(1:length(path.λ))
# path.λ
# pathaicc=aicc(path)
# pathdf=df(path)
# pathdeviance=deviance(path)*n
# path.λ[1]*0.01
# pathcoef=vec(coef(path)[2,:])
#
# ixinpath = pathseg
#
# gamlrmu=fits.mu
# μ ≈ gamlrmu
#
# gamlr = fits.gamlrs[j]
# gamlrObj = gamlr.Robj
# gamlrseg=collect(1:100)[ixinpath]
# gamlrλ=gamlr.attributes["lambda"][ixinpath]
# gamlraicc=rcopy(R"AICc($gamlrObj)")[ixinpath]
# gamlrdf=gamlr.attributes["df"][ixinpath]
# gamlrdeviance=gamlr.attributes["deviance"][ixinpath]
# gamlr.attributes
# gamlrcoef=vec(full(coef(gamlr)[2,:]))[ixinpath]
#
# gamlrλ[1] ≈ path.λ[1]
# gamlrdeviance[1] ≈ pathdeviance[1]
# gamlrλ[1]*0.01 ≈ gamlrλ[end]
#
# [path.λ gamlrλ[ixinpath]]
# [pathdf gamlrdf[ixinpath]]
# [pathdeviance gamlrdeviance[ixinpath]]
# [pathaicc gamlraicc[ixinpath]]
#
# findmin(pathaicc)
# findmin(gamlraicc)
#
# codes=vcat(repmat([:Julia],length(path.λ)),repmat([:R],length(gamlrλ)))
# seg=vcat(pathseg,gamlrseg)
# λs=vcat(path.λ,gamlrλ)
# aiccs=vcat(pathaicc,gamlraicc)
# dfs=vcat(pathdf,gamlrdf)
# deviances=vcat(pathdeviance,gamlrdeviance)
# coefs = vcat(pathcoef,gamlrcoef)
#
# plotdf = DataFrame(code=codes,seg=seg,λ=λs,logλ=log(λs),aicc=aiccs,df=dfs,deviance=deviances,β=coefs)
# plot(plotdf,x=:logλ,y=:df,color=:code)
# plot(plotdf,x=:logλ,y=:aicc,color=:code)
# plot(plotdf,x=:seg,y=:logλ,color=:code)
# plot(path;selectedvars=:all,x=:logλ)
# meltedplotdf = melt(plotdf,[:code,:λ,:seg,:logλ])
# plot(meltedplotdf,x=:logλ,y=:value,xgroup=:variable,color=:code,
#      Geom.subplot_grid(Geom.line),free_y_axis=true)

# rmprocs(workers())
