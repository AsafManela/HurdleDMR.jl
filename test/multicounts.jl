include("testutils.jl")

using Base.Test, Gadfly, Distributions

include("addworkers.jl")

using CSV, GLM, DataFrames, LassoPlot

import HurdleDMR; @everywhere using HurdleDMR
# uncomment following for debugging and comment the previous @everywhere line. then use reload after making changes
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

@time coefs = HurdleDMR.dmr(covars, multicounts[1]; γ=γ, λminratio=0.01)

@time Z, multicoefs = HurdleDMR.mcdmr(covars, multicounts, 1; γ=γ, λminratio=0.01)

@test size(coefs) == (p+1, d)
@test multicoefs[1] == coefs
@test size(multicoefs[2],2) == size(coefs,2)
@test size(multicoefs[2],1) == size(coefs,1) + 2
@test size(multicoefs[3],2) == size(coefs,2)
@test size(multicoefs[3],1) == size(coefs,1) + 4

# @time coefs2 = HurdleDMR.dmr(covars, counts; local_cluster=false, γ=γ, λminratio=0.01)
# @test coefs ≈ coefs2
#
# @time paths = HurdleDMR.dmrpaths(covars, counts; γ=γ, λminratio=0.01, verbose=false)
#
# plotterms=["first_date","chicken_wing","ate_here", "good_food","food_fabul","terribl_servic"]
# plotix=[find(we8thereTerms.==term)[1]::Int64 for term=plotterms]
# plotterms==we8thereTerms[plotix]
#
# plots=permutedims(convert(Matrix{Gadfly.Plot},reshape([plot(paths[plotix[i]].value,Guide.title(plotterms[i]);select=:AICc,x=:logλ) for i=1:length(plotterms)],2,3)), [2,1])
# plots=permutedims(convert(Matrix{Gadfly.Plot},reshape([plot(paths[plotix[i]].value,Guide.title(plotterms[i]);select=:AICc,x=:logλ) for i=1:length(plotterms)],2,3)), [2,1])
#
# filename = joinpath(testdir,"plots","we8there.svg")
# # # TODO: uncomment after Gadfly get's its get_stroke_vector bug fixed
# # draw(SVG(filename,9inch,11inch),Gadfly.gridstack(plots))
# # @test isfile(filename)
#
# #reload("HurdleDMR")
# @time z = HurdleDMR.srproj(coefs, counts)
# @test size(z) == (size(covars,1),p+1)
#
# regdata = DataFrame(y=covars[:,1], z=z[:,1], m=z[:,2])
# lm1 = lm(@formula(y ~ z+m), regdata)
# r21 = adjr2(lm1)
#
# # @time fits = Rdistrom.dmr(covars, counts; nlocal_workers=nworkers(), gamma=γ, verb=0)
# # coefsRdistrom = Rdistrom.coef(fits)
# # writetable(joinpath(testdir,"data","dmr_coefsRdistrom.csv.gz"),DataFrame(full(coefsRdistrom)))
# coefsRdistrom = sparse(convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","dmr_coefsRdistrom.csv.gz"))))
# @test coefs ≈ full(coefsRdistrom) rtol=2rtol
# # println("rdist(coefs,coefsRdistrom)=$(rdist(coefs,coefsRdistrom))")
#
# # zRdistrom = Rdistrom.srproj(fits,counts)
# # writetable(joinpath(testdir,"data","dmr_zRdistrom.csv.gz"),DataFrame(zRdistrom))
# zRdistrom = convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","dmr_zRdistrom.csv.gz")))
# @test z ≈ zRdistrom rtol=rtol
# # println("rdist(z,zRdistrom)=$(rdist(z,zRdistrom))")
#
# regdata3 = DataFrame(y=covars[:,1], z=zRdistrom[:,1], m=zRdistrom[:,2])
# lm3 = lm(@formula(y ~ z+m), regdata3)
# r23 = adjr2(lm3)
#
# @test r21 ≈ r23 rtol=rtol
# # println("rdist(r21,r23)=$(rdist(r21,r23))")
#
# # Rdistrom.dmrplots(fits.gamlrs[plotix],we8thereTerms[plotix])
#
# # project in a single direction
# @time z1 = HurdleDMR.srproj(coefs, counts, 1)
# @test z1 ≈ z[:,[1,end]]
#
# @time z1dense = HurdleDMR.srproj(coefs, full(counts), 1)
# @test z1dense ≈ z1
#
# # z1Rdistrom = Rdistrom.srproj(fits,counts,1)
# # writetable(joinpath(testdir,"data","dmr_z1Rdistrom.csv.gz"),DataFrame(z1Rdistrom))
# z1Rdistrom = convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","dmr_z1Rdistrom.csv.gz")))
# @test z1 ≈ z1Rdistrom rtol=rtol
#
# X1, X1_nocounts = HurdleDMR.srprojX(coefs,counts,covars,1; includem=true)
# @test X1_nocounts == [ones(n,1) covars[:,2:end]]
# @test X1 == [X1_nocounts z1]
#
# X2, X2_nocounts = HurdleDMR.srprojX(coefs,counts,covars,1; includem=false)
# @test X2_nocounts == X1_nocounts
# @test X2 == X1[:,1:end-1]
#
# @time cvstats13 = HurdleDMR.cross_validate_dmr_srproj(covars,counts,1; k=2, gentype=MLBase.Kfold, γ=γ)
# @time cvstats13b = HurdleDMR.cross_validate_dmr_srproj(covars,counts,1; k=2, gentype=MLBase.Kfold, γ=γ)
# @test isequal(cvstats13,cvstats13b)
#
# cvstats14 = HurdleDMR.cross_validate_dmr_srproj(covars,counts,1; k=2, gentype=MLBase.Kfold, γ=γ, seed=14)
# @test !(isequal(cvstats13,cvstats14))
#
# cvstatsSerialKfold = HurdleDMR.cross_validate_dmr_srproj(covars,counts,1; k=5, gentype=HurdleDMR.SerialKfold, γ=γ)
#
# #########################################################################3
# # profile cv
# #########################################################################3
#
# # # NOTE: to run this and get the profile, do not add any parallel workers
# # # this makes it slow (about 350 secs) and may miss some serialization costs
# # # but I don't know how to profile all the workers too ...
# # @time cvstats13 = HurdleDMR.cross_validate_dmr_srproj(covars,counts,1; k=2, gentype=MLBase.Kfold, γ=γ)
# # using ProfileView
# # Profile.init(delay=0.001)
# # Profile.clear()
# # @profile cvstats13 = HurdleDMR.cross_validate_dmr_srproj(covars,counts,1; k=2, gentype=MLBase.Kfold, γ=γ);
# # ProfileView.view()
# # # ProfileView.svgwrite(joinpath(tempdir(),"profileview.svg"))
# # # Profile.print()
#
# #########################################################################3
# # chicken wing focus
# #########################################################################3
# # j = plotix[2]
# # path = paths[j]
# # pathseg=collect(1:length(path.λ))
# # path.λ
# # pathaicc=aicc(path)
# # pathdf=df(path)
# # pathdeviance=deviance(path)*n
# # path.λ[1]*0.01
# # pathcoef=vec(coef(path)[2,:])
# #
# # ixinpath = pathseg
# #
# # gamlrmu=fits.mu
# # μ ≈ gamlrmu
# #
# # gamlr = fits.gamlrs[j]
# # gamlrObj = gamlr.Robj
# # gamlrseg=collect(1:100)[ixinpath]
# # gamlrλ=gamlr.attributes["lambda"][ixinpath]
# # gamlraicc=rcopy(R"AICc($gamlrObj)")[ixinpath]
# # gamlrdf=gamlr.attributes["df"][ixinpath]
# # gamlrdeviance=gamlr.attributes["deviance"][ixinpath]
# # gamlr.attributes
# # gamlrcoef=vec(full(coef(gamlr)[2,:]))[ixinpath]
# #
# # gamlrλ[1] ≈ path.λ[1]
# # gamlrdeviance[1] ≈ pathdeviance[1]
# # gamlrλ[1]*0.01 ≈ gamlrλ[end]
# #
# # [path.λ gamlrλ[ixinpath]]
# # [pathdf gamlrdf[ixinpath]]
# # [pathdeviance gamlrdeviance[ixinpath]]
# # [pathaicc gamlraicc[ixinpath]]
# #
# # findmin(pathaicc)
# # findmin(gamlraicc)
# #
# # codes=vcat(repmat([:Julia],length(path.λ)),repmat([:R],length(gamlrλ)))
# # seg=vcat(pathseg,gamlrseg)
# # λs=vcat(path.λ,gamlrλ)
# # aiccs=vcat(pathaicc,gamlraicc)
# # dfs=vcat(pathdf,gamlrdf)
# # deviances=vcat(pathdeviance,gamlrdeviance)
# # coefs = vcat(pathcoef,gamlrcoef)
# #
# # plotdf = DataFrame(code=codes,seg=seg,λ=λs,logλ=log(λs),aicc=aiccs,df=dfs,deviance=deviances,β=coefs)
# # plot(plotdf,x=:logλ,y=:df,color=:code)
# # plot(plotdf,x=:logλ,y=:aicc,color=:code)
# # plot(plotdf,x=:seg,y=:logλ,color=:code)
# # plot(path;selectedvars=:all,x=:logλ)
# # meltedplotdf = melt(plotdf,[:code,:λ,:seg,:logλ])
# # plot(meltedplotdf,x=:logλ,y=:value,xgroup=:variable,color=:code,
# #      Geom.subplot_grid(Geom.line),free_y_axis=true)
end

# rmprocs(workers())
