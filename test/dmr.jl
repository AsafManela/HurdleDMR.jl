include("testutils.jl")

using Distributions

include("addworkers.jl")

using CSV, GLM, DataFrames

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
# names!(we8thereCounts,broadcast(Symbol,we8thereTerms))
#
# R"cl <- makeCluster(2,type=\"FORK\")"
# R"fits <- dmr(cl, we8thereRatings, we8thereCounts, gamma=$γ, verb=0)"
# R"stopCluster(cl)"
# coefsRdistrom = rcopy(R"as.matrix(coef(fits))")
# zRdistrom = rcopy(R"as.matrix(srproj(fits,we8thereCounts))")
# z1Rdistrom = rcopy(R"as.matrix(srproj(fits,we8thereCounts,1))")
# predictRdistrom = rcopy(R"as.matrix(predict(fits,we8thereRatings[1:10,],type=\"response\"))")
#
#
# CSV.write(joinpath(testdir,"data","dmr_we8thereCounts.csv.gz"),we8thereCounts)
# CSV.write(joinpath(testdir,"data","dmr_we8thereRatings.csv.gz"),we8thereRatings)
# CSV.write(joinpath(testdir,"data","dmr_coefsRdistrom.csv.gz"),DataFrame(full(coefsRdistrom)))
# CSV.write(joinpath(testdir,"data","dmr_zRdistrom.csv.gz"),DataFrame(zRdistrom))
# CSV.write(joinpath(testdir,"data","dmr_z1Rdistrom.csv.gz"),DataFrame(z1Rdistrom))
# CSV.write(joinpath(testdir,"data","dmr_predictRdistrom.csv.gz"),DataFrame(predictRdistrom))

we8thereCounts = CSV.read(joinpath(testdir,"data","dmr_we8thereCounts.csv.gz"))
we8thereRatings = CSV.read(joinpath(testdir,"data","dmr_we8thereRatings.csv.gz"))
we8thereTerms = broadcast(string,names(we8thereCounts))
coefsRdistrom = sparse(convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","dmr_coefsRdistrom.csv.gz"))))
zRdistrom = convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","dmr_zRdistrom.csv.gz")))
z1Rdistrom = convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","dmr_z1Rdistrom.csv.gz")))
predictRdistrom = convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","dmr_predictRdistrom.csv.gz")))

# covars = we8thereRatings[:,[:Overall]]
covars = we8thereRatings[:,:]
n,p = size(covars)
f = @model(c ~ Food + Service + Value + Atmosphere + Overall)
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
newcovars = covars[1:10,:]

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

# using a dataframe and list of variables
@time dmrcoefsdf = fit(DMRCoefs, f, we8thereRatings, counts; γ=γ, λminratio=0.01)
@test coef(dmrcoefsdf) == coefs
@time dmrpathsdf = fit(DMRPaths, f, we8thereRatings, counts; γ=γ, λminratio=0.01)
@test coef(dmrpathsdf) ≈ coefs

@time dmrcoefs2 = dmr(covars, counts; local_cluster=false, γ=γ, λminratio=0.01)
coefs2 = coef(dmrcoefs2)
@test coefs ≈ coefs2
@time dmrcoefs2 = fit(DMRCoefs, covars, counts; local_cluster=false, γ=γ, λminratio=0.01)
@test coef(dmrcoefs2) == coefs2

@time dmrPaths = dmrpaths(covars, counts; γ=γ, λminratio=0.01, verbose=false)
@time dmrPaths2 = fit(DMRPaths, covars, counts; γ=γ, λminratio=0.01, verbose=false)
@test coef(dmrPaths) == coef(dmrPaths2)
@test coef(dmrPaths) ≈ coefs
@test n > nobs(dmrPaths)
@test d == ncategories(dmrPaths)
@test p == ncovars(dmrPaths)

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

η = predict(dmrPaths,newcovars)
@test η ≈ predictRdistrom rtol=rtol
@test sum(η,2) ≈ ones(size(newcovars,1))

@test_throws ErrorException predict(dmrcoefs,newcovars)

@test z ≈ zRdistrom rtol=rtol
# println("rdist(z,zRdistrom)=$(rdist(z,zRdistrom))")

regdata3 = DataFrame(y=covars[:,1], z=zRdistrom[:,1], m=zRdistrom[:,2])
lm3 = lm(@formula(y ~ z+m), regdata3)
r23 = adjr2(lm3)

@test r21 ≈ r23 rtol=rtol
# println("rdist(r21,r23)=$(rdist(r21,r23))")

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
@time mnirglm = fit(CIR{DMR,GeneralizedLinearModel},covars,counts,1,Poisson(); γ=γ, λminratio=0.01, nocounts=true)
@test coefbwd(mnirglm) ≈ coef(dmrcoefs)
@test !(coeffwd(mnirglm) ≈ coeffwd(mnir))
@test !(predict(mnir,covars[1:10,:],counts[1:10,:]) ≈ predict(mnirglm,covars[1:10,:],counts[1:10,:]))

@time mnirdf = fit(CIR{DMR,LinearModel},f,we8thereRatings,counts,:Food; γ=γ, λminratio=0.01, nocounts=true)
@test coefbwd(mnirdf) ≈ coef(dmrcoefs)
@test coeffwd(mnirdf) ≈ coeffwd(mnir)
@time mnirglmdf = fit(CIR{DMR,GeneralizedLinearModel},f,we8thereRatings,counts,:Food,Poisson(); γ=γ, λminratio=0.01, nocounts=true)
@test coefbwd(mnirglmdf) ≈ coef(dmrcoefs)

zlm = lm(hcat(ones(n,1),z1,covars[:,2:end]),covars[:,1])
@test r2(zlm) ≈ r2(mnir)
@test adjr2(zlm) ≈ adjr2(mnir)
@test predict(zlm,hcat(ones(10,1),z1[1:10,:],covars[1:10,2:end])) ≈ predict(mnir,covars[1:10,:],counts[1:10,:])
@test predict(zlm,hcat(ones(10,1),z1[1:10,:],covars[1:10,2:end])) ≈ predict(mnirdf,covars[1:10,:],counts[1:10,:])

zlmnocounts = lm(hcat(ones(n,1),covars[:,2:end]),covars[:,1])
@test r2(zlmnocounts) ≈ r2(mnir; nocounts=true)
@test adjr2(zlmnocounts) ≈ adjr2(mnir; nocounts=true)
@test predict(zlmnocounts,hcat(ones(10,1),covars[1:10,2:end])) ≈ predict(mnir,covars[1:10,:],counts[1:10,:]; nocounts=true)
@test predict(zlmnocounts,hcat(ones(10,1),covars[1:10,2:end])) ≈ predict(mnirdf,covars[1:10,:],counts[1:10,:]; nocounts=true)

# CV
srand(13)
@time cvstats13 = cv(CIR{DMR,LinearModel},covars,smallcounts,1; gen=MLBase.Kfold(size(covars,1),2), γ=γ)
@time cvstats13b = cv(CIR{DMR,LinearModel},f,we8thereRatings,smallcounts,:Food; k=2, gentype=MLBase.Kfold, seed=13, γ=γ)
@test isequal(cvstats13,cvstats13b)

cvstats14 = cv(CIR{DMR,LinearModel},covars,smallcounts,1; k=2, gentype=MLBase.Kfold, γ=γ, seed=14)
@test !(isequal(cvstats13,cvstats14))

cvstats13glm = cv(CIR{DMR,GeneralizedLinearModel},f,we8thereRatings,smallcounts,:Food,Poisson(); k=2, gentype=MLBase.Kfold, γ=γ, seed=13)
@test !(isequal(cvstats13,cvstats13glm))

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
