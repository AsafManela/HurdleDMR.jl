# eventually, these path lines should only occur in runtests.jl
# testfolder = dirname(@__FILE__)
# srcfolder = joinpath(testfolder,"..","src")
# # push!(LOAD_PATH, joinpath(testfolder,".."))
# push!(LOAD_PATH, srcfolder)

using FactCheck, Gadfly, Distributions

include("addworkers.jl")

using GLM, DataFrames, LassoPlot

@everywhere using HurdleDMR

# reload("HurdleDMR")

# # uncomment to generate R benchmark
# using RCall
# import Rdistrom
# R"library(textir)"
# R"library(Matrix)"
# R"data(we8there)"
# we8thereCounts = DataFrame(rcopy(R"as.matrix(we8thereCounts)"))
# we8thereRatings = rcopy("we8thereRatings")
# we8thereTerms = rcopy(R"we8thereCounts@Dimnames$Terms")
# names!(we8thereCounts,map(Symbol,we8thereTerms))
# writetable(joinpath(testfolder,"data","dmr_we8thereCounts.csv.gz"),we8thereCounts)
# writetable(joinpath(testfolder,"data","dmr_we8thereRatings.csv.gz"),we8thereRatings)

we8thereCounts = readtable(joinpath(testfolder,"data","dmr_we8thereCounts.csv.gz"))
we8thereRatings = readtable(joinpath(testfolder,"data","dmr_we8thereRatings.csv.gz"))
we8thereTerms = map(string,names(we8thereCounts))

# covars = we8thereRatings[:,[:Overall]]
covars = we8thereRatings[:,:]
n,p = size(covars)
inzero = 1:p

inpos = [1,3]
covarspos = we8thereRatings[:,inpos]

T = Float64
counts=sparse(convert(Matrix{Float64},we8thereCounts))
covars=convert(Array{T,2},covars)
covarspos=convert(Array{T,2},covarspos)

npos,ppos = size(covarspos)
d = size(counts,2)

γ=1.0

###########################################################
# hurdle with covarspos == covarszero
###########################################################
facts("hurdle-dmr with covarspos == covarszero") do

# reload("HurdleDMR")

# hurdle dmr parallel local cluster
@time coefsHppos, coefsHpzero = HurdleDMR.hdmr(covars, counts; parallel=true)
@fact size(coefsHppos) --> (p+1, d)
@fact size(coefsHpzero) --> (p+1, d)

# hurdle dmr parallel remote cluster
@time coefsHppos2, coefsHpzero2 = HurdleDMR.hdmr(covars, counts; parallel=true, local_cluster=false)
@fact coefsHppos --> roughly(coefsHppos2)
@fact coefsHpzero --> roughly(coefsHpzero)

# # # hurdle dmr serial
# @time coefsHspos, coefsHszero = HurdleDMR.hdmr(covars, counts; parallel=false)
# @fact coefsHppos --> roughly(coefsHspos)
# @fact coefsHpzero --> roughly(coefsHszero)

# test posindic used by srproj
m = rand(Poisson(0.1),30,500)
ms = sparse(m)
Im = HurdleDMR.posindic(m)
Ims = HurdleDMR.posindic(ms)
@fact Im --> Ims

zHpos = HurdleDMR.srproj(coefsHppos, counts)
@fact size(zHpos) --> (n,p+1)

zHzero = HurdleDMR.srproj(coefsHpzero, HurdleDMR.posindic(counts))
@fact size(zHzero) --> (n,p+1)

# first half of coefs belongs to covarspos
z1pos = HurdleDMR.srproj(coefsHppos, counts, 1)
@fact z1pos --> roughly(zHpos[:,[1,p+1]])

# second half of coefs belongs to covarszero
z1zero = HurdleDMR.srproj(coefsHpzero, HurdleDMR.posindic(counts), 1)
@fact z1zero --> roughly(zHzero[:,[1,p+1]])

Z1 = HurdleDMR.srproj(coefsHppos, coefsHpzero, counts, 1, 1; intercept=true)
@fact Z1 --> [z1pos[:,1] z1zero[:,1] z1pos[:,2]]

Z3 = HurdleDMR.srproj(coefsHppos, coefsHpzero, counts, 3, 3; intercept=true)
z3pos = HurdleDMR.srproj(coefsHppos, counts, 3)
z3zero = HurdleDMR.srproj(coefsHpzero, HurdleDMR.posindic(counts), 3)
@fact Z3 --> [z3pos[:,1] z3zero[:,1] z3pos[:,2]]

regdata = DataFrame(y=covars[:,1], zw1=z1zero[:,1], zv1=z1pos[:,1], m=z1pos[:,2])
lmw1 = lm(@formula(y ~ zw1+m), regdata)
r2w1 = adjr2(lmw1)

lmv1 = lm(@formula(y ~ zv1+m), regdata)
r2v1 = adjr2(lmv1)

lmw1v1 = lm(@formula(y ~ zw1+zv1+m), regdata)
r2w1v1 = adjr2(lmw1v1)

X1, X1_nocounts, includezpos = HurdleDMR.srprojX(coefsHppos,coefsHpzero,counts,covars,1; includem=true)
@fact X1_nocounts --> [ones(n) covars[:,2:end]]
@fact X1 --> [X1_nocounts Z1]

X2, X2_nocounts, includezpos = HurdleDMR.srprojX(coefsHppos,coefsHpzero,counts,covars,1; includem=false)
@fact X2_nocounts --> X1_nocounts
@fact X2 --> X1[:,1:end-1]

X3, X3_nocounts, includezpos = HurdleDMR.srprojX(coefsHppos,coefsHpzero,counts,covars,3; includem=true)
@fact X3_nocounts --> [ones(n) covars[:,[1,2,4,5]]]
@fact X3 --> [X3_nocounts Z3]

@time cvstats13 = HurdleDMR.cross_validate_hdmr_srproj(covars,counts,1; k=2, gentype=MLBase.Kfold, γ=γ)
cvstats13b = HurdleDMR.cross_validate_hdmr_srproj(covars,counts,1; k=2, gentype=MLBase.Kfold, γ=γ)
@fact isequal(cvstats13,cvstats13b) --> true

@time cvstats14 = HurdleDMR.cross_validate_hdmr_srproj(covars,counts,1; k=2, gentype=MLBase.Kfold, γ=γ, seed=14)
@fact isequal(cvstats13,cvstats14) --> false

@time cvstatsSerialKfold = HurdleDMR.cross_validate_hdmr_srproj(covars,counts,1; k=3, gentype=HurdleDMR.SerialKfold, γ=γ)
@fact isequal(cvstats13,cvstatsSerialKfold) --> false

end

####################################################################
# hurdle with covarspos ≠ covarszero, both models includes projdir
####################################################################
facts("hurdle-dmr with covarspos ≠ covarszero, both models includes projdir") do

# hurdle dmr parallel local cluster
@time coefsHppos, coefsHpzero = HurdleDMR.hdmr(covars, counts; covarspos=covarspos, parallel=true)
@fact size(coefsHppos) --> (ppos+1, d)
@fact size(coefsHpzero) --> (p+1, d)

# hurdle dmr parallel remote cluster
@time coefsHppos2, coefsHpzero2 = HurdleDMR.hdmr(covars, counts; covarspos=covarspos, parallel=true, local_cluster=false)
@fact coefsHppos --> roughly(coefsHppos2)
@fact coefsHpzero --> roughly(coefsHpzero2)

# # # hurdle dmr serial
# @time coefsHspos, coefsHszero = HurdleDMR.hdmr(covars, counts; covarspos=covarspos, parallel=false)
# @fact coefsHppos --> roughly(coefsHspos)
# @fact coefsHpzero --> roughly(coefsHszero)

zHpos = HurdleDMR.srproj(coefsHppos, counts)
@fact size(zHpos) --> (n,ppos+1)

zHzero = HurdleDMR.srproj(coefsHpzero, HurdleDMR.posindic(counts))
@fact size(zHzero) --> (n,p+1)

# first half of coefs belongs to covarspos
z1pos = HurdleDMR.srproj(coefsHppos, counts, 1)
@fact z1pos --> roughly(zHpos[:,[1,ppos+1]])

# second half of coefs belongs to covarszero
z1zero = HurdleDMR.srproj(coefsHpzero, HurdleDMR.posindic(counts), 1)
@fact z1zero --> roughly(zHzero[:,[1,p+1]])

Z1 = HurdleDMR.srproj(coefsHppos, coefsHpzero, counts, 1, 1; intercept=true)
@fact Z1 --> [z1pos[:,1] z1zero[:,1] z1pos[:,2]]

Z3 = HurdleDMR.srproj(coefsHppos, coefsHpzero, counts, 2, 3; intercept=true)
z3pos = HurdleDMR.srproj(coefsHppos, counts, 2)
z3zero = HurdleDMR.srproj(coefsHpzero, HurdleDMR.posindic(counts), 3)
@fact Z3 --> [z3pos[:,1] z3zero[:,1] z3pos[:,2]]

regdata = DataFrame(y=covars[:,1], zw1=z1zero[:,1], zv1=z1pos[:,1], m=z1pos[:,2], v1=covarspos[:,1], w1=covars[:,1])
lmw1 = lm(@formula(y ~ zw1+m), regdata)
r2w1 = adjr2(lmw1)

lmv1 = lm(@formula(y ~ zv1+m), regdata)
r2v1 = adjr2(lmv1)

lmw1v1 = lm(@formula(y ~ zw1+zv1+m), regdata)
r2w1v1 = adjr2(lmw1v1)

@time X1, X1_nocounts, includezpos = HurdleDMR.srprojX(coefsHppos,coefsHpzero,counts,covars,1; inpos=inpos, includem=true)
@fact X1_nocounts --> [ones(n) covars[:,2:end]]
@fact X1 --> [X1_nocounts Z1]

X2, X2_nocounts, includezpos = HurdleDMR.srprojX(coefsHppos,coefsHpzero,counts,covars,1; inpos=inpos, includem=false)
@fact X2_nocounts --> X1_nocounts
@fact X2 --> X1[:,1:end-1]

X3, X3_nocounts, includezpos = HurdleDMR.srprojX(coefsHppos,coefsHpzero,counts,covars,3; inpos=inpos, includem=true)
@fact X3_nocounts --> [ones(n) covars[:,[1,2,4,5]]]
@fact X3 --> [X3_nocounts Z3]

@time cvstats13 = HurdleDMR.cross_validate_hdmr_srproj(covars,counts,1; inpos=inpos, k=2, gentype=MLBase.Kfold, γ=γ)
@time cvstats13b = HurdleDMR.cross_validate_hdmr_srproj(covars,counts,1; inpos=inpos, k=2, gentype=MLBase.Kfold, γ=γ)
@fact isequal(cvstats13,cvstats13b) --> true

@time cvstats14 = HurdleDMR.cross_validate_hdmr_srproj(covars,counts,1; inpos=inpos, k=2, gentype=MLBase.Kfold, γ=γ, seed=14)
@fact isequal(cvstats13,cvstats14) --> false

@time cvstatsSerialKfold = HurdleDMR.cross_validate_hdmr_srproj(covars,counts,1; inpos=inpos, k=3, gentype=HurdleDMR.SerialKfold, γ=γ)
@fact isequal(cvstats13,cvstatsSerialKfold) --> false

end

########################################################################
# hurdle with covarspos ≠ covarszero, only pos model includes projdir
########################################################################
facts("hurdle-dmr with covarspos ≠ covarszero, only pos model includes projdir") do

covarszero = covars[:,2:end]
nzero,pzero = size(covarszero)
inzero = 2:p

# hurdle dmr parallel local cluster
@time coefsHppos, coefsHpzero = HurdleDMR.hdmr(covarszero, counts; covarspos=covarspos, parallel=true)
@fact size(coefsHppos) --> (ppos+1, d)
@fact size(coefsHpzero) --> (pzero+1, d)

# hurdle dmr parallel remote cluster
@time coefsHppos2, coefsHpzero2 = HurdleDMR.hdmr(covarszero, counts; covarspos=covarspos, parallel=true, local_cluster=false)
@fact coefsHppos --> roughly(coefsHppos2)
@fact coefsHpzero --> roughly(coefsHpzero2)

# # hurdle dmr serial
# @time coefsHspos, coefsHszero = HurdleDMR.hdmr(covarszero, counts; covarspos=covarspos, parallel=false)
# @fact coefsHppos --> roughly(coefsHspos)
# @fact coefsHpzero --> roughly(coefsHszero)

zHpos = HurdleDMR.srproj(coefsHppos, counts)
@fact size(zHpos) --> (n,ppos+1)

zHzero = HurdleDMR.srproj(coefsHpzero, HurdleDMR.posindic(counts))
@fact size(zHzero) --> (n,pzero+1)

z1pos = HurdleDMR.srproj(coefsHppos, counts, 1)
@fact z1pos --> roughly(zHpos[:,[1,ppos+1]])

# projdir is not included in zero model
# z1zero = HurdleDMR.srproj(coefsHpzero, HurdleDMR.posindic(counts), 1)
# @fact z1zero --> roughly(zHzero[:,[1,p+1]])

Z1 = HurdleDMR.srproj(coefsHppos, coefsHpzero, counts, 1, 0; intercept=true)
@fact Z1 --> z1pos

Z3 = HurdleDMR.srproj(coefsHppos, coefsHpzero, counts, 2, 2; intercept=true)
z3pos = HurdleDMR.srproj(coefsHppos, counts, 2)
z3zero = HurdleDMR.srproj(coefsHpzero, HurdleDMR.posindic(counts), 2)
@fact Z3 --> [z3pos[:,1] z3zero[:,1] z3pos[:,2]]

regdata = DataFrame(y=covars[:,1], zv1=z1pos[:,1], m=z1pos[:,2], v1=covarspos[:,1], w1=covarszero[:,1])

lmv1 = lm(@formula(y ~ zv1+m), regdata)
r2v1 = adjr2(lmv1)

@time X1, X1_nocounts, includezpos = HurdleDMR.srprojX(coefsHppos,coefsHpzero,counts,covars,1; inzero=inzero, inpos=inpos, includem=true)
@fact X1_nocounts --> [ones(n) covars[:,2:end]]
@fact X1 --> [X1_nocounts Z1]

X2, X2_nocounts, includezpos = HurdleDMR.srprojX(coefsHppos,coefsHpzero,counts,covars,1; inzero=inzero, inpos=inpos, includem=false)
@fact X2_nocounts --> X1_nocounts
@fact X2 --> X1[:,1:end-1]

X3, X3_nocounts, includezpos = HurdleDMR.srprojX(coefsHppos,coefsHpzero,counts,covars,3; inzero=inzero, inpos=inpos, includem=true)
@fact X3_nocounts --> [ones(n) covars[:,[2,4,5,1]]]
@fact X3 --> [X3_nocounts Z3]

@time cvstats13 = HurdleDMR.cross_validate_hdmr_srproj(covars,counts,1; inzero=inzero, inpos=inpos, k=2, gentype=MLBase.Kfold, γ=γ)
@time cvstats13b = HurdleDMR.cross_validate_hdmr_srproj(covars,counts,1; inzero=inzero, inpos=inpos, k=2, gentype=MLBase.Kfold, γ=γ)
@fact isequal(cvstats13,cvstats13b) --> true

@time cvstats14 = HurdleDMR.cross_validate_hdmr_srproj(covars,counts,1; inzero=inzero, inpos=inpos, k=2, gentype=MLBase.Kfold, γ=γ, seed=14)
@fact isequal(cvstats13,cvstats14) --> false

@time cvstatsSerialKfold = HurdleDMR.cross_validate_hdmr_srproj(covars,counts,1; inzero=inzero, inpos=inpos, k=3, gentype=HurdleDMR.SerialKfold, γ=γ)
@fact isequal(cvstats13,cvstatsSerialKfold) --> false

end

rmprocs(workers())

###########################################################
# Development
###########################################################
# reload("HurdleDMR")


###########################################################
# End Development
###########################################################

# # exception analysis code:
# (obj,objold,path,f,newcoef,oldcoef,b0,oldb0,b0diff,coefdiff,scratchmu,cd,r,α,curλ)=(0,0,[],0,[],[],0,0,0,[],[],[],[],0,0)
# ex=[]
# try
#   coefsH = HurdleDMR.dmr(covars, counts; hurdle=true, parallel=false, verbose=true)
# catch e
#   # if typeof(e) == Lasso.ConvergenceException
#   #   (obj,objold,path,f,newcoef,oldcoef,b0,oldb0,b0diff,coefdiff,scratchmu,cd,r,α,curλ) = e.debugvars
#   #   m = path.m
#   #   warn(e.msg)
#   #   nothing
#   if typeof(e) <: ErrorException
#     warn("caught exception $e")
#     ex=e
#   else
#     ex=e
#   end
# end
# typeof(ex) == Base.LinAlg.PosDefException
# coefsH
#
# path.m.rr
# ex.msg
# fieldnames(ex)
# rr = path.m.rr
# pp = path.m.pp

#   end
# end #do facts

# @time b0serial, coefsserial = HurdleDMR.dmr(covars, counts; γ=γ, nlocal_workers=1, irls_maxiter=2000)
# using ProfileView
# Profile.init(delay=0.001)
# Profile.clear()
# @profile b0, coefs = dmr(covars, counts; irls_maxiter=2000);
# ProfileView.view()
# # Profile.print()
#
# size(b0)
# size(coefs)
# [b0 coefs']'
#


#
