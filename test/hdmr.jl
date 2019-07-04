# common args for all hdmr tests
testargs = Dict(:verbose=>false,:showwarnings=>true)

@testset "hdmr" begin

hdmrmodels = [
    @model(h ~ x + z + cat + y, c ~ x + z + cat + y)
    @model(h ~ x + z + cat + y, c ~ z + cat + y)
    @model(h ~ x + z + cat, c ~ x + z + cat + y)
    @model(h ~ x + z + cat, c ~ z + cat + y)
    ]

@testset "$M" for M in (InclusionRepetition,Hurdle)

includel = (M == InclusionRepetition)

@testset "$(replace(string(f),"\n" => " "))" for f in hdmrmodels

# parse and merge rhs terms
trmszero = HurdleDMR.getrhsterms(f, :h)
trmspos = HurdleDMR.getrhsterms(f, :c)
trms, inzero, inpos = HurdleDMR.mergerhsterms(trmszero,trmspos)

# create model matrix
mf, mm, countsb = HurdleDMR.createmodelmatrix(trms, covarsdf, counts, Dict())

# inzero and inpos may be different in mm with factor variables
inzero, inpos = HurdleDMR.mapins(inzero, inpos, mm)
pzero = length(inzero)
ppos = length(inpos)

# resolve projdirs
projdir = HurdleDMR.ixprojdir(trms, :y, mm)
dirpos = something(findfirst(:y .== trmspos.terms), -1) + 1
dirzero = something(findfirst(:y .== trmszero.terms), -1) + 1

# hurdle dmr parallel local cluster
hdmrcoefs = hdmr(covars, counts, M; inzero=inzero, inpos=inpos, parallel=true, testargs...)
coefsHppos, coefsHpzero = coef(hdmrcoefs)
@test size(coefsHppos) == (ppos+1, d)
@test size(coefsHpzero) == (pzero+1, d)
@test_throws MethodError coef(hdmrcoefs, AllSeg())

# test Int matrix for counts
hdmrcoefsint = hdmr(covars, countsint, M; inzero=inzero, inpos=inpos, parallel=true, testargs...)
coefsHpposint, coefsHpzeroint = coef(hdmrcoefsint)
@test coefsHpposint == coefsHppos
@test coefsHpzeroint == coefsHpzero

hdmrcoefsb = fit(HDMRCoefs{M}, covars, counts; inzero=inzero, inpos=inpos, parallel=true, testargs...)
@test coef(hdmrcoefsb)[1] == coefsHppos
@test coef(hdmrcoefsb)[2] == coefsHpzero
@test n == nobs(hdmrcoefs)
@test d == ncategories(hdmrcoefs)
@test ppos == ncovarspos(hdmrcoefs)
@test pzero == ncovarszero(hdmrcoefs)

# select=MinBIC()
hdmrcoefsb = fit(HDMRCoefs{M}, covars, counts; inzero=inzero, inpos=inpos, select=MinBIC(), testargs...)
@test coef(hdmrcoefsb)[1] != coefsHppos
# @test coef(hdmrcoefsb)[2] != coefsHpzero
hdmrb = fit(HDMR{M}, covars, counts; inzero=inzero, inpos=inpos, select=MinBIC(), testargs...)
@test coef(hdmrb)[1] == coef(hdmrcoefsb)[1]
@test coef(hdmrb)[2] == coef(hdmrcoefsb)[2]

# hurdle dmr parallel remote cluster
hdmrcoefs2 = hdmr(covars, counts, M; inzero=inzero, inpos=inpos, parallel=true, local_cluster=false, testargs...)
coefsHppos2, coefsHpzero2 = coef(hdmrcoefs2)
@test coefsHppos ≈ coefsHppos2
@test coefsHpzero ≈ coefsHpzero2
hdmrcoefs2 = fit(HDMRCoefs{M}, covars, counts; inzero=inzero, inpos=inpos, parallel=true, local_cluster=false, testargs...)
@test coef(hdmrcoefs2)[1] ≈ coefsHppos2
@test coef(hdmrcoefs2)[2] ≈ coefsHpzero2
@test_throws ErrorException predict(hdmrcoefs2,newcovars)

hdmrpaths3 = fit(HDMRPaths{M}, covars, counts; inzero=inzero, inpos=inpos, parallel=true, testargs...)
coefsHppos3, coefsHpzero3 = coef(hdmrpaths3)
@test coefsHppos3 ≈ coefsHppos
@test coefsHpzero3 ≈ coefsHpzero
@test n == nobs(hdmrpaths3)
@test d == ncategories(hdmrpaths3)
@test ppos == ncovarspos(hdmrpaths3)
@test pzero == ncovarszero(hdmrpaths3)
@test size(hdmrpaths3.nlpaths,1) == d
η = predict(hdmrpaths3,newcovars)
@test sum(η, dims=2) ≈ ones(size(η, 1))
coefsallpos, coefsallzero = coef(hdmrpaths3, AllSeg())
@test size(coefsallpos,1) > 1
@test size(coefsallpos,2) == ppos+1
@test size(coefsallpos,3) == d
@test size(coefsallzero,1) > 1
@test size(coefsallzero,2) == pzero+1
@test size(coefsallzero,3) == d

# hurdle dmr serial
hdmrcoefs3 = hdmr(covars, counts, M; inzero=inzero, inpos=inpos, parallel=false, testargs...)
coefsHspos, coefsHszero = coef(hdmrcoefs3)
@test coefsHppos ≈ coefsHspos
@test coefsHpzero ≈ coefsHszero
hdmrcoefs3 = fit(HDMRCoefs{M}, covars, counts; inzero=inzero, inpos=inpos, parallel=false, testargs...)
@test coef(hdmrcoefs3)[1] ≈ coefsHspos
@test coef(hdmrcoefs3)[2] ≈ coefsHszero

# serial paths
hdmrcoefs4 = hdmrpaths(covars, counts, M; inzero=inzero, inpos=inpos, parallel=false, testargs...)
coefsHspos4, coefsHszero4 = coef(hdmrcoefs4)
@test coefsHspos == coefsHspos4
@test coefsHszero == coefsHszero4

# using a dataframe and formula
hdmrcoefsdf = fit(HDMRCoefs{M}, f, covarsdf, counts; inzero=inzero, inpos=inpos, parallel=true, testargs...)
@test coef(hdmrcoefsdf)[1] == coefsHppos
@test coef(hdmrcoefsdf)[2] ≈ coefsHpzero
@test_throws ErrorException predict(hdmrcoefsdf,newcovars)

hdmrpathsdf = fit(HDMRPaths{M}, f, covarsdf, counts; inzero=inzero, inpos=inpos, parallel=true, testargs...)
@test coef(hdmrpathsdf)[1] == coefsHppos3
@test coef(hdmrpathsdf)[2] ≈ coefsHpzero3
@test predict(hdmrpathsdf,newcovars) ≈ η

zHpos = srproj(coefsHppos, counts)
@test size(zHpos) == (n,ppos+1)

zHzero = srproj(coefsHpzero, posindic(counts))
@test size(zHzero) == (n,pzero+1)

Z1 = srproj(coefsHppos, coefsHpzero, counts, dirpos, dirzero;
    includel=includel, intercept=true)

if dirpos > 0
    z1pos = srproj(coefsHppos, counts, dirpos)
    @test z1pos ≈ zHpos[:,[dirpos,ppos+1]]
    @test hascol(Z1,z1pos[:,1])
    @test !includel || hascol(Z1,z1pos[:,2])
end

if dirzero > 0
    z1zero = srproj(coefsHpzero, posindic(counts), dirzero)
    @test z1zero ≈ zHzero[:,[dirzero,pzero+1]]
    @test hascol(Z1,z1zero[:,1])
    @test !includel || hascol(Z1,z1zero[:,2])
end

Z1b = srproj(hdmrcoefs, counts, dirpos, dirzero; intercept=true)
@test Z1 == Z1b

@test_throws ErrorException srproj(coefsHppos, coefsHpzero, counts, 0, 0; intercept=true)

X1, X1_nocounts, inz = srprojX(coefsHppos,coefsHpzero,counts,covars,projdir;
    inzero=inzero, inpos=inpos, includel=includel, includem=true)
ix = filter!(x->x!=projdir,collect(1:p))
@test X1_nocounts ≈ [ones(n) covars[:,ix]] rtol=1e-8
@test X1 ≈ [X1_nocounts Z1] rtol=1e-8
X1b, X1_nocountsb, inzb = srprojX(hdmrcoefs,counts,covars,projdir;
    inzero=inzero, inpos=inpos, includem=true)
@test X1 ≈ X1b rtol=1e-8
@test X1_nocountsb ≈ X1_nocountsb rtol=1e-8
@test inz == inzb

X2, X2_nocounts, inz = srprojX(coefsHppos,coefsHpzero,counts,covars,projdir;
    inzero=inzero, inpos=inpos, includel=includel, includem=false)
@test X2_nocounts ≈ X1_nocounts rtol=1e-8
if includel && dirzero > 0
    @test X2 ≈ [X1[:,1:end-2] X1[:,end]] rtol=1e-8
else
    @test X2 ≈ X1[:,1:end-1] rtol=1e-8
end
X2b, X2_nocountsb, inzb = srprojX(hdmrcoefs,counts,covars,projdir;
    inzero=inzero, inpos=inpos, includem=false)
@test X2 ≈ X2b rtol=1e-8
@test X2_nocountsb ≈ X2_nocountsb rtol=1e-8
@test inz == inzb

X3, X3_nocounts, inz3 = @test_logs (:info, r"rank\(X\)") srprojX(coefsHppos,coefsHpzero,zero(counts),covars,projdir;
    inzero=inzero, inpos=inpos, includel=includel, includem=true)
@test X3_nocounts == [ones(n) covars[:,setdiff(1:p,[projdir])]]
@test inz3 == [2]

X3, X3_nocounts, inz3b = @test_logs (:info, "includezpos == false. dropping zpos.") srprojX(coefsHppos,coefsHpzero,zero(counts),covars,projdir;
    inzero=inzero, inpos=inpos, includel=includel, includem=true, inz=inz3)
@test X3_nocounts == [ones(n) covars[:,setdiff(1:p,[projdir])]]
@test inz3 == inz3b

# HIR
hir = fit(CIR{HDMR{M}, LinearModel},covars,counts,projdir;
    inzero=inzero, inpos=inpos, nocounts=true, testargs...)
@test coefbwd(hir)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hir)[2] ≈ coef(hdmrcoefs)[2]

hirglm = fit(CIR{HDMR{M}, GeneralizedLinearModel},covars,counts,projdir,Gamma();
    inzero=inzero, inpos=inpos, nocounts=true, testargs...)
@test coefbwd(hirglm)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hirglm)[2] ≈ coef(hdmrcoefs)[2]
@test !(coeffwd(hirglm)[1] ≈ coeffwd(hir)[1])
@test !(coeffwd(hirglm)[2] ≈ coeffwd(hir)[2])

hirdf = fit(CIR{HDMR{M}, LinearModel},f,covarsdf,counts,:y;
    inzero=inzero, inpos=inpos, nocounts=true, testargs...)
@test coefbwd(hirdf)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hirdf)[2] ≈ coef(hdmrcoefs)[2]
@test coeffwd(hirdf) ≈ coeffwd(hir)
hirglmdf = fit(CIR{HDMR{M}, GeneralizedLinearModel},f,covarsdf,counts,:y,Gamma();
    inzero=inzero, inpos=inpos, nocounts=true, testargs...)
@test coefbwd(hirglmdf)[1] ≈ coef(hdmrcoefs)[1]
@test coefbwd(hirglmdf)[2] ≈ coef(hdmrcoefs)[2]
@test !(coeffwd(hirglmdf)[2] ≈ coeffwd(hirdf)[2])
@test !(coeffwd(hirglmdf)[1] ≈ coeffwd(hirdf)[1])

# select=MinBIC()
hirdfb = fit(CIR{HDMR{M}, LinearModel},f,covarsdf,counts,:y;
    inzero=inzero, inpos=inpos, nocounts=true, select=MinBIC(), testargs...)
@test coefbwd(hirdf)[1] != coefbwd(hirdfb)[1]
# @test coefbwd(hirdf)[2] != coefbwd(hirdfb)[2]
@test coeffwd(hirdf)[1] != coeffwd(hirdfb)[1]
@test coeffwd(hirdf)[2] != coeffwd(hirdfb)[2]

zlm = lm(hcat(ones(n,1),Z1,covars[:,1:4]),covars[:,projdir])
@test r2(zlm) ≈ r2(hir)
@test adjr2(zlm) ≈ adjr2(hir)
@test predict(zlm,hcat(ones(10,1),Z1[1:10,:],covars[1:10,1:4])) ≈ predict(hir,covars[1:10,:],counts[1:10,:])

zlmnocounts = lm(hcat(ones(n,1),covars[:,1:4]),covars[:,projdir])
@test r2(zlmnocounts) ≈ r2(hir; nocounts=true)
@test adjr2(zlmnocounts) ≈ adjr2(hir; nocounts=true)
@test predict(zlmnocounts,hcat(ones(10,1),covars[1:10,1:4])) ≈ predict(hir,covars[1:10,:],counts[1:10,:]; nocounts=true)

end

@testset "HDMR{$M} degenerate cases" begin

@info("Testing hdmr degenerate cases. The 2 following warnings by workers are expected ...")

# column j is never zero, so hj=1 for all observations
zcounts = deepcopy(counts)
zcounts[:,2] = zeros(n)
zcounts[:,3] = ones(n)

# make sure we are not adding all zero obseravtions
m = sum(zcounts, dims=2)
@test sum(m .== 0) == 0

# hurdle dmr parallel local cluster
hdmrcoefs = fit(HDMR{M}, covars, zcounts; parallel=true, testargs...)
coefsHppos, coefsHpzero = coef(hdmrcoefs)
@test size(coefsHppos) == (p+1, d)
@test size(coefsHpzero) == (p+1, d)
@test coefsHppos[:,2] == zeros(p+1)
@test coefsHpzero[:,2] == zeros(p+1)
@test coefsHppos[:,3] == zeros(p+1)
@test coefsHpzero[:,3] == zeros(p+1)
@test HurdleDMR.includelinX(hdmrcoefs) == includel

if M == Hurdle
    η3 = 0.36
    η4 = 0.6
else
    η3 = 0.32
    η4 = 0.64
end

# hurdle dmr parallel remote cluster
hdmrcoefs2 = fit(HDMRPaths{M},covars, zcounts; parallel=true, local_cluster=false, testargs...)
coefsHppos2, coefsHpzero2 = coef(hdmrcoefs)
@test coefsHppos ≈ coefsHppos2
@test coefsHpzero ≈ coefsHpzero2
η = predict(hdmrcoefs2,newcovars)
@test sum(η, dims=2) ≈ ones(size(newcovars, 1))
@test η[:,2] == zeros(size(newcovars,1))
@test η[:,3] ≈ ones(size(newcovars,1))*η3 rtol=0.05
@test η[:,4] ≈ ones(size(newcovars,1))*η4 rtol=0.06
@test HurdleDMR.includelinX(hdmrcoefs2) == includel

# hurdle dmr serial paths
rx = Regex("fit\\($M...\\) failed for countsj")
hdmrcoefs3 = @test_logs (:warn, rx) fit(HDMRPaths{M},covars, zcounts; parallel=false, testargs...)
coefsHppos3, coefsHpzero3 = coef(hdmrcoefs3)
@test coefsHppos ≈ coefsHppos3
@test coefsHpzero ≈ coefsHpzero3
η = predict(hdmrcoefs3,newcovars)
@test sum(η, dims=2) ≈ ones(size(newcovars, 1))
@test η[:,2] == zeros(size(newcovars,1))
@test η[:,3] ≈ ones(size(newcovars,1))*η3 rtol=0.05
@test η[:,4] ≈ ones(size(newcovars,1))*η4 rtol=0.06
@test HurdleDMR.includelinX(hdmrcoefs3) == includel

# hurdle dmr serial coefs
hdmrcoefs4 = @test_logs (:warn, r"failed on count dimension 2") fit(HDMR{M},covars, zcounts; parallel=false, testargs...)
coefsHppos4, coefsHpzero4 = coef(hdmrcoefs4)
@test size(coefsHppos4) == (p+1, d)
@test size(coefsHpzero4) == (p+1, d)
@test coefsHppos4[:,2] == zeros(p+1)
@test coefsHpzero4[:,2] == zeros(p+1)
@test coefsHppos4[:,3] == zeros(p+1)
@test coefsHpzero4[:,3] == zeros(p+1)
@test HurdleDMR.includelinX(hdmrcoefs4) == includel

if M == InclusionRepetition
    @info("Testing hdmr degenerate cases. The 12 following warnings by workers are expected ...")
end

zcounts = Matrix(deepcopy(counts))
Random.seed!(13)
for I = eachindex(zcounts)
    if iszero(zcounts[I])
        zcounts[I] = rand(1:10)
    end
end

# make sure we are not adding all zero obseravtions
m = sum(zcounts, dims=2)
@test sum(m .== 0) == 0
if M == InclusionRepetition
    l = sum(posindic(zcounts), dims=2)
    @test sum((m .- l) .== 0) == 0
    @test sum((d .- l) .== 0) == n
end

# hurdle dmr parallel local cluster
hdmrcoefs = fit(HDMR{M}, covars, zcounts; parallel=true, testargs...)
coefsHppos, coefsHpzero = coef(hdmrcoefs)
@test size(coefsHppos) == (p+1, d)
@test size(coefsHpzero) == (p+1, d)
if M == Hurdle
	@test coefsHppos[:,2] != zeros(p+1)
	@test coefsHppos[:,3] != zeros(p+1)
else
	@test coefsHpzero[:,2] == zeros(p+1)
	@test coefsHpzero[:,3] == zeros(p+1)
end
@test HurdleDMR.includelinX(hdmrcoefs) == includel

# hurdle dmr parallel remote cluster
hdmrcoefs2 = fit(HDMRPaths{M}, covars, zcounts; parallel=true, local_cluster=false, testargs...)
coefsHppos2, coefsHpzero2 = coef(hdmrcoefs2)
@test coefsHppos ≈ coefsHppos2
@test coefsHpzero ≈ coefsHpzero2
@test HurdleDMR.includelinX(hdmrcoefs2) == includel

# just checking the fit(HDMRPaths...) ignore local_cluster
hdmrcoefs3 = fit(HDMRPaths{M},covars, zcounts; parallel=true, local_cluster=true, testargs...)
coefsHppos3, coefsHpzero3 = coef(hdmrcoefs3)
@test coefsHppos2 == coefsHppos3
@test coefsHpzero2 == coefsHpzero3
@test HurdleDMR.includelinX(hdmrcoefs3) == includel

# hurdle dmr parallel remote cluster
if M == Hurdle
    hdmrcoefs4 = fit(HDMRPaths{M},covars, zcounts; parallel=false, testargs...)
else
    rx = Regex("fit\\($M...\\) failed for countsj")
    warnings = [(:warn, rx) for i=1:4]
    hdmrcoefs4 = @test_logs(warnings..., fit(HDMRPaths{M},covars, zcounts; parallel=false, testargs...))
end
coefsHppos4, coefsHpzero4 = coef(hdmrcoefs4)
@test coefsHppos ≈ coefsHppos4
@test coefsHpzero ≈ coefsHpzero4
@test HurdleDMR.includelinX(hdmrcoefs4) == includel

end

end

end
