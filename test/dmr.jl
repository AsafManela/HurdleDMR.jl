# common args for all dmr tests
# to get exactly Rdistrom's run use λminratio=0.01 because our gamlr's have different defaults
testargs = Dict(:γ=>γdistrom, :λminratio=>0.01, :verbose=>false,:showwarnings=>true)

@testset "dmr" begin

f = @model(c ~ x + z + cat + y)
@test_show f "1-part model: [Formula: c ~ x + z + cat + y]"

dmrcoefs = dmr(covars, counts; testargs...)
coefs = coef(dmrcoefs)
@test size(coefs) == (p+1, d)
@test_throws ErrorException coef(dmrcoefs; select=:all)
@test coefs ≈ coefsRdistrom rtol=rtol
# println("rdist(coefs,coefsRdistrom)=$(rdist(coefs,coefsRdistrom))")

# test Int matrix for counts
dmrcoefsint = dmr(covars, countsint; testargs...)
coefsint = coef(dmrcoefsint)
@test coefsint == coefs

dmrcoefsb = fit(DMRCoefs, covars, counts; testargs...)
@test coef(dmrcoefsb) == coefs
dmrb = fit(DMR, covars, counts; testargs...)
@test coef(dmrb) == coefs
@test n == nobs(dmrcoefsb)
@test d == ncategories(dmrcoefsb)
@test p == ncovars(dmrcoefsb)

# select=:BIC
dmrcoefsb = fit(DMRCoefs, covars, counts; select=:BIC, testargs...)
@test coef(dmrcoefsb) != coefs
dmrb = fit(DMR, covars, counts; select=:BIC, testargs...)
@test coef(dmrb) != coefs

# serial run
dmrcoefss = dmr(covars, counts; parallel=false, testargs...)
@test coefs ≈ coef(dmrcoefss)

# using a dataframe and list of variables
dmrcoefsdf = fit(DMRCoefs, f, covarsdf, counts; testargs...)
@test coef(dmrcoefsdf) == coefs
dmrpathsdf = fit(DMRPaths, f, covarsdf, counts; testargs...)
@test coef(dmrpathsdf) ≈ coefs

dmrcoefs2 = dmr(covars, counts; local_cluster=false, testargs...)
coefs2 = coef(dmrcoefs2)
@test coefs ≈ coefs2
dmrcoefs2 = fit(DMRCoefs, covars, counts; local_cluster=false, testargs...)
@test coef(dmrcoefs2) == coefs2

# keeping regulatrization paths
dmrPaths = dmrpaths(covars, counts; testargs...)
dmrPaths2 = fit(DMRPaths, covars, counts; testargs...)
# serial paths
dmrPathss = fit(DMRPaths, covars, counts; parallel=false, testargs...)
@test coef(dmrPaths) == coef(dmrPaths2)
@test coef(dmrPaths) ≈ coefs
@test coef(dmrPathss) ≈ coefs
coefsall = coef(dmrPaths; select=:all)
@test size(coefsall,1) > 1
@test size(coefsall,2) == p+1
@test size(coefsall,3) == d
@test n == nobs(dmrPaths)
@test d == ncategories(dmrPaths)
@test p == ncovars(dmrPaths)

z = srproj(coefs, counts)
zb = srproj(dmrcoefs, counts)
zc = srproj(dmrPaths, counts)
@test z == zb
@test z ≈ zc
@test size(z) == (size(counts,1),p+1)

regdata = DataFrame(y=covars[:,1], z=z[:,1], m=z[:,2])
lm1 = lm(@formula(y ~ z+m), regdata)
r21 = adjr2(lm1)

η = predict(dmrPaths,newcovars)
@test η ≈ predictRdistrom rtol=5rtol
# rdist(η,predictRdistrom)
@test sum(η, dims=2) ≈ ones(size(newcovars, 1))

@test_throws ErrorException predict(dmrcoefs,newcovars)

@test z ≈ zRdistrom rtol=rtol
# println("rdist(z,zRdistrom)=$(rdist(z,zRdistrom))")

regdata3 = DataFrame(y=covars[:,1], z=zRdistrom[:,1], m=zRdistrom[:,2])
lm3 = lm(@formula(y ~ z+m), regdata3)
r23 = adjr2(lm3)

@test r21 ≈ r23 rtol=rtol
# println("rdist(r21,r23)=$(rdist(r21,r23))")

# project in a single direction
z1 = srproj(coefs, counts, projdir)
z1b = srproj(dmrcoefs, counts, projdir)
z1c = srproj(coefs, counts, projdir)
@test z1 ≈ z[:,[projdir,end]]

z1dense = srproj(coefs, Matrix(counts), projdir)
z1denseb = srproj(dmrcoefs, Matrix(counts), projdir)
z1densec = srproj(dmrPaths, Matrix(counts), projdir)
@test z1dense == z1denseb
@test z1dense ≈ z1densec
@test z1dense ≈ z1

@test z1 ≈ z1Rdistrom rtol=rtol
X1, X1_nocounts, inz = srprojX(coefs,counts,covars,projdir; includem=true)
@test X1_nocounts == [ones(n,1) covars[:,1:4]]
@test X1 == [X1_nocounts z1]
@test inz == [1]

X1b, X1_nocountsb, inz = srprojX(dmrcoefs,counts,covars,projdir; includem=true)
@test X1 == X1b
@test X1_nocounts == X1_nocountsb
@test inz == [1]

X2, X2_nocounts, inz = srprojX(coefs,counts,covars,projdir; includem=false)
@test X2_nocounts == X1_nocounts
@test X2 == X1[:,1:end-1]
@test inz == [1]

X2b, X2_nocountsb, inz = srprojX(dmrcoefs,counts,covars,projdir; includem=false)
@test X2 == X2b
@test X2_nocounts == X2_nocountsb
@test inz == [1]

# project in a single direction, focusing only on cj
focusj = 3
z1j = srproj(coefs, counts, projdir; focusj=focusj)
z1jdense = srproj(coefs, Matrix(counts), projdir; focusj=focusj)
@test z1jdense == z1j

X1j, X1j_nocounts, inz = srprojX(coefs,counts,covars,projdir; includem=true, focusj=focusj)
@test X1j_nocounts == [ones(n,1) covars[:,1:4]]
@test X1j == [X1_nocounts z1j]
@test inz == [1]

X2j, X2j_nocounts, inz = srprojX(coefs,counts,covars,projdir; includem=false, focusj=focusj)
@test X2j_nocounts == X1_nocounts
@test X2j == X1j[:,1:end-1]
@test inz == [1]

X1jfull, X1jfull_nocounts, inz = srprojX(coefs,Matrix(counts),covars,projdir; includem=true, focusj=focusj)
@test X1jfull ≈ X1j
@test X1jfull_nocounts ≈ X1jfull_nocounts
@test inz == [1]

X2jfull, X2jfull_nocounts, inz = srprojX(coefs,Matrix(counts),covars,projdir; includem=false, focusj=focusj)
@test X2jfull ≈ X2j
@test X2jfull_nocounts ≈ X2jfull_nocounts
@test inz == [1]

# MNIR
mnir = fit(CIR{DMR,LinearModel},covars,counts,projdir; nocounts=true, testargs...)
@test coefbwd(mnir) ≈ coef(dmrcoefs)
mnirglm = fit(CIR{DMR,GeneralizedLinearModel},covars,counts,projdir,Gamma(); nocounts=false, testargs...)
@test coefbwd(mnirglm) ≈ coef(dmrcoefs)
@test !(coeffwd(mnirglm) ≈ coeffwd(mnir))
@test_throws ErrorException coeffwd(mnirglm; nocounts=true)
@test !(predict(mnir,covars[1:10,:],counts[1:10,:]) ≈ predict(mnirglm,covars[1:10,:],counts[1:10,:]))
@test !(predict(mnir,covars[1:10,:],counts[1:10,:]) ≈ predict(mnir,covars[1:10,:],counts[1:10,:];nocounts=true))
@test_throws ErrorException predict(mnirglm,covars[1:10,:],counts[1:10,:];nocounts=true)

mnirdf = fit(CIR{DMR,LinearModel},f,covarsdf,counts,:y; nocounts=true, testargs...)
@test coefbwd(mnirdf) ≈ coef(dmrcoefs)
@test coeffwd(mnirdf) ≈ coeffwd(mnir)
@test coeffwd(mnirdf) != coeffwd(mnir; nocounts=true)
@test coef(mnirdf) ≈ coef(mnir)
mnirglmdf = fit(CIR{DMR,GeneralizedLinearModel},f,covarsdf,counts,:y,Gamma(); nocounts=false, testargs...)
@test coefbwd(mnirglmdf) ≈ coef(dmrcoefs)
@test_throws ErrorException coeffwd(mnirglmdf; nocounts=true)

# select=:BIC
mnirdfb = fit(CIR{DMR,LinearModel},f,covarsdf,counts,:y; nocounts=true, select=:BIC, testargs...)
@test coefbwd(mnirdf) != coefbwd(mnirdfb)
@test coeffwd(mnirdf) != coeffwd(mnirdfb)

# #### debug start
# mm = mnirdf
# mdf = covarsdf[1:10,:]
# newTerms = StatsModels.dropresponse!(mm.mf.terms)
# # create new model frame/matrix
# newTerms.intercept = true
# mf = ModelFrame(newTerms, mdf; contrasts = mm.mf.contrasts)
# mf.terms.intercept = false
# newX = ModelMatrix(mf).m
# if !all(mf.nonmissing)
# counts = counts[mf.nonmissing,:]
# end
# yp = predict(mm, newX, counts; kwargs...)
# out = missings(eltype(yp), size(df, 1))
# out[mf.nonmissing] = yp
#
# predict(mnirdf,covarsdf[1:10,:],counts[1:10,:])
# size(covarsdf)
# mnirdf.model.projdir
# #### debug end
@test !(predict(mnirdf,covarsdf[1:10,:],counts[1:10,:]) ≈ predict(mnirglmdf,covarsdf[1:10,:],counts[1:10,:]))
@test !(predict(mnirdf,covarsdf[1:10,:],counts[1:10,:]) ≈ predict(mnirglmdf,covarsdf[1:10,:],counts[1:10,:]))
@test !(predict(mnirdf,covarsdf[1:10,:],counts[1:10,:]) ≈ predict(mnirdf,covarsdf[1:10,:],counts[1:10,:];nocounts=true))
@test_throws ErrorException predict(mnirglmdf,covarsdf[1:10,:],counts[1:10,:];nocounts=true)

zlm = lm(hcat(ones(n,1),z1,covars[:,1:4]),covars[:,projdir])
@test r2(zlm) ≈ r2(mnir)
@test adjr2(zlm) ≈ adjr2(mnir)
@test predict(zlm,hcat(ones(10,1),z1[1:10,:],covars[1:10,1:4])) ≈ predict(mnir,covars[1:10,:],counts[1:10,:])
@test predict(zlm,hcat(ones(10,1),z1[1:10,:],covars[1:10,1:4])) ≈ predict(mnirdf,covars[1:10,:],counts[1:10,:])

zlmnocounts = lm(hcat(ones(n,1),covars[:,1:4]),covars[:,projdir])
@test r2(zlmnocounts) ≈ r2(mnir; nocounts=true)
@test adjr2(zlmnocounts) ≈ adjr2(mnir; nocounts=true)
@test predict(zlmnocounts,hcat(ones(10,1),covars[1:10,1:4])) ≈ predict(mnir,covars[1:10,:],counts[1:10,:]; nocounts=true)
@test predict(zlmnocounts,hcat(ones(10,1),covars[1:10,1:4])) ≈ predict(mnirdf,covars[1:10,:],counts[1:10,:]; nocounts=true)

end

#########################################################################3
# degenerate cases
#########################################################################3

@testset "dmr degenerate cases" begin

@info("Testing dmr degenerate cases. The 2 following warnings by workers are expected ...")

f = @model(c ~ x + z + cat + y)

# always one (zero var) counts columns
zcounts = deepcopy(counts)
zcounts[:,2] = zeros(n)
zcounts[:,3] = ones(n)

# make sure we are not adding all zero obseravtions
m = sum(zcounts,dims=2)
@test sum(m .== 0) == 0

# this one should warn on dimension 2 but @test_warn doen't capture the workers' warnings
dmrzcoefs = dmr(covars, zcounts; testargs...)
zcoefs = coef(dmrzcoefs)
@test size(zcoefs) == (p+1, d)
@test zcoefs[:,2] ≈ zeros(p+1)

dmrzcoefs2 = dmr(covars, zcounts; local_cluster=false, testargs...)
zcoefs2 = coef(dmrzcoefs2)
@test zcoefs2 ≈ zcoefs

# serial runs can also test for warnings
dmrzcoefs3 = @test_logs (:warn, r"failed on count dimension 2") dmr(covars, zcounts; parallel=false, testargs...)
zcoefs2 = coef(dmrzcoefs2)
@test zcoefs2 ≈ zcoefs

dmrzpaths3 = @test_logs (:warn, r"failed for countsj") dmrpaths(covars, zcounts; parallel=false, testargs...)
zcoefs3 = coef(dmrzpaths3)
@test zcoefs3 ≈ zcoefs rtol=rtol

# test an observation with all zeros
zcounts = deepcopy(counts)
zcounts[1,:] .= 0.0
m = sum(zcounts, dims=2)
@test sum(m .== 0) == 1
dmrzcoefs = @test_logs (:warn, r"omitting 1") dmr(covars, zcounts; testargs...)
dmrzcoefs2 = dmr(covars[2:end,:], counts[2:end,:]; testargs...)
zcoefs3 = coef(dmrzcoefs)
@test size(zcoefs3) == (p+1, d)
@test nobs(dmrzcoefs) == n-1
@test zcoefs3 == coef(dmrzcoefs2)

zcovarsdf = deepcopy(covarsdf)
zcovarsdf[1] = convert(Vector{Union{Float64,Missing}},zcovarsdf[1])
zcovarsdf[1,1] = missing
dmrzcoefsdf = fit(DMR, f, zcovarsdf, counts; testargs...)
zcoefsdf = coef(dmrzcoefsdf)
@test zcoefsdf == zcoefs3

zmnirdf = fit(CIR{DMR,LinearModel},f,zcovarsdf,counts,:y; nocounts=true, testargs...)
zyhat = predict(zmnirdf, zcovarsdf, counts)
@test ismissing(zyhat[1])
@test !any(ismissing,zyhat[2:end])

end
