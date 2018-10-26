# generate test data
n = 200
p = 5
d = 4
ncats = 3

Random.seed!(33)
ctotal = 1 .+ rand(Poisson(5),n)
vs = rand(n,p-ncats+1)
covarsdf = DataFrame(vs, [:y, :x, :z])
covarsdf[:cat] = CategoricalArray(rand(["$i" for i=1:ncats], n))
ηfn(vi,g) = exp.([0 + i*sum(vi) - parse(Float64, g) for i=1:d])
q = [ηfn(vs[i,:], covarsdf[i,:cat]) for i=1:n]
for i=1:n
  q[i] ./= sum(q[i])
end
counts = convert(SparseMatrixCSC{Float64,Int}, hcat(broadcast((qi,mi)->rand(Multinomial(mi, qi)),q,ctotal)...)')
countsint = convert(Matrix{Int64},counts) # used for testing the int eltype case

# construct equivalent covars matrix so we can show how that api works too
covars = ModelMatrix(ModelFrame(@formula(y ~ x + z + cat + y), covarsdf)).m[:,2:end]

# used for testing predict
newcovars = covars[1:10,:]

# used for testing srproj and fit(CIR)
projdir = size(covars,2)

γdistrom = 1.0
# # uncomment to generate R benchmark
# using RCall
# R"library(textir)"
# R"library(Matrix)"
# R"cl <- makeCluster(2,type=\"FORK\")"
# R"fits <- dmr(cl, $covars, $counts, gamma=$γdistrom, verb=0)"
# R"stopCluster(cl)"
# coefsRdistrom = rcopy(R"as.matrix(coef(fits))")
# zRdistrom = rcopy(R"as.matrix(srproj(fits,$counts))")
# z1Rdistrom = rcopy(R"as.matrix(srproj(fits,$counts,$projdir))")
# predictRdistrom = rcopy(R"as.matrix(predict(fits,$newcovars,type=\"response\"))")
#
# CSV.write(joinpath(testdir,"data","dmr_coefsRdistrom.csv.gz"),DataFrame(Matrix(coefsRdistrom)))
# CSV.write(joinpath(testdir,"data","dmr_zRdistrom.csv.gz"),DataFrame(zRdistrom))
# CSV.write(joinpath(testdir,"data","dmr_z1Rdistrom.csv.gz"),DataFrame(z1Rdistrom))
# CSV.write(joinpath(testdir,"data","dmr_predictRdistrom.csv.gz"),DataFrame(predictRdistrom))

coefsRdistrom = sparse(convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","dmr_coefsRdistrom.csv.gz"))))
zRdistrom = convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","dmr_zRdistrom.csv.gz")))
z1Rdistrom = convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","dmr_z1Rdistrom.csv.gz")))
predictRdistrom = convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","dmr_predictRdistrom.csv.gz")))
