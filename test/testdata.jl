# generate test data
n = 100
p = 3
d = 4

srand(13)
m = 1+rand(Poisson(5),n)
covars = rand(n,p)
ηfn(vi) = exp.([0 + i*sum(vi) for i=1:d])
q = [ηfn(covars[i,:]) for i=1:n]
scale!.(q,ones(n)./sum.(q))
counts = convert(SparseMatrixCSC{Float64,Int},hcat(broadcast((qi,mi)->rand(Multinomial(mi, qi)),q,m)...)')
countsint = convert(Matrix{Int64},counts) # used for testing the int eltype case

newcovars = covars[1:10,:]

covarsdf = DataFrame(covars,[:v1, :v2, :vy])
projdir = findfirst(names(covarsdf),:vy)

# # uncomment to generate R benchmark
# using RCall
# R"library(textir)"
# R"library(Matrix)"
# R"cl <- makeCluster(2,type=\"FORK\")"
# R"fits <- dmr(cl, $covars, $counts, gamma=$γ, verb=0)"
# R"stopCluster(cl)"
# coefsRdistrom = rcopy(R"as.matrix(coef(fits))")
# zRdistrom = rcopy(R"as.matrix(srproj(fits,$counts))")
# z1Rdistrom = rcopy(R"as.matrix(srproj(fits,$counts,3))")
# predictRdistrom = rcopy(R"as.matrix(predict(fits,$newcovars,type=\"response\"))")
#
# CSV.write(joinpath(testdir,"data","dmr_coefsRdistrom.csv.gz"),DataFrame(full(coefsRdistrom)))
# CSV.write(joinpath(testdir,"data","dmr_zRdistrom.csv.gz"),DataFrame(zRdistrom))
# CSV.write(joinpath(testdir,"data","dmr_z1Rdistrom.csv.gz"),DataFrame(z1Rdistrom))
# CSV.write(joinpath(testdir,"data","dmr_predictRdistrom.csv.gz"),DataFrame(predictRdistrom))

coefsRdistrom = sparse(convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","dmr_coefsRdistrom.csv.gz"))))
zRdistrom = convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","dmr_zRdistrom.csv.gz")))
z1Rdistrom = convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","dmr_z1Rdistrom.csv.gz")))
predictRdistrom = convert(Matrix{Float64},CSV.read(joinpath(testdir,"data","dmr_predictRdistrom.csv.gz")))
