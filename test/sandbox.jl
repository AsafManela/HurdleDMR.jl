# Install the HurdleDMR package
#Pkg.clone("https://github.com/AsafManela/HurdleDMR.jl")

# Add parallel workers and make package available to workers
addprocs(Sys.CPU_CORES-2)
import HurdleDMR; @everywhere using HurdleDMR

# Setup your data into an n-by-p covars matrix, and a (sparse) n-by-d counts matrix
# Here we generate some random data
using CSV, GLM, DataFrames, Distributions
n = 100
p = 3
d = 4

Random.seed!(13)
m = 1+rand(Poisson(5),n)
covars = rand(n,p)
ηfn(vi) = exp.([0 + i*sum(vi) for i=1:d])
q = [ηfn(covars[i,:]) for i=1:n]
scale!.(q,ones(n)./sum.(q))
counts = convert(SparseMatrixCSC{Float64,Int},hcat(broadcast((qi,mi)->rand(Multinomial(mi, qi)),q,m)...)')
covarsdf = DataFrame(covars,[:vy, :v1, :v2])

## To fit a hurdle distribtued multiple regression (hdmr):
m = hdmr(covars, counts; inpos=1:2, inzero=1:3)

# or with a dataframe and formula
mf = @model(h ~ vy + v1 + v2, c ~ vy + v1)
m = fit(HDMR, mf, covarsdf, counts)
# where the h ~ equation is the model for zeros (hurdle crossing) and c ~ is the model for positive counts

# in either case we can get the coefficients matrix for each variable + intercept as usual with
coefspos, coefszero = coef(m)

# By default we only return the AICc maximizing coefficients.
# To also get back the entire regulatrization paths, run
paths = fit(HDMRPaths, mf, covarsdf, counts)

coef(paths; select=:all)

# To get a sufficient reduction projection in direction of vy
z = srproj(m,counts,1,1)

# Counts inverse regression (cir) allows us to predict a covariate with the counts and other covariates
# Here we use hdmr for the backward regression and another model for the forward regression
# This can be accomplished with a single command, by fitting a CIR{HDMR,FM} where the forward model is FM <: RegressionModel
cir = fit(CIR{HDMR,LinearModel},mf,covarsdf,counts,:vy; nocounts=true)
# where the argument nocounts=true means we also fit a benchmark model without counts

# we can get the forward and backward model coefficients with
coefbwd(cir)
coeffwd(cir)

# the fitted model can be used to predict vy with new data
yhat = predict(cir, covarsdf[1:10,:], counts[1:10,:])

# and we can also predict only with the other covariates, which in this case
# is just a linear regression
yhat_nocounts = predict(cir, covarsdf[1:10,:], counts[1:10,:]; nocounts=true)

# To fit a distribtued multinomial regression (dmr):
m = dmr(covars, counts)

# or with a dataframe and formula
mf = @model(c ~ vy + v1 + v2)
m = fit(DMR, mf, covarsdf, counts)

# in either case we can get the coefficients matrix for each variable + intercept as usual with
coef(m)

# By default we only return the AICc maximizing coefficients.
# To also get back the entire regulatrization paths, run
paths = fit(DMRPaths, mf, covarsdf, counts)

# we can now select, for example the coefficients that minimize CV mse (takes a while)
coef(paths; select=:CVmin)

# To get a sufficient reduction projection in direction of vy
z = srproj(m,counts,1)

# A multinomial inverse regression (mnir) uses dmr for the backward regression and another model for the forward regression
# This can be accomplished with a single command, by fitting a CIR{DMR,FM} where FM is the forward RegressionModel
mnir = fit(CIR{DMR,LinearModel},mf,covarsdf,counts,:vy)

# we can get the forward and backward model coefficients with
coefbwd(mnir)
coeffwd(mnir)

# the fitted model can be used to predict vy with new data
yhat = predict(mnir, covarsdf[1:10,:], counts[1:10,:])

# Suppose instead we want to predict a discrete variable, then perhaps use a Poisson GLM as follows
mnir = fit(CIR{DMR,GeneralizedLinearModel},mf,covarsdf,counts,:vy,Gamma())




using Coverage
# defaults to src/; alternatively, supply the folder name as argument
coverage = process_folder(srcfolder)
# Get total coverage for all Julia files
covered_lines, total_lines = get_summary(coverage)
# Or process a single file
@show get_summary(process_file("src/HurdleDMR.jl"))

function focus(X; j=:)
  X[:,j]
end
function focus2(X; j=indices(X,2))
  X[:,j]
end

A = sprand(5000,4000,0.2)
focus(A)
focus2(A)
focus(A; j=3)
focus2(A; j=3)

j = indices(A,2)
j == 3:3
A
in(4000,indices(A,2))
in(4000,3)
A[indices(A,1),indices(A,2)]
j = 4
focusj = to_indices(A,(:,4))
A[indices(A,1),indices(A,2)]
A[:,4]
A[focusj[1],focusj[2]]

typeof(j)
typeof(:)
colon(1,1,3)
j == 3:3

# clean_folder("/home/amanela/.julia/v0.4")
