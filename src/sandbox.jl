# Install the HurdleDMR package
#Pkg.clone("https://github.com/AsafManela/HurdleDMR.jl")

# Add parallel workers and make package available to workers
addprocs(Sys.CPU_CORES-2)
import HurdleDMR; @everywhere using HurdleDMR

# Setup your data into an n-by-p covars matrix, and a (sparse) n-by-d counts matrix
using CSV, GLM, DataFrames, Distributions
we8thereCounts = CSV.read(joinpath(Pkg.dir("HurdleDMR"),"test","data","dmr_we8thereCounts.csv.gz"))
counts = sparse(convert(Matrix{Float64},we8thereCounts))
covarsdf = CSV.read(joinpath(Pkg.dir("HurdleDMR"),"test","data","dmr_we8thereRatings.csv.gz"))
covars = convert(Matrix{Float64},covarsdf)
terms = map(string,names(we8thereCounts))

## To fit a hurdle distribtued multiple regression (hdmr):
m = hdmr(covars, counts; inpos=[1,3], inzero=1:5)

# or with a dataframe and formula
mf = @model(h ~ Food + Service + Value + Atmosphere + Overall, c ~ Food + Value)
m = fit(HDMR, mf, covarsdf, counts)
# where the h ~ equation is the model for zeros (hurdle crossing) and c ~ is the model for positive counts

# in either case we can get the coefficients matrix for each variable + intercept as usual with
coefspos, coefszero = coef(m)

# By default we only return the AICc maximizing coefficients.
# To also get back the entire regulatrization paths, run
paths = fit(HDMRPaths, mf, covarsdf, counts)

coef(paths; select=:all)

# To get a sufficient reduction projection in direction of Food
z = srproj(m,counts,1,1)

# Counts inverse regression (cir) allows us to predict a covariate with the counts and other covariates
# Here we use hdmr for the backward regression and another model for the forward regression
# This can be accomplished with a single command, by fitting a CIR{HDMR,FM} where the forward model is FM <: RegressionModel
cir = fit(CIR{HDMR,LinearModel},mf,covarsdf,counts,:Food; nocounts=true)
# where the argument nocounts=true means we also fit a benchmark model without counts

# we can get the forward and backward model coefficients with
coefbwd(cir)
coeffwd(cir)

# the fitted model can be used to predict Food with new data
yhat = predict(cir, covarsdf[1:10,:], counts[1:10,:])

# and we can also predict only with the other covariates, which in this case
# is just a linear regression
yhat_nocounts = predict(hir, covarsdf[1:10,:], counts[1:10,:]; nocounts=true)

# To fit a distribtued multinomial regression (dmr):
m = dmr(covars, counts)

# or with a dataframe and formula
m = fit(DMR, @model(c ~ Food + Service + Value + Atmosphere + Overall), covarsdf, counts)

# in either case we can get the coefficients matrix for each variable + intercept as usual with
coef(m)

# By default we only return the AICc maximizing coefficients.
# To also get back the entire regulatrization paths, run
mf = @model(c ~ Food + Service + Value + Atmosphere + Overall)
paths = fit(DMRPaths, mf, covarsdf, counts)

# we can now select, for example the coefficients that minimize CV mse (takes a while)
coef(paths; select=:CVmin)

# To get a sufficient reduction projection in direction of Food
z = srproj(m,counts,1)

# A multinomial inverse regression (mnir) uses dmr for the backward regression and another model for the forward regression
# This can be accomplished with a single command, by fitting a CIR{DMR,FM} where FM is the forward RegressionModel
mnir = fit(CIR{DMR,LinearModel},mf,covarsdf,counts,:Food)

# we can get the forward and backward model coefficients with
coefbwd(mnir)
coeffwd(mnir)

# the fitted model can be used to predict Food with new data
yhat = predict(mnir, covarsdf[1:10,:], counts[1:10,:])

# Suppose instead we want to predict a discrete variable, then perhaps use a Poisson GLM as follows
mnir = fit(CIR{DMR,GeneralizedLinearModel},mf,covarsdf,counts,:Food,Poisson())




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
@time focus(A)
@time focus2(A)
@time focus(A; j=3)
@time focus2(A; j=3)

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
