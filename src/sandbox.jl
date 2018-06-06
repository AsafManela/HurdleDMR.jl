# Install the HurdleDMR package
#Pkg.clone("https://github.com/AsafManela/HurdleDMR.jl")

# Add parallel workers and make package available to workers
addprocs(Sys.CPU_CORES-2)
import HurdleDMR; @everywhere using HurdleDMR

# Setup your data into an n-by-p covars matrix, and a (sparse) n-by-d counts matrix
using CSV, GLM, DataFrames, Distributions
we8thereCounts = CSV.read(joinpath(Pkg.dir("HurdleDMR"),"test","data","dmr_we8thereCounts.csv.gz"))
counts = sparse(convert(Matrix{Float64},we8thereCounts))
covars = convert(Matrix{Float64},CSV.read(joinpath(Pkg.dir("HurdleDMR"),"test","data","dmr_we8thereRatings.csv.gz")))
terms = map(string,names(we8thereCounts))

# To fit a distribtued multinomial regression (dmr):
coefs = HurdleDMR.dmr(covars, counts)

# To use the dmr coefficients for a forward regression:
projdir = 1
X, X_nocounts = HurdleDMR.srprojX(coefs,counts,covars,projdir)
y = covars[:,projdir]

# benchmark model w/o text
insamplelm_nocounts = lm(X_nocounts,y)
yhat_nocounts = predict(insamplelm_nocounts,X_nocounts)

# dmr model w/ text
insamplelm = lm(X,y)
yhat = predict(insamplelm,X)

## To fit a hurdle distribtued multinomial regression (hdmr):
inzero = 1:size(covars,2)
inpos = [1,3]

# covariates that go into the model for positive counts
covarspos = covars[:,inpos]

# covariates that go into the hurdle crossing model for indicators
covarszero = covars[:,inzero]

# run the backward regression
coefspos, coefszero = HurdleDMR.hdmr(covarszero, counts; covarspos=covarspos)

## We can now use the hdmr coefficients for a forward regression:
# collapse counts into low dimensional SR projection + covars
X, X_nocounts, includezpos = HurdleDMR.srprojX(coefspos, coefszero, counts, covars, projdir; inpos=inpos, inzero=inzero)

# benchmark model w/o text
insamplelm_nocounts = lm(X_nocounts,y)
yhat_nocounts = predict(insamplelm_nocounts,X_nocounts)

# dmr model w/ text
insamplelm = lm(X,y)
yhat = predict(insamplelm,X)




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

A = spzeros(Float64,4,5)
convert(SharedArray, A)
A = zeros(Float64,4,5)
@time convert(SharedArray{Float64}, A)
@time convert(SharedArray, A)

function fun(x; kwargs...)
  kwargs
end
kwargs = fun(3;b=1,c=2)
kwargsdict = Dict(kwargs)
haskey(kwargsdict,:b)
kwargsdict[:a]
getindex(kwargsdict,:b)
