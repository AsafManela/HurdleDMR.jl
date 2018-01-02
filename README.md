# HurdleDMR

<!-- [![Build Status](https://travis-ci.org/simonster/Lasso.jl.svg?branch=master)](https://travis-ci.org/simonster/Lasso.jl)
[![Coverage Status](https://coveralls.io/repos/simonster/Lasso.jl/badge.svg?branch=master)](https://coveralls.io/r/simonster/Lasso.jl?branch=master) -->

HurdledMR.jl is a Julia implementation of the Hurdle Distributed Multinomial Regression (HDMR), as described in:

Kelly, Bryan, Asaf Manela, and Alan Moreira (2017). Text Selection. Working paper

It includes a Julia implementation of the Distributed Multinomial Regression (DMR) model of Taddy (2015).

## Quick start

Install the HurdleDMR package
```julia
Pkg.clone("https://github.com/AsafManela/HurdleDMR.jl")
```

Add parallel workers and make package available to workers
```julia
addprocs(Sys.CPU_CORES-2)
import HurdleDMR; @everywhere using HurdleDMR
```

Setup your data into an n-by-p covars matrix, and a (sparse) n-by-d counts matrix.
```julia
using GLM, DataFrames, Distributions
we8thereCounts = readtable(joinpath(Pkg.dir("HurdleDMR"),"test","data","dmr_we8thereCounts.csv.gz"))
counts = sparse(convert(Matrix{Float64},we8thereCounts))
covars = convert(Matrix{Float64},readtable(joinpath(Pkg.dir("HurdleDMR"),"test","data","dmr_we8thereRatings.csv.gz")))
terms = map(string,names(we8thereCounts))
```

To fit a distribtued multinomial regression (dmr):
```julia
# fit dmr
coefs = HurdleDMR.dmr(covars, counts)
```

To use the dmr coefficients for an multinomial inverse regression:
```julia
# collapse counts into low dimensional SR projection in direction first covar + other covars
projdir = 1
X, X_nocounts = HurdleDMR.srprojX(coefs,counts,covars,projdir)
y = covars[:,projdir]

# benchmark model w/o text
insamplelm_nocounts = lm(X_nocounts,y)
yhat_nocounts = predict(insamplelm_nocounts,X_nocounts)

# dmr model w/ text
insamplelm = lm(X,y)
yhat = predict(insamplelm,X)
```

To fit a hurdle distributed multinomial regression (hdmr):
```julia

# pick covariates to go into model for positive counts and to model for zeros (hurdle crossing)
inzero = 1:size(covars,2)
inpos = [1,3]
covarspos = covars[:,inpos]
covarszero = covars[:,inzero]

# run the backward regression
coefspos, coefszero = HurdleDMR.hdmr(covarszero, counts; covarspos=covarspos)
```

We can now use the hdmr coefficients for a forward regression:
```julia
# collapse counts into low dimensional SR projection + covars
X, X_nocounts, includezpos = HurdleDMR.srprojX(coefspos, coefszero, counts, covars, projdir; inpos=inpos, inzero=inzero)

# benchmark model w/o text
insamplelm_nocounts = lm(X_nocounts,y)
yhat_nocounts = predict(insamplelm_nocounts,X_nocounts)

# dmr model w/ text
insamplelm = lm(X,y)
yhat = predict(insamplelm,X)

```


## TODO

 - Inverse regression interface

<!--More documentation is available at [ReadTheDocs](http://lassojl.readthedocs.org/en/latest/).



## See also

 - [GLMNet.jl](https://github.com/simonster/GLMNet.jl), a wrapper for the
   glmnet Fortran code.
 - [LARS.jl](https://github.com/simonster/LARS.jl), an implementation
   of least angle regression for fitting entire linear (but not
   generalized linear) Lasso and Elastic Net coordinate paths. -->
