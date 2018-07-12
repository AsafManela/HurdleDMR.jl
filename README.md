# HurdleDMR

<!-- [![Build Status](https://travis-ci.org/AsafManela/HurdleDMR.jl.svg?branch=master)](https://travis-ci.org/AsafManela/HurdleDMR.jl)
[![Coverage Status](https://coveralls.io/repos/AsafManela/HurdleDMR.jl/badge.svg?branch=master)](https://coveralls.io/r/AsafManela/HurdleDMR.jl?branch=master) -->

HurdleDMR.jl is a Julia implementation of the Hurdle Distributed Multiple Regression (HDMR), as described in:

Kelly, Bryan, Asaf Manela, and Alan Moreira (2018). Text Selection. Working paper

It includes a Julia implementation of the Distributed Multinomial Regression (DMR) model of Taddy (2015).

## Quick start

Install the HurdleDMR package
```julia
Pkg.clone("https://github.com/AsafManela/Lasso.jl")
Pkg.clone("https://github.com/AsafManela/HurdleDMR.jl")
```
(You may get warnings about other missing packages. Just add those too.)

Add parallel workers and make package available to workers
```julia
addprocs(Sys.CPU_CORES-2)
import HurdleDMR; @everywhere using HurdleDMR
```

Setup your data into an n-by-p covars matrix, and a (sparse) n-by-d counts matrix.
```julia
using CSV, GLM, DataFrames, Distributions
we8thereCounts = CSV.read(joinpath(Pkg.dir("HurdleDMR"),"test","data","dmr_we8thereCounts.csv.gz"))
counts = sparse(convert(Matrix{Float64},we8thereCounts))
covarsdf = CSV.read(joinpath(Pkg.dir("HurdleDMR"),"test","data","dmr_we8thereRatings.csv.gz"))
covars = convert(Matrix{Float64},covarsdf)
terms = map(string,names(we8thereCounts))
```

### Hurdle distributed multiple regression (HDMR)
HDMR can be fitted:
```julia
m = hdmr(covars, counts; inpos=[1,3], inzero=1:5)
```

or with a dataframe and formula
```julia
mf = @model(h ~ Food + Service + Value + Atmosphere + Overall, c ~ Food + Value)
m = fit(HDMR, mf, covarsdf, counts)
```
where the h ~ equation is the model for zeros (hurdle crossing) and c ~ is the model for positive counts

in either case we can get the coefficients matrix for each variable + intercept as usual with
```julia
coefspos, coefszero = coef(m)
```

By default we only return the AICc maximizing coefficients.
To also get back the entire regulatrization paths, run
```julia
paths = fit(HDMRPaths, mf, covarsdf, counts)

coef(paths; select=:all)
```

To get a sufficient reduction projection in direction of Food
```julia
z = srproj(m,counts,1,1)
```

### Counts inverse regression (CIR)
Counts inverse regression allows us to predict a covariate with the counts and other covariates
Here we use hdmr for the backward regression and another model for the forward regression
This can be accomplished with a single command, by fitting a CIR{HDMR,FM} where the forward model is FM <: RegressionModel
```julia
cir = fit(CIR{HDMR,LinearModel},mf,covarsdf,counts,:Food; nocounts=true)
```
where the ```nocounts=true``` means we also fit a benchmark model without counts.

we can get the forward and backward model coefficients with
```julia
coefbwd(cir)
coeffwd(cir)
```

The fitted model can be used to predict Food with new data
```julia
yhat = predict(cir, covarsdf[1:10,:], counts[1:10,:])
```

We can also predict only with the other covariates, which in this case
is just a linear regression
```julia
yhat_nocounts = predict(hir, covarsdf[1:10,:], counts[1:10,:]; nocounts=true)
```

### Distribtued multinomial regression (DMR)
To fit a DMR:
```julia
m = dmr(covars, counts)
```
or with a dataframe and formula
```julia
m = fit(DMR, @model(c ~ Food + Service + Value + Atmosphere + Overall), covarsdf, counts)
```
in either case we can get the coefficients matrix for each variable + intercept as usual with
```julia
coef(m)
```

By default we only return the AICc maximizing coefficients.
To also get back the entire regulatrization paths, run
```julia
mf = @model(c ~ Food + Service + Value + Atmosphere + Overall)
paths = fit(DMRPaths, mf, covarsdf, counts)
```
we can now select, for example the coefficients that minimize CV mse (takes a while)
```julia
coef(paths; select=:CVmin)
```

To get a sufficient reduction projection in direction of Food
```julia
z = srproj(m,counts,1)
```

### Multinomial inverse regression (MNIR)
MNIR is a special case of CIR that uses DMR for the backward regression. This can be accomplished by fitting a CIR{DMR,FM} model
```julia
mnir = fit(CIR{DMR,LinearModel},mf,covarsdf,counts,:Food)
```

The fitted model can be used to predict Food with new data
```julia
yhat = predict(mnir, covarsdf[1:10,:], counts[1:10,:])
```
