# HurdleDMR.jl

HurdleDMR.jl is a Julia implementation of the Hurdle Distributed Multiple Regression (HDMR), as described in:

Kelly, Bryan, Asaf Manela, and Alan Moreira (2018). Text Selection. Working paper

It includes a Julia implementation of the Distributed Multinomial Regression (DMR) model of Taddy (2015).

```@contents
```

## Setup

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
using CSV, GLM, DataFrames, Distributions
we8thereCounts = CSV.read(joinpath(Pkg.dir("HurdleDMR"),"test","data","dmr_we8thereCounts.csv.gz"))
counts = sparse(convert(Matrix{Float64},we8thereCounts))
covarsdf = CSV.read(joinpath(Pkg.dir("HurdleDMR"),"test","data","dmr_we8thereRatings.csv.gz"))
covars = convert(Matrix{Float64},covarsdf)
terms = map(string,names(we8thereCounts))
```


## Distributed Multinomial Regression (DMR)

The Distributed Multinomial Regression (DMR) model of Taddy (2015) is a highly scalable
approximation to the Multinomial using distributed (independent, parallel)
Poisson regressions, one for each of the d categories (columns) of a large `counts` matrix,
on the `covars`.

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

```@autodocs
Modules = [HurdleDMR]
Order   = [:macro, :type, :function]
Pages   = ["src/dmr.jl"]
Private = false
```

## Hurdle Distributed Multiple Regression (HDMR)

For highly sparse counts, as is often the case with text that is selected for
various reasons, the Hurdle Distributed Multiple Regression (HDMR) model of
Kelly, Manela, and Moreira (2018), may be superior to the DMR. It approximates
a higher dispersion Multinomial using distributed (independent, parallel)
Hurdle regressions, one for each of the d categories (columns) of a large `counts` matrix,
on the `covars`. It allows a potentially different sets of covariates to explain
category inclusion ($h=1{c>0}$), and repetition ($c>0$).

Both the model for zeroes and for positive counts are regularized by default,
using [`GammaLassoPath`](@ref), picking the AICc optimal segment of the regularization
path.

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

Syntax:
```@autodocs
Modules = [HurdleDMR]
Order   = [:macro, :type, :function]
Pages   = ["src/hdmr.jl"]
Private = false
```

## Sufficient reduction projection

A sufficient reduction projection summarizes the counts, much like a sufficient
statistic, and is useful for reducing the d dimensional counts in a potentially
much lower dimension matrix `z`.

To get a sufficient reduction projection in direction of Food for the above
example
```julia
z = srproj(m,counts,1,1)
```

Syntax:
```@autodocs
Modules = [HurdleDMR]
Order   = [:macro, :type, :function]
Pages   = ["src/srproj.jl"]
Private = false
```
## Counts Inverse Regression (CIR)

Counts inverse regression allows us to predict a covariate with the counts and other covariates.
Here we use hdmr for the backward regression and another model for the forward regression.
This can be accomplished with a single command, by fitting a CIR{HDMR,FM} where the forward model is FM <: RegressionModel.
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

Syntax:
```@autodocs
Modules = [HurdleDMR]
Order   = [:macro, :type, :function]
Pages   = ["src/invreg.jl"]
Private = false
```

## Cross-validation utilities

Syntax:
```@autodocs
Modules = [HurdleDMR]
Order   = [:macro, :type, :function]
Pages   = ["src/cross_validation.jl"]
Private = false
```

## Hurdle
This package also provides a regularized Hurdle model (Mullahy, 1986) that can be
fit using a fast coordinate decent algorithm, or simply by running two
`fit(GeneralizedLinearModel,...)` regressions, one for each of its two parts.

Syntax:
```@autodocs
Modules = [HurdleDMR]
Order   = [:macro, :type, :function]
Pages   = ["src/cross_validation.jl"]
Private = false
```

## Positive Poisson
This package also implements the `PositivePoisson` distribution and the GLM
necessary methods to facilitate fit with [`fit(::GeneralizedLinearModel`](@ref).

Syntax:
```@autodocs
Modules = [HurdleDMR]
Order   = [:macro, :type, :function]
Pages   = ["src/positive_poisson.jl"]
Private = false
```

## API / Index

```@index
```
