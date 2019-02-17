# HurdleDMR.jl

HurdleDMR.jl is a Julia implementation of the Hurdle Distributed Multiple Regression (HDMR), as described in:

Kelly, Bryan, Asaf Manela, and Alan Moreira (2018). Text Selection. [Working paper](http://apps.olin.wustl.edu/faculty/manela/kmm/textselection/).

It includes a Julia implementation of the Distributed Multinomial Regression (DMR) model of [Taddy (2015)](https://arxiv.org/abs/1311.6139).

## Setup

Install the HurdleDMR package
```julia
pkg> add HurdleDMR
```

Add parallel workers and make package available to workers
```julia
using Distributed
addprocs(Sys.CPU_THREADS-2)
import HurdleDMR; @everywhere using HurdleDMR
```

Setup your data into an n-by-p covars matrix, and a (sparse) n-by-d counts matrix.
Here we generate some random data.
```julia
using CSV, GLM, DataFrames, Distributions, Random, LinearAlgebra, SparseArrays
n = 100
p = 3
d = 4

Random.seed!(13)
m = 1 .+ rand(Poisson(5),n)
covars = rand(n,p)
ηfn(vi) = exp.([0 + i*sum(vi) for i=1:d])
q = [ηfn(covars[i,:]) for i=1:n]
rmul!.(q,ones(n)./sum.(q))
counts = convert(SparseMatrixCSC{Float64,Int},hcat(broadcast((qi,mi)->rand(Multinomial(mi, qi)),q,m)...)')
covarsdf = DataFrame(covars,[:vy, :v1, :v2])
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
mf = @model(c ~ vy + v1 + v2)
m = fit(DMR, mf, covarsdf, counts)
```
in either case we can get the coefficients matrix for each variable + intercept as usual with
```julia
coef(m)
```

By default we only return the AICc maximizing coefficients.
To also get back the entire regulatrization paths, run
```julia
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
m = hdmr(covars, counts; inpos=1:2, inzero=1:3)
```

or with a dataframe and formula
```julia
mf = @model(h ~ vy + v1 + v2, c ~ vy + v1)
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

To get a sufficient reduction projection in direction of vy for the above
example
```julia
z = srproj(m,counts,1,1)
```
Here, the first column is the SR projection from the model for positive counts, the second is the the SR projection from the model for hurdle crossing (zeros), and the third is the total count for each observation.

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
cir = fit(CIR{HDMR,LinearModel},mf,covarsdf,counts,:vy; nocounts=true)
```
where the ```nocounts=true``` means we also fit a benchmark model without counts.

we can get the forward and backward model coefficients with
```julia
coefbwd(cir)
coeffwd(cir)
```

The fitted model can be used to predict vy with new data
```julia
yhat = predict(cir, covarsdf[1:10,:], counts[1:10,:])
```

We can also predict only with the other covariates, which in this case
is just a linear regression
```julia
yhat_nocounts = predict(cir, covarsdf[1:10,:], counts[1:10,:]; nocounts=true)
```

Syntax:
```@autodocs
Modules = [HurdleDMR]
Order   = [:macro, :type, :function]
Pages   = ["src/invreg.jl"]
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
Pages   = ["src/hurdle.jl"]
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
