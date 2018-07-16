# HurdleDMR

| Documentation | Linux/MacOS | Windows | Code |
| --- | --- | --- | --- |
| [![][docs-latest-img]][docs-latest-url] | [![][travis-img]][travis-url]  | [![][appveyor-img]][appveyor-url]  | [![][coveralls-img]][coveralls-url] |

<!-- | **Package Evaluator**   | [![][pkg-0.5-img]][pkg-0.5-url] [![][pkg-0.6-img]][pkg-0.6-url] | -->

HurdleDMR.jl is a Julia implementation of the Hurdle Distributed Multiple Regression (HDMR), as described in:

Kelly, Bryan, Asaf Manela, and Alan Moreira (2018). Text Selection. [Working paper](http://apps.olin.wustl.edu/faculty/manela/kmm/textselection).

It includes a Julia implementation of the Distributed Multinomial Regression (DMR) model of [Taddy (2015)](https://arxiv.org/abs/1311.6139).

## Quick start

Install the HurdleDMR package
```julia
Pkg.clone("https://github.com/AsafManela/Lasso.jl")
Pkg.clone("https://github.com/AsafManela/HurdleDMR.jl")
```

Add parallel workers and make package available to workers
```julia
addprocs(Sys.CPU_CORES-2)
import HurdleDMR; @everywhere using HurdleDMR
```

Setup your data into an n-by-p covars matrix, and a (sparse) n-by-d counts matrix.
Here we generate some random data.
```julia
using CSV, GLM, DataFrames, Distributions
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
covarsdf = DataFrame(covars,[:vy, :v1, :v2])
```

### Hurdle distributed multiple regression (HDMR)
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

To get a sufficient reduction projection in direction of vy
```julia
z = srproj(m,counts,1,1)
```

### Counts inverse regression (CIR)
Counts inverse regression allows us to predict a covariate with the counts and other covariates
Here we use hdmr for the backward regression and another model for the forward regression
This can be accomplished with a single command, by fitting a CIR{HDMR,FM} where the forward model is FM <: RegressionModel
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
yhat_nocounts = predict(hir, covarsdf[1:10,:], counts[1:10,:]; nocounts=true)
```

### Distribtued multinomial regression (DMR)
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

To get a sufficient reduction projection in direction of vy
```julia
z = srproj(m,counts,1)
```

### Multinomial inverse regression (MNIR)
MNIR is a special case of CIR that uses DMR for the backward regression. This can be accomplished by fitting a CIR{DMR,FM} model
```julia
mnir = fit(CIR{DMR,LinearModel},mf,covarsdf,counts,:vy)
```

The fitted model can be used to predict vy with new data
```julia
yhat = predict(mnir, covarsdf[1:10,:], counts[1:10,:])
```

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://asafmanela.github.io/HurdleDMR.jl/latest

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://asafmanela.github.io/HurdleDMR.jl/stable

[travis-img]: https://travis-ci.org/AsafManela/HurdleDMR.jl.svg?branch=master
[travis-url]: https://travis-ci.org/AsafManela/HurdleDMR.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/hurdledmr-jl?svg=true
[appveyor-url]: https://ci.appveyor.com/project/AsafManela/hurdledmr-jl

[coveralls-img]: https://coveralls.io/repos/AsafManela/HurdleDMR.jl/badge.svg?branch=master
[coveralls-url]: https://coveralls.io/r/AsafManela/HurdleDMR.jl?branch=master

[pkg-0.6-img]: http://pkg.julialang.org/badges/HurdleDMR_0.6.svg
[pkg-0.6-url]: http://pkg.julialang.org/?pkg=HurdleDMR&ver=0.6
[pkg-0.7-img]: http://pkg.julialang.org/badges/HurdleDMR_0.7.svg
[pkg-0.7-url]: http://pkg.julialang.org/?pkg=HurdleDMR&ver=0.7
