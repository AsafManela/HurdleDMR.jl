# HurdleDMR.jl

HurdleDMR.jl is a Julia implementation of the Hurdle Distributed Multiple Regression (HDMR), as described in:

Kelly, Bryan, Asaf Manela, and Alan Moreira (2018). Text Selection. Working paper

It includes a Julia implementation of the Distributed Multinomial Regression (DMR) model of Taddy (2015).

```@contents
```

## Distributed Multinomial Regression (DMR)

```@autodocs
Modules = [HurdleDMR]
Order   = [:macro, :type, :function]
Pages   = ["src/dmr.jl"]
Private = false
```

## Hurdle Distributed Multiple Regression (HDMR)

```@autodocs
Modules = [HurdleDMR]
Order   = [:macro, :type, :function]
Pages   = ["src/hdmr.jl"]
Private = false
```

## Sufficient reduction projection

```@autodocs
Modules = [HurdleDMR]
Order   = [:macro, :type, :function]
Pages   = ["src/srproj.jl"]
Private = false
```
## Counts Inverse Regression (CIR)

```@autodocs
Modules = [HurdleDMR]
Order   = [:macro, :type, :function]
Pages   = ["src/invreg.jl"]
Private = false
```

## Cross-validation utilities

```@autodocs
Modules = [HurdleDMR]
Order   = [:macro, :type, :function]
Pages   = ["src/cross_validation.jl"]
Private = false
```

<!-- ```@docs
fit(::DCR)
predict(::DCR)
coef
srproj
dmr
dmrpaths
hdmr
hdmrpaths
``` -->

## Index

```@index
```
