# performance evaluation stats
"Mean squared error"
mse(y::V,yhat::V) where {T<:Number,V<:AbstractVector{T}} = mean(abs2.(y-yhat))
mse(y::V,yhat::V) where {T<:AbstractVector,V<:AbstractVector{T}} = mse(vcat(y...),vcat(yhat...))

"Root-mean squared error"
rmse(args...) = sqrt(mse(args...))

StatsBase.r2(y::V,yhat::V) where {T<:Number,V<:AbstractVector{T}} = 1-sum(abs2,y-yhat)/sum(abs2,y.-mean(y))
StatsBase.r2(y::V,yhat::V) where {T<:AbstractVector,V<:AbstractVector{T}} = r2(vcat(y...),vcat(yhat...))

"Standard error of mean squared error"
σmse(y::V,yhat::V) where {T<:AbstractVector,V<:AbstractVector{T}} = std(mse.(y,yhat)) / sqrt(length(y)-1)

"Standard error of root-mean squared error"
σrmse(y::V,yhat::V) where {T<:AbstractVector,V<:AbstractVector{T}} = σmse(y,yhat) / (2rmse(y,yhat))

"Standard error of difference in mean squared error between yhat1 and yhat2"
σΔmse(y::V,yhat1::V,yhat2::V) where {T<:AbstractVector,V<:AbstractVector{T}} = std(mse.(y,yhat1) .- mse.(y,yhat2)) / sqrt(length(y)-1)

"Standard error of difference in root-mean squared error between yhat1 and yhat2"
σΔrmse(y::V,yhat1::V,yhat2::V) where {T<:AbstractVector,V<:AbstractVector{T}} = std(rmse.(y,yhat1) .- rmse.(y,yhat2)) / sqrt(length(y)-1)

"""
Non-random K-fold generator that simply splits into consequtive blocks of data.
Useful for time-series data.
"""
immutable SerialKfold <: CrossValGenerator
    n::Int
    k::Int
    foldlength::Int

    function SerialKfold(n::Int, k::Int)
        2 <= k <= n || error("The value of k must be in [2, n].")
        new(n, k, round(Int,n/k))
    end
end

Base.length(c::SerialKfold) = c.k
Base.start(c::SerialKfold) = 1
Base.next(c::SerialKfold, i::Int) = (vcat(1:(i-1)*c.foldlength,i*c.foldlength+1:c.n) , i+1)
Base.done(c::SerialKfold, i::Int) = (i > c.k)
# vcat(1:0,3:10)
# collect(SerialKfold(13,3))

"Splits the sample into one train and one test subsamples."
immutable LeaveOutSample <: CrossValGenerator
  n::Int
  testlength::Int
  random::Bool
  forward::Bool

  function LeaveOutSample(n::Int, k::Int=2; testlength::Int=round(Int,n/k), random=false, forward=true)
    1 <= testlength < n || error("The value of testlength ($testlength) must be in [1, n-1].")
    new(n, testlength, random, forward)
  end
end

Base.length(c::LeaveOutSample) = 1
Base.start(c::LeaveOutSample) = 1
function Base.next(c::LeaveOutSample, i::Int)
  trainlength = c.n-c.testlength
  if c.random
    sample(1:c.n, trainlength; replace=false, ordered=true), i+1
  elseif c.forward
    collect(1:trainlength), i+1
  else
    collect(c.n-trainlength+1:c.n), i+1
  end
end
Base.done(c::LeaveOutSample, i::Int) = (i > 1)
# vcat(1:0,3:10)
# collect(SerialKfold(13,3))

abstract type CVType{T} end

struct CVDataRow{T} <: CVType{T}
  ins_y::AbstractVector{T}
  oos_y::AbstractVector{T}
  ins_yhat::AbstractVector{T}
  oos_yhat::AbstractVector{T}
  ins_yhat_nocounts::AbstractVector{T}
  oos_yhat_nocounts::AbstractVector{T}
end

const VV{T} = Vector{Vector{T}}

"Container for cross validation y and yhats"
mutable struct CVData{T} <: CVType{T}
  ins_ys::Vector{AbstractVector{T}}
  oos_ys::Vector{AbstractVector{T}}
  ins_yhats::Vector{AbstractVector{T}}
  oos_yhats::Vector{AbstractVector{T}}
  ins_yhats_nocounts::Vector{AbstractVector{T}}
  oos_yhats_nocounts::Vector{AbstractVector{T}}

  CVData(::Type{T}) where T = new{T}(VV{T}(),VV{T}(),VV{T}(),VV{T}(),VV{T}(),VV{T}())

  # Constructor taking a a vector of results vector
  CVData(ins_ys::VV{T},oos_ys::VV{T},ins_yhats::VV{T},oos_yhats::VV{T},
    ins_yhats_nocounts::VV{T},oos_yhats_nocounts::VV{T}) where T =
    new{T}(ins_ys,oos_ys,ins_yhats,oos_yhats,ins_yhats_nocounts,oos_yhats_nocounts)

  # Constructor taking a single results vector
  CVData(ins_y::V,oos_y::V,ins_yhat::V,oos_yhat::V,
    ins_yhat_nocounts::V,oos_yhat_nocounts::V) where {T, V<:AbstractVector{T}} =
    new{T}([ins_y],[oos_y],[ins_yhat],[oos_yhat],[ins_yhat_nocounts],[oos_yhat_nocounts])
end

function Base.append!(d::CVData, r::CVDataRow)
  push!(d.ins_ys,r.ins_y)
  push!(d.oos_ys,r.oos_y)
  push!(d.ins_yhats,r.ins_yhat)
  push!(d.oos_yhats,r.oos_yhat)
  push!(d.ins_yhats_nocounts,r.ins_yhat_nocounts)
  push!(d.oos_yhats_nocounts,r.oos_yhat_nocounts)
  d
end

function Base.append!(d::CVData, r::CVData)
  append!(d.ins_ys,r.ins_ys)
  append!(d.oos_ys,r.oos_ys)
  append!(d.ins_yhats,r.ins_yhats)
  append!(d.oos_yhats,r.oos_yhats)
  append!(d.ins_yhats_nocounts,r.ins_yhats_nocounts)
  append!(d.oos_yhats_nocounts,r.oos_yhats_nocounts)
  d
end

"Params are equal if all their fields are equal"
function Base.isequal(x::T,y::T) where {T <: CVType}
  all(broadcast(field->isequal(getfield(x,field),getfield(y,field)),fieldnames(T)))
end

"Params are approximatly equal if all their fields are equal"
function Base.isapprox(x::T,y::T; kwargs...) where {T <: CVType}
  all(broadcast(field->isapprox(getfield(x,field),getfield(y,field);kwargs...),fieldnames(T)))
end

"Params have the same hash if all their fields have the same hash"
function Base.hash(a::T, h::UInt=zero(UInt)) where {T <: CVType}
  recursiveh = h
#   display("fieldnames=$(fieldnames(Params))")
  for field=fieldnames(T)
    recursiveh=hash(getfield(a,field), recursiveh)
  end
  recursiveh
end

"""
    cvstats(D, cvdata; stats=[:rmse, :r2])

Create a dictionary with fit statistics (mse, rmse, r2) from `CVData`.

# Example
```julia
  s = cvstats(OrderedDict, cvdata; stats=[:mse])
  s[:oos_mse]
```

# Keywords
- `stats::Vector{Symbol}` one or more of `:mse`, `:rmse`, or `:r2`
"""
function cvstats(::Type{D}, d::CVData; stats=[:rmse, :r2]) where {D<:Associative}

  s = D{Symbol,Float64}()

  for smse in intersect(stats, [:rmse,:mse])
    msefn = eval(smse)
    σmsefn = eval(Symbol(:σ,smse))
    σΔmsefn = eval(Symbol(:σΔ,smse))

    for ios in [:ins, :oos]
      for cnc in ["", "_nocounts"]
        s[Symbol(ios,"_",smse,cnc)] = msefn(getfield(d,Symbol(ios,"_ys")), getfield(d,Symbol(ios,"_yhats",cnc)))
        s[Symbol(ios,"_σ",smse,cnc)] = σmsefn(getfield(d,Symbol(ios,"_ys")), getfield(d,Symbol(ios,"_yhats",cnc)))
      end

      s[Symbol(ios,"_change_",smse)] = s[Symbol(ios,"_",smse)] - s[Symbol(ios,"_",smse,"_nocounts")]
      s[Symbol(ios,"_σchange_",smse)] = σΔmsefn(getfield(d,Symbol(ios,"_ys")), getfield(d,Symbol(ios,"_yhats")), getfield(d,Symbol(ios,"_yhats_nocounts")))
    end
  end

  if contains(==, stats, :r2)
    for ios in [:ins, :oos]
      for cnc in ["", "_nocounts"]
        s[Symbol(ios,"_r2",cnc)] = r2(getfield(d,Symbol(ios,"_ys")), getfield(d,Symbol(ios,"_yhats",cnc)))
      end

      s[Symbol(ios,"_change_r2")] = s[Symbol(ios,"_r2")] - s[Symbol(ios,"_r2_nocounts")]
    end
  end

  s
end

"When no dictionary type is specified, uses a simple Dict"
cvstats(d::CVData; kwargs...) = cvstats(Dict, d::CVData; kwargs...)

function initcv(gen,seed,gentype,n,k)
  # instantiate generator if not specified directly
  if gen == nothing
    # seed so that all different specs use same set of folds
    srand(seed)

    # fold generator
    gen = gentype(n,k)
  end

  # allocate space
  cvd = CVData(Float64)

  gen, cvd
end

"""
    cv(::CIR,covars,counts,projdir[,fmargs...]; <keyword arguments>)

Cross-validates a Counts Inverse Regression (CIR) of
  `covars[:,projdir] ~ counts + covars[:,~projdir]`. See also [`fit(::CIR)`](@ref).


# Example with Lasso regularization:
```julia
  cvdata = cv(CIR{HDMR,LinearModel},covars,counts,1; k=5, gentype=MLBase.Kfold, seed=123)
  CVStats(cvdata)
```

# Arguments
- `covars` n-by-p matrix of covariates
- `counts` n-by-d matrix of counts (usually sparse)
- `projdir` index of covars column used as dependent variable in forward model
- `fmargs...` optional arguments passed along to the forward regression model

# Keywords
- `gentype=Kfold` cross validation fold generator [`CrossValGenerator`](@ref)
- `k=10` number of folds
- `seed=13` random seed for generating folds
- `gen=nothing` if specified, uses this instantiated cross validation fold generator
  and disregards `gentype`, `k`, and `seed`
- `dcrkwargs...` additional keyword arguments passed along to backward regression model
"""
function cv(::Type{C},covars::AbstractMatrix{T},counts::AbstractMatrix{V},projdir::Int, fmargs...;
  gentype=Kfold, k=10, seed=13, gen=nothing, verbose=true,
  dcrkwargs...) where {T<:AbstractFloat,V,BM<:DCR,FM<:RegressionModel,C<:CIR{BM,FM}}

  # dims
  n,p = size(covars)

  #init
  gen, cvd = initcv(gen,seed,gentype,n,k)

  # k may have changed in init if gen was specified directly
  k = length(gen)

  # run cv
  for (i, ixtrain) in enumerate(gen)
      ixtest = setdiff(1:n, ixtrain)

      # estimate dmr in train subsample
      cir = fit(C,covars[ixtrain,:],counts[ixtrain,:],projdir,fmargs...; nocounts=true, verbose=verbose, dcrkwargs...)

      # target variable
      ins_y = covars[ixtrain,projdir]

      # benchmark model w/o text
      ins_yhat_nocounts = predict(cir,covars[ixtrain,:],counts[ixtrain,:]; nocounts=true)

      # model with text
      ins_yhat = predict(cir,covars[ixtrain,:],counts[ixtrain,:]; nocounts=false)

      # evaluate out-of-sample in test subsample
      # target variable
      oos_y = covars[ixtest,projdir]

      # benchmark model w/o text
      oos_yhat_nocounts = predict(cir,covars[ixtest,:],counts[ixtest,:]; nocounts=true)

      # dmr model w/ text
      oos_yhat = predict(cir,covars[ixtest,:],counts[ixtest,:]; nocounts=false)

      # save results
      append!(cvd, CVDataRow(ins_y,oos_y,ins_yhat,oos_yhat,ins_yhat_nocounts,oos_yhat_nocounts))

      verbose && info("estimated fold $i/$k")
  end

  verbose && info("calculated aggreagate fit for $(length(cvd.ins_ys)) in-sample and $(length(cvd.oos_ys)) out-of-sample total observations (with duplication).")

  cvd
end

function cv(::Type{C}, m::Model, df::AbstractDataFrame, counts::AbstractMatrix, sprojdir::Symbol, fmargs...;
  contrasts::Dict = Dict(), kwargs...) where {BM<:DMR,FM<:RegressionModel,C<:CIR{BM,FM}}
  # parse and merge rhs terms
  trms = getrhsterms(m, :c)

  # create model matrix
  mf, mm, counts = createmodelmatrix(trms, df, counts, contrasts)

  # resolve projdir
  projdir = ixprojdir(trms, sprojdir)

  # delegates but does not wrap in DataFrameRegressionModel
  cv(C, mm.m, counts, projdir, fmargs...; kwargs...)
end

function cv(::Type{C}, m::Model, df::AbstractDataFrame, counts::AbstractMatrix, sprojdir::Symbol, fmargs...;
  contrasts::Dict = Dict(), kwargs...) where {BM<:HDMR,FM<:RegressionModel,C<:CIR{BM,FM}}
  # parse and merge rhs terms
  trmszero = getrhsterms(m, :h)
  trmspos = getrhsterms(m, :c)
  trms, inzero, inpos = mergerhsterms(trmszero,trmspos)

  # create model matrix
  mf, mm, counts = createmodelmatrix(trms, df, counts, contrasts)

  # resolve projdir
  projdir = ixprojdir(trms, sprojdir)

  # delegates but does not wrap in DataFrameRegressionModel
  cv(C, mm.m, counts, projdir, fmargs...; inzero=inzero, inpos=inpos, kwargs...)
end
