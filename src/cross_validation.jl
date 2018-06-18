using MLBase, StatsBase, DataFrames

# performance evaluation stats
mse(y::V,yhat::V) where {T<:Number,V<:AbstractVector{T}} = mean(abs2.(y-yhat))
mse(y::V,yhat::V) where {T<:AbstractVector,V<:AbstractVector{T}} = mse(vcat(y...),vcat(yhat...))
StatsBase.r2(y::V,yhat::V) where {T<:Number,V<:AbstractVector{T}} = 1-sum(abs2,y-yhat)/sum(abs2,y.-mean(y))
StatsBase.r2(y::V,yhat::V) where {T<:AbstractVector,V<:AbstractVector{T}} = r2(vcat(y...),vcat(yhat...))
rmse(args...) = sqrt(mse(args...))
# rmse(y::V,yhat::V) where {T<:Number,V<:AbstractVector{T}} = sqrt(mean(abs2.(y-yhat)))
# rmse(y::V,yhat::V) where {T<:AbstractVector,V<:AbstractVector{T}} = rmse(vcat(y...),vcat(yhat...))
σmse(y::V,yhat::V) where {T<:AbstractVector,V<:AbstractVector{T}} = std(mse.(y,yhat)) / sqrt(length(y)-1)
σrmse(y::V,yhat::V) where {T<:AbstractVector,V<:AbstractVector{T}} = σmse(y,yhat) / (2rmse(y,yhat))
σΔmse(y::V,yhat1::V,yhat2::V) where {T<:AbstractVector,V<:AbstractVector{T}} = std(mse.(y,yhat1) .- mse.(y,yhat2)) / sqrt(length(y)-1)
σΔrmse(y::V,yhat1::V,yhat2::V) where {T<:AbstractVector,V<:AbstractVector{T}} = std(rmse.(y,yhat1) .- rmse.(y,yhat2)) / sqrt(length(y)-1)


# non-random K-fold that simply splits into consequtive blocks of data
# useful for time-series CV
immutable SerialKfold <: CrossValGenerator
    n::Int
    k::Int
    foldlength::Int

    function SerialKfold(n::Int, k::Int)
        2 <= k <= n || error("The value of k must be in [2, length(a)].")
        new(n, k, round(Int,n/k))
    end
end

Base.length(c::SerialKfold) = c.k
Base.start(c::SerialKfold) = 1
Base.next(c::SerialKfold, i::Int) = (vcat(1:(i-1)*c.foldlength,i*c.foldlength+1:c.n) , i+1)
Base.done(c::SerialKfold, i::Int) = (i > c.k)
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

mutable struct CVData{T} <: CVType{T}
  ins_ys::Vector{AbstractVector{T}}
  oos_ys::Vector{AbstractVector{T}}
  ins_yhats::Vector{AbstractVector{T}}
  oos_yhats::Vector{AbstractVector{T}}
  ins_yhats_nocounts::Vector{AbstractVector{T}}
  oos_yhats_nocounts::Vector{AbstractVector{T}}
end

const VV{T} = Vector{Vector{T}}

CVData(::Type{T}) where T = CVData{T}(VV{T}(),VV{T}(),VV{T}(),VV{T}(),VV{T}(),VV{T}())

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


mutable struct CVStats{T} <: CVType{T}
  oos_mse::T
  oos_mse_nocounts::T
  oos_change_mse::T
  oos_σmse::T
  oos_σmse_nocounts::T
  oos_σchange_mse::T
  ins_mse::T
  ins_mse_nocounts::T
  ins_change_mse::T
  ins_σmse::T
  ins_σmse_nocounts::T
  ins_σchange_mse::T
  oos_r2::T
  oos_r2_nocounts::T
  oos_change_r2::T
  ins_r2::T
  ins_r2_nocounts::T
  ins_change_r2::T
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

"Create a DataFrame from a CVType instance"
function DataFrames.DataFrame(x::T) where {T <: CVType}
  fnames = fieldnames(T)
  fvalues = [[getfield(x,f)] for f = fnames]
  DataFrames.DataFrame(fvalues,fnames)
end

"Create a DataFrame from a vector of CVTypes"
function DataFrames.DataFrame(v::Vector{T}) where {T <: CVType}
    vcat(DataFrames.DataFrame.(v)...)
end

CVStats(T::Type) = CVStats{T}(zeros(T,18)...)

function CVStats{T}(d::CVData{T}; root=false)

  s = CVStats(T)

  # we kept the entire vectors of y/yhats so we can calculate r2s correctly
  s.ins_r2_nocounts = r2(d.ins_ys,d.ins_yhats_nocounts)
  s.ins_r2 = r2(d.ins_ys,d.ins_yhats)
  s.oos_r2_nocounts = r2(d.oos_ys,d.oos_yhats_nocounts)
  s.oos_r2 = r2(d.oos_ys,d.oos_yhats)

  if root
    msefn = rmse
    σmsefn = σrmse
    σΔmsefn = σΔrmse
  else
    msefn = mse
    σmsefn = σmse
    σΔmsefn = σΔmse
  end

  s.ins_mse_nocounts = msefn(d.ins_ys,d.ins_yhats_nocounts)
  s.ins_mse = msefn(d.ins_ys,d.ins_yhats)
  s.oos_mse_nocounts = msefn(d.oos_ys,d.oos_yhats_nocounts)
  s.oos_mse = msefn(d.oos_ys,d.oos_yhats)

  s.ins_σmse_nocounts = σmsefn(d.ins_ys,d.ins_yhats_nocounts)
  s.ins_σmse = σmsefn(d.ins_ys,d.ins_yhats)
  s.oos_σmse_nocounts = σmsefn(d.oos_ys,d.oos_yhats_nocounts)
  s.oos_σmse = σmsefn(d.oos_ys,d.oos_yhats)

  # mse changes
  s.oos_change_mse = s.oos_mse - s.oos_mse_nocounts
  s.ins_change_mse = s.ins_mse - s.ins_mse_nocounts
  s.oos_change_r2 = s.oos_r2 - s.oos_r2_nocounts
  s.ins_change_r2 = s.ins_r2 - s.ins_r2_nocounts
  s.oos_σchange_mse = σΔmsefn(d.oos_ys,d.oos_yhats,d.oos_yhats_nocounts)
  s.ins_σchange_mse = σΔmsefn(d.ins_ys,d.ins_yhats,d.ins_yhats_nocounts)

  # # vec and transpose are unnecessary, but for code sanity
  # vec([oos_rmse oos_rmse_nocounts oos_pct_change_rmse
  #      ins_rmse ins_rmse_nocounts ins_pct_change_rmse
  #      oos_r2 oos_r2_nocounts oos_pct_change_r2
  #      ins_r2 ins_r2_nocounts ins_pct_change_r2]')'

  s
end

function initcv(seed,gentype,n,k)
  # seed so that all different specs use same set of folds
  srand(seed)

  # fold generator
  gen = gentype(n,k)

  # allocate space
  cvd = CVData(Float64)

  gen, cvd
end

function cv(::Type{C},covars::AbstractMatrix{T},counts::AbstractMatrix{V},projdir::Int, fmargs...;
  k=10, gentype=Kfold, seed=13,
  dcrkwargs...) where {T<:AbstractFloat,V,BM<:DCR,FM<:RegressionModel,C<:CIR{BM,FM}}

  # dims
  n,p = size(covars)
  # ixnotdir = 1:p .!= projdir

  # init
  gen, cvd = initcv(seed,gentype,n,k)

  # run cv
  for (i, ixtrain) in enumerate(gen)
      ixtest = setdiff(1:n, ixtrain)

      # estimate dmr in train subsample
      cir = fit(C,covars[ixtrain,:],counts[ixtrain,:],projdir,fmargs...; nocounts=true, dcrkwargs...)

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

      info("estimated fold $i/$k")
  end

  info("calculated aggreagate fit for $(length(cvd.ins_ys)) in-sample and $(length(cvd.oos_ys)) out-of-sample total observations (with duplication).")

  cvd
end

function cv(::Type{C}, m::Model, df::AbstractDataFrame, counts::AbstractMatrix, sprojdir::Symbol, fmargs...;
  contrasts::Dict = Dict(), kwargs...) where {BM<:DMR,FM<:RegressionModel,C<:CIR{BM,FM}}
  # parse and merge rhs terms
  trms = getrhsterms(m, :c)

  # create model matrix
  mf, mm = createmodelmatrix(trms, df, contrasts)

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
  mf, mm = createmodelmatrix(trms, df, contrasts)

  # resolve projdir
  projdir = ixprojdir(trms, sprojdir)

  # delegates but does not wrap in DataFrameRegressionModel
  cv(C, mm.m, counts, projdir, fmargs...; inzero=inzero, inpos=inpos, kwargs...)
end
