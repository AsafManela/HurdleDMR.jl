using MLBase, StatsBase, DataFrames

# performance evaluation stats
mse(y::AbstractVector,yhat::AbstractVector) = mean(abs2.(y-yhat))
rmse(y::AbstractVector,yhat::AbstractVector) = sqrt(mean(abs2.(y-yhat)))
StatsBase.r2(y::AbstractVector,yhat::AbstractVector) = 1-sum(abs2,y-yhat)/sum(abs2,y.-mean(y))

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

struct CVData{T} <: CVType{T}
  ins_ys::AbstractVector{T}
  oos_ys::AbstractVector{T}
  ins_yhats::AbstractVector{T}
  oos_yhats::AbstractVector{T}
  ins_yhats_nocounts::AbstractVector{T}
  oos_yhats_nocounts::AbstractVector{T}
end


CVData(T::Type) = CVData{T}(Vector{T}(0),Vector{T}(0),Vector{T}(0),Vector{T}(0),Vector{T}(0),Vector{T}(0))

function Base.append!(d::CVData, r::CVDataRow)
  append!(d.ins_ys,r.ins_y)
  append!(d.oos_ys,r.oos_y)
  append!(d.ins_yhats,r.ins_yhat)
  append!(d.oos_yhats,r.oos_yhat)
  append!(d.ins_yhats_nocounts,r.ins_yhat_nocounts)
  append!(d.oos_yhats_nocounts,r.oos_yhat_nocounts)
end

mutable struct CVStats{T} <: CVType{T}
  oos_rmse::T
  oos_rmse_nocounts::T
  oos_pct_change_rmse::T
  ins_rmse::T
  ins_rmse_nocounts::T
  ins_pct_change_rmse::T
  oos_r2::T
  oos_r2_nocounts::T
  oos_pct_change_r2::T
  ins_r2::T
  ins_r2_nocounts::T
  ins_pct_change_r2::T
end

"Params are equal if all their fields are equal"
function Base.isequal(x::T,y::T) where {T <: CVType}
  all(map(field->isequal(getfield(x,field),getfield(y,field)),fieldnames(T)))
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
    vcat(DataFrames.DataFrame.(v))
end

CVStats(T::Type) = CVStats{T}(zeros(T,12)...)

function CVStats{T}(d::CVData{T})

  s = CVStats(T)

  # we kept the entire vectors of y/yhats so we can calculate r2s correctly
  s.ins_r2_nocounts = r2(d.ins_ys,d.ins_yhats_nocounts)
  s.ins_r2 = r2(d.ins_ys,d.ins_yhats)
  s.oos_r2_nocounts = r2(d.oos_ys,d.oos_yhats_nocounts)
  s.oos_r2 = r2(d.oos_ys,d.oos_yhats)

  s.ins_rmse_nocounts = rmse(d.ins_ys,d.ins_yhats_nocounts)
  s.ins_rmse = rmse(d.ins_ys,d.ins_yhats)
  s.oos_rmse_nocounts = rmse(d.oos_ys,d.oos_yhats_nocounts)
  s.oos_rmse = rmse(d.oos_ys,d.oos_yhats)

  # pct changes
  s.oos_pct_change_rmse = 100*(s.oos_rmse/s.oos_rmse_nocounts - 1)
  s.ins_pct_change_rmse = 100*(s.ins_rmse/s.ins_rmse_nocounts - 1)
  s.oos_pct_change_r2 = 100*(s.oos_r2/s.oos_r2_nocounts - 1)
  s.ins_pct_change_r2 = 100*(s.ins_r2/s.ins_r2_nocounts - 1)

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

function cross_validate_dmr_srproj(covars,counts,projdir; k=10, gentype=Kfold, γ=0.0, seed=13, showwarnings=false)
  # dims
  n,p = size(covars)
  # ixnotdir = 1:p .!= projdir

  # init
  gen, cvd = initcv(seed,gentype,n,k)

  # run cv
  for (i, ixtrain) in enumerate(gen)
      ixtest = setdiff(1:n, ixtrain)

      # estimate dmr in train subsample
      coefs = dmr(covars[ixtrain,:],counts[ixtrain,:]; γ=γ, showwarnings=showwarnings)

      # target variable
      ins_y = covars[ixtrain,projdir]

      # get train sample design matrices for regressions
      X, X_nocounts = srprojX(coefs,counts[ixtrain,:],covars[ixtrain,:],projdir)

      # benchmark model w/o text
      insamplelm_nocounts = lm(X_nocounts,ins_y)
      ins_yhat_nocounts = predict(insamplelm_nocounts,X_nocounts)

      # model with text
      insamplelm = lm(X,ins_y)
      ins_yhat = predict(insamplelm,X)

      # evaluate out-of-sample in test subsample
      # target variable
      oos_y = covars[ixtest,projdir]

      # get test sample design matrices
      newX, newX_nocounts = srprojX(coefs,counts[ixtest,:],covars[ixtest,:],projdir)

      # benchmark model w/o text
      oos_yhat_nocounts = predict(insamplelm_nocounts,newX_nocounts)

      # dmr model w/ text
      oos_yhat = predict(insamplelm,newX)

      # save results
      append!(cvd, CVDataRow(ins_y,oos_y,ins_yhat,oos_yhat,ins_yhat_nocounts,oos_yhat_nocounts))

      info("estimated fold $i/$k")
  end

  info("calculated aggreagate fit for $(length(cvd.ins_ys)) in-sample and $(length(cvd.oos_ys)) out-of-sample total observations (with duplication).")

  CVStats(cvd)
end

function cross_validate_hdmr_srproj(covars,counts,projdir; inpos=1:size(covars,2), inzero=1:size(covars,2),
                        k=10, gentype=Kfold, γ=0.0, seed=13, showwarnings=false)
  # dims
  n,p = size(covars)

  # init
  gen, cvd = initcv(seed,gentype,n,k)

  # run cv
  for (i, ixtrain) in enumerate(gen)
      ixtest = setdiff(1:n, ixtrain)

      # estimate hdmr in train subsample
      coefspos, coefszero = hdmr(getindex(covars,ixtrain,inzero), getindex(counts,ixtrain,:); covarspos=getindex(covars,ixtrain,inpos), γ=γ, showwarnings=showwarnings)

      # get full sample design matrices for regressions at the same time.
      # this makes sure the same model is used for both train and test,
      # otherwise, we could be dropping zpos from only one of them.
      X, X_nocounts, includezpos = srprojX(coefspos,coefszero,counts,covars,projdir; inpos=inpos, inzero=inzero)

      # train subsample design matrices
      Xtrain_nocounts = getindex(X_nocounts,ixtrain,:)
      Xtrain = getindex(X,ixtrain,:)

      # target variable
      ins_y = getindex(covars,ixtrain,projdir)

      # benchmark model w/o text
      insamplelm_nocounts = lm(Xtrain_nocounts,ins_y)
      ins_yhat_nocounts = predict(insamplelm_nocounts,Xtrain_nocounts)

      # model with text
      insamplelm = lm(Xtrain,ins_y)
      ins_yhat = predict(insamplelm,Xtrain)

      # evaluate out-of-sample in test subsample
      # target variable
      oos_y = getindex(covars,ixtest,projdir)

      # get test sample design matrices
      # newX, newX_nocounts, newincludezpos = srprojX(coefspos,coefszero,getindex(counts,ixtest,:),getindex(covars,ixtest,:),projdir; inpos=inpos, inzero=inzero, includezpos=includezpos)
      Xtest = getindex(X,ixtest,:)
      Xtest_nocounts = getindex(X_nocounts,ixtest,:)

      # benchmark model w/o text
      oos_yhat_nocounts = predict(insamplelm_nocounts,Xtest_nocounts)

      # dmr model w/ text
      oos_yhat = predict(insamplelm,Xtest)

      # save results
      append!(cvd, CVDataRow(ins_y,oos_y,ins_yhat,oos_yhat,ins_yhat_nocounts,oos_yhat_nocounts))

      info("estimated fold $i/$k")
  end

  info("calculated aggreagate fit for $(length(cvd.ins_ys)) in-sample and $(length(cvd.oos_ys)) out-of-sample total observations (with duplication).")

  CVStats(cvd)
end
