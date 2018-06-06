# Hurdle inverse regression

"""
srproj for hurdle dmr takes two coefficent matrices
coefspos, coefszero, and a two specific directions
and returns an n-by-3 matrix Z = [zpos zzero m].
dirpos = 0 omits positive counts projections and
dirzero = 0 omits zero counts projections.
"""
srproj(m::HDMR, counts, dirpos::D=nothing, dirzero::D=nothing; select=:AICc, kwargs...) where {D<:Union{Void,Int}}=
  srproj(coef(m; select=select)..., counts, dir; intercept=hasintercept(m), kwargs...)

"""
srproj for hurdle dmr takes two coefficent matrices
coefspos, coefszero, and a two specific directions
and returns an n-by-3 matrix Z = [zpos zzero m].
dirpos = 0 omits positive counts projections and
dirzero = 0 omits zero counts projections.
"""
function srproj(coefspos::C, coefszero::C, counts, dirpos::D, dirzero::D; kwargs...) where {T, C<:AbstractMatrix{T}, D<:Union{Void,Int}}
  if (dirpos == nothing || dirpos>0) && (dirzero == nothing || dirzero>0)
    zpos = srproj(coefspos, counts, dirpos; kwargs...)
    zzero = srproj(coefszero, posindic(counts), dirzero; kwargs...)
    # second element should be same m in both, but because zero model
    # only sums indicators it generates smaller totals, so use the one
    # from the pos model
    # TODO: this needs to be fleshed out better in the theory to guide this choice
    [zpos[:,1] zzero[:,1] zpos[:,2]]
  elseif (dirpos == nothing || dirpos>0)
    srproj(coefspos, counts, dirpos; kwargs...)
  elseif (dirzero == nothing || dirzero>0)
    srproj(coefszero, posindic(counts), dirzero; kwargs...)
  else
    error("No direction to project to (dirpos=$dirpos,dirzero=$dirzero)")
  end
end

function ixcovars(p::Int, dir::Int, inpos, inzero)
  # @assert dir ∈ inzero "projection direction $dir must be included in coefzero estimation!"
  # @assert dir ∈ inpos "projection direction $dir must be included in coefpos estimation!"

  # findfirst returns 0 if not found
  dirpos = findfirst(inpos,dir)
  dirzero = findfirst(inzero,dir)

  ineither = union(inzero,inpos)
  ixnotdir = setdiff(ineither,[dir])

  dirpos,dirzero,ineither,ixnotdir
end

"""
  Builds the design matrix X for predicting covar in direction projdir
  hdmr version
"""
function srprojX(coefspos, coefszero, counts, covars, dir::Int; inpos=1:size(covars,2), inzero=1:size(covars,2), includem=true, includezpos=true, testrank=true, srprojargs...)
  # dims
  n,p = size(covars)

  # get pos and zero subset indices
  dirpos,dirzero,ineither,ixnotdir = ixcovars(p, dir, inpos, inzero)

  # design matrix w/o counts data
  X_nocounts = [ones(n) getindex(covars,:,ixnotdir)]

  # add srproj of counts data to X
  if includezpos
    Z = srproj(coefspos, coefszero, counts, dirpos, dirzero; srprojargs...)
    X = [X_nocounts Z]
  end

  if !includezpos || (testrank && rank(X) < size(X,2))
    if includezpos
      info("rank(X) = $(rank(X)) < $(size(X,2)) = size(X,2). dropping zpos.")
    else
      info("includezpos == false. dropping zpos.")
    end
    Z = srproj(coefszero, posindic(counts), dirzero; srprojargs...)
    X = [X_nocounts Z]
    includezpos = false
  end

  if !includem
    # drop last column with total counts m
    X = X[:,1:end-1]
  end

  X, X_nocounts, includezpos
end
