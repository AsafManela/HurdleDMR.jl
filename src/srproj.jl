# Sufficient reduction projections

"""
srproj calculates the MNIR Sufficient Reduction projection from text counts on
to the attribute dimensions of interest (covars in mnlm). In particular, for
counts C, with row sums m, and mnlm coefficients φ_j corresponding to attribute
j, z_j = C'φ_j/m is the SR projection in the direction of j.
The MNIR paper explains how V=[v_1 ... v_K],
your original covariates/attributes, are independent of text counts C given SR
projections Z=[z_1 ... z_K].
dir == nothing returns projections in all directions.
"""
srproj(m::DMRPaths, counts, dir::Union{Nothing,Int}=nothing; focusj=axes(counts,2), select=defsegselect) =
  srproj(coef(m, select), counts, dir; intercept=hasintercept(m), focusj=focusj)

srproj(m::DMRCoefs, counts, dir::Union{Nothing,Int}=nothing; focusj=axes(counts,2)) =
  srproj(coef(m), counts, dir; intercept=hasintercept(m), focusj=focusj)

"""
srproj calculates the MNIR Sufficient Reduction projection from text counts on
to the attribute dimensions of interest (covars in mnlm). In particular, for
counts C, with row sums m, and mnlm coefficients φ_j corresponding to attribute
j, z_j = C'φ_j/m is the SR projection in the direction of j.
The MNIR paper explains how V=[v_1 ... v_K],
your original covariates/attributes, are independent of text counts C given SR
projections Z=[z_1 ... z_K].
dir == nothing returns projections in all directions.
"""
function srproj(coefs::AbstractMatrix{T}, counts, dir::Union{Nothing,Int}=nothing; intercept=true, focusj=axes(counts,2)) where T
   ixoffset = intercept ? 1 : 0 # omitting the intercept
   if dir==nothing
     Φ=coefs[ixoffset+1:end,focusj]' # all directions
   else
     Φ=coefs[ixoffset+dir:ixoffset+dir,focusj]' # keep only desired directions
   end
   m = sum(counts, dims=2) # total counts per observation
   z = counts[:,focusj]*Φ ./ (m+(m.==0)) # scale to get frequencies
   [z m] # m is part of the sufficient reduction
end

"""
Like srproj but efficiently interates over a sparse counts matrix, and
only projects in a single direction (dir).
"""
function srproj(coefs::AbstractMatrix{T}, counts::SparseMatrixCSC, dir::Int; intercept=true, focusj=axes(counts,2)) where T
   ixoffset = intercept ? 1 : 0 # omitting the intercept
   n,d = size(counts)
   zm = zeros(n,2)
   φ = vec(coefs[ixoffset+dir,:]) # keep only desired directions
   rows = rowvals(counts)
   vals = nonzeros(counts)
   for j = 1:d
      for i in nzrange(counts, j)
         row = rows[i]
         val = vals[i]
         if in(j,focusj)
           zm[row,1] += val*φ[j]  # projection part
         end
         zm[row,2] += val       # m = total count
      end
   end
   for i=1:n
     mi = zm[i,2]
     if mi > 0
       # scale to get frequencies
       zm[i,1] /= mi
     end
   end
   zm # m is part of the sufficient reduction
end

"""
  Builds the design matrix X for predicting covar in direction projdir
  dmr version
  inz=[1] and testrank=false always for dmr, so variables are ignored and only here for convinence
    of unified calling function
"""
function srprojX(coefs::AbstractMatrix{T},counts,covars,projdir; includem=true, srprojargs...) where T
  # dims
  n,p = size(covars)
  # ixnotdir = 1:p .!= projdir
  ixnotdir = setdiff(1:p,[projdir])

  # design matrix w/o counts data
  X_nocounts = [ones(n) getindex(covars,:,ixnotdir)]

  # add srproj of counts data to X
  Z = srproj(coefs,counts,projdir; srprojargs...)
  X = [X_nocounts Z]

  if !includem
    # drop last column with total counts m
    X = X[:,1:end-1]
  end

  inz = [1]

  X, X_nocounts, inz
end
srprojX(m::DMRCoefs,counts,covars,projdir; inz=[1], testrank=false,
  kwargs...) = srprojX(coef(m),counts,covars,projdir; kwargs...)
srprojX(m::DMRPaths,counts,covars,projdir; select=defsegselect, inz=[1], testrank=false,
  kwargs...) = srprojX(coef(m, select),counts,covars,projdir; kwargs...)

"""
srproj for hurdle dmr takes two coefficent matrices
coefspos, coefszero, and a two specific directions
and returns an n-by-4 matrix Z = [zpos zzero m l].
dirpos = 0 omits positive counts projections and
dirzero = 0 omits zero counts projections.
Setting any of these to nothing will return projections in all directions.
"""
srproj(m::HDMRCoefs, counts, dirpos::D=0, dirzero::D=0; includel=includelinX(m), kwargs...) where {D<:Int}=
  srproj(coef(m)..., counts, dirpos, dirzero; intercept=hasintercept(m), includel=includel, kwargs...)
srproj(m::HDMRPaths, counts, dirpos::D=0, dirzero::D=0; includel=includelinX(m), select=defsegselect, kwargs...) where {D<:Int}=
  srproj(coef(m, select)..., counts, dirpos, dirzero; intercept=hasintercept(m), includel=includel, kwargs...)

"""
srproj for hurdle dmr takes two coefficent matrices
coefspos, coefszero, and a two specific directions
and returns an n-by-4 matrix Z = [zpos zzero m l].
dirpos = 0 omits positive counts projections and
dirzero = 0 omits zero counts projections.
Setting any of these to nothing will return projections in all directions.
"""
function srproj(coefspos::C, coefszero::C, counts, dirpos::D, dirzero::D;
  includem=true,        # whether to include total counts in Z
  includel=false,      # whether to include total 1(counts) in Z
  kwargs...) where {T, C<:AbstractMatrix{T}, D<:Int}

  if dirpos>0 && dirzero>0
    zpos = srproj(coefspos, counts, dirpos; kwargs...)
    zzero = srproj(coefszero, posindic(counts), dirzero; kwargs...)
    # second element should be same m in both, but because zero model
    # only sums indicators it generates smaller totals, so use the one
    # from the pos model
    # TODO: this needs to be fleshed out better in the theory to guide this choice
    # revised paper says to use l too
    Z = [zpos[:,1] zzero[:,1] zpos[:,2] zzero[:,2]]
    Z[:, [true, true, includem, includel]]
  elseif dirpos>0
    Z = srproj(coefspos, counts, dirpos; kwargs...)
    Z[:, [true, includem]]
  elseif dirzero>0
    Z = srproj(coefszero, posindic(counts), dirzero; kwargs...)
    Z[:, [true, includel]]
  else
    error("No direction to project to (dirpos=$dirpos,dirzero=$dirzero)")
  end
end

function ixcovars(dir::Int, inpos, inzero)
  # @assert dir ∈ inzero "projection direction $dir must be included in coefzero estimation!"
  # @assert dir ∈ inpos "projection direction $dir must be included in coefpos estimation!"

  # findfirst returns 0 if not found
  dirpos = something(findfirst(isequal(dir), inpos), 0)
  dirzero = something(findfirst(isequal(dir), inzero), 0)

  ineither = union(inzero,inpos)
  ixnotdir = setdiff(ineither,[dir])

  dirpos,dirzero,ineither,ixnotdir
end

"""
  Builds the design matrix X for predicting covar in direction projdir
  hdmr version
  Assumes that covars include all variables for both positives and zeros models
  and indicates which variables are where with the index arrays inpos and inzero.
  inz=[1,2] if both zpos and zzero are included
  inz=[2] if zpos is dropped due to collinearity
"""
function srprojX(coefspos::M, coefszero::M, counts, covars, projdir::Int;
  inpos=1:size(covars,2), inzero=1:size(covars,2),
  includem=true,        # whether to include total counts in Z
  includel=false,      # whether to include total 1(counts) in Z
  inz=[1,2], testrank=true, srprojargs...) where {T, M<:AbstractMatrix{T}}

  # dims
  n,p = size(covars)

  # get pos and zero subset indices
  dirpos,dirzero,ineither,ixnotdir = ixcovars(projdir, inpos, inzero)

  # design matrix w/o counts data
  X_nocounts = [ones(n) getindex(covars,:,ixnotdir)]

  includezpos = 1 ∈ inz
  # add srproj of counts data to X
  if includezpos
    Z = srproj(coefspos, coefszero, counts, dirpos, dirzero;
      includem=includem, includel=includel, srprojargs...)
    X = [X_nocounts Z]
  end

  if !includezpos || (testrank && rank(X) < size(X,2))
    if includezpos
      @info("rank(X) = $(rank(X)) < $(size(X,2)) = size(X,2). dropping zpos.")
    else
      @info("includezpos == false. dropping zpos.")
    end
    Z = srproj(coefszero, posindic(counts), dirzero; srprojargs...)
    X = [X_nocounts Z]
    inz = [2]
  end

  X, X_nocounts, inz
end

"Set the default value for whether to include obs-specific lexicon (l) in the srproj"
includelinX(m::HDMRCoefs{<:Hurdle}) = false
includelinX(m::HDMRPaths{<:Union{Missing, Hurdle}}) = false
includelinX(m::HDMRPaths{<:Hurdle}) = false

includelinX(m::HDMRCoefs{<:InclusionRepetition}) = true
includelinX(m::HDMRPaths{<:InclusionRepetition}) = true
includelinX(m::HDMRPaths{<:Union{Missing, InclusionRepetition}}) = true
includelinX(m::HDMRPaths{Missing}) = true

srprojX(m::HDMRCoefs,counts,covars,projdir; includel=includelinX(m), kwargs...) = srprojX(coef(m)...,counts,covars,projdir; inpos=m.inpos, inzero=m.inzero, includel=includel, kwargs...)
srprojX(m::HDMRPaths,counts,covars,projdir; includel=includelinX(m), select=defsegselect, kwargs...) = srprojX(coef(m, select)...,counts,covars,projdir; inpos=m.inpos, inzero=m.inzero, includel=includel, kwargs...)
