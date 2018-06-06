# Multinomial inverse regression

"""
srproj calculates the MNIR Sufficient Reduction projection from text counts on
to the attribute dimensions of interest (covars in mnlm). In particular, for
counts C, with row sums m, and mnlm coefficients φ_j corresponding to attribute
j, z_j = C'φ_j/m is the SR projection in the direction of j.
The MNIR paper explains how V=[v_1 ... v_K],
your original covariates/attributes, are independent of text counts C given SR
projections Z=[z_1 ... z_K].
"""
srproj(m::DMR, counts, dir::Union{Void,Int}=nothing; focusj=indices(counts,2), select=:AICc) =
  srproj(coef(m; select=select), counts, dir; intercept=hasintercept(m), focusj=focusj)

"""
srproj calculates the MNIR Sufficient Reduction projection from text counts on
to the attribute dimensions of interest (covars in mnlm). In particular, for
counts C, with row sums m, and mnlm coefficients φ_j corresponding to attribute
j, z_j = C'φ_j/m is the SR projection in the direction of j.
The MNIR paper explains how V=[v_1 ... v_K],
your original covariates/attributes, are independent of text counts C given SR
projections Z=[z_1 ... z_K].
"""
function srproj(coefs::AbstractMatrix{T}, counts, dir::Union{Void,Int}=nothing; intercept=true, focusj=indices(counts,2)) where T
   ixoffset = intercept ? 1 : 0 # omitting the intercept
   if dir==nothing
     Φ=coefs[ixoffset+1:end,focusj]' # all directions
   else
     Φ=coefs[ixoffset+dir:ixoffset+dir,focusj]' # keep only desired directions
   end
   m = sum(counts,2) # total counts per observation
   z = counts[:,focusj]*Φ ./ (m+(m.==0)) # scale to get frequencies
   [z m] # m is part of the sufficient reduction
end

"""
Like srproj but efficiently interates over a sparse counts matrix, and
only projects in a single direction (dir).
"""
function srproj(coefs::AbstractMatrix{T}, counts::SparseMatrixCSC, dir::Int; intercept=true, focusj=indices(counts,2)) where T
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

  X, X_nocounts
end
srprojX(m::DMR,counts,covars,projdir; kwargs...) = srprojX(coef(m),counts,covars,projdir; kwargs...)
