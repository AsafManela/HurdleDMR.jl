module Rdistrom
##### Distributed Logistic Multinomial Regression  ######
# A Julia wrapper for dmr.R by Matt Taddy

export dmr, gamlr, dmrplots

# Pkg.add("RCall")
using RCall, RDatasets

R"if(!require(distrom)){install.packages(\"distrom\"); library(distrom)}"
R"if(!require(textir)){install.packages(\"textir\"); library(textir)}"
R"if(!require(Matrix)){install.packages(\"Matrix\"); library(Matrix)}"

@rimport distrom
@rimport textir
@rimport Matrix as RMatrix

# some converters for sparse matrices
sparseRMatrix(I::Vector,J::Vector,V::Vector) = RMatrix.sparseMatrix(I,J,x=V)
sparseRMatrix(m::SparseMatrixCSC) = sparseRMatrix(findnz(m)...)
function sparseJMatrix(m::RCall.RObject{RCall.S4Sxp},T=Float64)
     sparseclass = rcopy(RCall.getclass(m))

     if sparseclass == "dgCMatrix"
          m = R"as ($m, \"dgTMatrix\")"
     else
          @assert sparseclass == "dgTMatrix" "sparseJMatrix called on unknown matrix format"
     end

     I = rcopy(R"$m@i")+1
     J = rcopy(R"$m@j")+1
     V = rcopy(Vector{T},R"$m@x")
     rows,cols=rcopy(R"$m@Dim")

     sparse(I,J,V,rows,cols)
end

type gamlr
     attributes::Dict{UTF8String,Any}
     Robj::RCall.RObject{RCall.VecSxp}

     function gamlr(fit::RCall.RObject{RCall.VecSxp})
          @assert rcopy(RCall.getclass(fit)) == "gamlr" "RObject is not gamlr"
          attributes = Dict{UTF8String,Any}()
          attribute_names = rcopy(getnames(fit))
          for i=1:length(attribute_names)
               attribute_class = rcopy(RCall.getclass(fit[i]))
               if attribute_class == "dgCMatrix" || attribute_class == "dgTMatrix"
                    attributes[attribute_names[i]] = sparseJMatrix(fit[i])
               else
                    attributes[attribute_names[i]] = rcopy(fit[i])
               end
          end
          new(attributes,fit)
     end
end

# covars: nxp matrix
# counts: nxd matrix
type dmr
     gamlrs::Vector{gamlr}
     n::Int
     d::Int
     p::Int
     names
     nlambda::Int
     mu
     RObj::RCall.RObject{RCall.VecSxp}

     function dmr(fits::RCall.RObject{RCall.VecSxp},n::Int,p::Int)
          @assert rcopy(RCall.getclass(fits)) == "dmr" "RObject is not dmr"
          d = length(fits)
          gamlrs = [gamlr(fits[i]) for i=1:d]
          names = rcopy(getnames(fits))
          nlambda = rcopy(getattrib(fits,"nlambda"))
          mu = rcopy(getattrib(fits,"mu"))
          new(gamlrs,n,d,p,names,nlambda,mu,fits)
     end
end

# length(fits::dmr) = fits.nfits

function dmr(covars, counts; nlocal_workers=Sys.CPU_CORES-2, gamma=1.0, kwargs...)
     # get dimensions
     n = size(counts,1)
     n1,p = size(covars)
     p += 1 # intercept included
     @assert n==n1 "counts and covars should have the same number of observations"

     ## make a parallel cluster
     cl = R"makeCluster($nlocal_workers,type=\"FORK\")"

     # convert sparse matrices to R's format if needed
     if typeof(covars) == SparseMatrixCSC
          covars = sparseRMatrix(covars)
     end
     if typeof(counts) == SparseMatrixCSC
          counts = sparseRMatrix(counts)
     end

     info("starting Rdistrom.dmr with kwargs=$kwargs...")
     ## fit in parallel
     fits = distrom.dmr(cl, covars, counts; gamma=gamma, kwargs...)

     ## its good practice stop the cluster once you're done
     R"stopCluster($cl)"

     dmr(fits,n,p)
end

function dmrplots(fits::dmr,fitnames=[]; kwargs...)
     if fitnames==[] && fits.names != []
          fitnames = fits.names
     end
     dmrplots(fits.gamlrs,fitnames; kwargs...)
end

function dmrplots(gamlrs::Vector{gamlr},fitnames=[];cols=2)
     nfits = length(gamlrs)
     rows = ceil(nfits/cols)
     R"par(mfrow=c($rows,$cols))"
     for j=1:nfits
          R"plot($(gamlrs[j].Robj))"
          if fitnames!=[]
               R"mtext($(fitnames[j]),font=2,line=2)"
          end
     end
end

function srproj(fits::dmr, counts, dir=1:fits.p-1)
     if typeof(counts) == SparseMatrixCSC
          counts = sparseRMatrix(counts)
     end
     rcopy(R"srproj($(fits.RObj),$counts,dir=$dir)")
end

function srproj2(fits::dmr, counts, dir=1:fits.p-1)
     Φ=coef(fits)[2:end,:] # omitting the intercept
     Φ=Φ[dir,:] # keep only desired directions
     m = sum(counts,2) # total counts per observation
     z = counts*Φ' ./ (m+(m.==0)) # scale to get frequencies
     [z m] # m is part of the sufficient reduction
end

import StatsBase.coef
function coef(fit::gamlr, select=:ALL, k=2)
     if select == :ALL
          [sparse(fit.attributes["alpha"])'; fit.attributes["beta"]]
     elseif select == :AICc
          ix = rcopy(R"which.min(AICc($(fit.Robj),k=$k))")
          [sparse(fit.attributes["alpha"][ix:ix]); fit.attributes["beta"][:,ix]]
     elseif select == :CV
          error("CV wrapper not yet implemented")
     end
end

function allcoef(fits::dmr, k=2)
     coefs = zeros(fits.nlambda,fits.p,fits.d)
     for j=1:fits.d
          fit = fits.gamlrs[j]
          c = coef(fit,:ALL,k)
          for i=1:fits.p
               for s=1:fits.nlambda
                    coefs[s,i,j] = c[i,s]
               end
          end
     end
     # coefs = coef(fits.gamlrs[1],:ALL,k)
     #
     # hcat(map(fit->coef(fit,:ALL,k),fits.gamlrs)...)
     coefs
end

function aicc(fits::dmr, k=2)
     aiccs = zeros(fits.nlambda,fits.d)
     for j=1:fits.d
          fit = fits.gamlrs[j]
          aiccs[:,j] = rcopy(R"AICc($(fit.Robj),k=$k)")
     end
     aiccs
end

function coef(fits::dmr, select=:AICc, k=2)
     I = Vector{Int64}()
     J = Vector{Int64}()
     V = Vector{Float64}()

     for j = 1:fits.d
          coef!(I, J, V, j, fits.gamlrs[j], select, k)
     end

     # show([I J V])
     # show((fits.p,fits.d))
     sparse(I,J,V,fits.p,fits.d)
end

function coef!(I::Vector{Int64}, J::Vector{Int64}, V::Vector{Float64}, j::Int64, fit::gamlr, select=:AICc, k=2)
     if select == :AICc
          # get index for min AICc estimates
          ix = rcopy(R"which.min(AICc($(fit.Robj),k=$k))")

          # intecept
          push!(I,1)
          push!(J,j)
          push!(V,fit.attributes["alpha"][ix])

          # other coefs
          i = 2
          for b = fit.attributes["beta"][:,ix]
               push!(I,i)
               push!(J,j)
               push!(V,b)
               i += 1
          end

     elseif select == :CV
          error("CV wrapper not yet implemented")
     end
end


end # module
