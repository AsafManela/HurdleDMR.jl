var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#HurdleDMR.jl-1",
    "page": "Home",
    "title": "HurdleDMR.jl",
    "category": "section",
    "text": "HurdleDMR.jl is a Julia implementation of the Hurdle Distributed Multiple Regression (HDMR), as described in:Kelly, Bryan, Asaf Manela, and Alan Moreira (2018). Text Selection. Working paper.It includes a Julia implementation of the Distributed Multinomial Regression (DMR) model of Taddy (2015)."
},

{
    "location": "#Setup-1",
    "page": "Home",
    "title": "Setup",
    "category": "section",
    "text": "Install the HurdleDMR packagepkg> add HurdleDMRAdd parallel workers and make package available to workersaddprocs(Sys.CPU_CORES-2)\nimport HurdleDMR; @everywhere using HurdleDMRSetup your data into an n-by-p covars matrix, and a (sparse) n-by-d counts matrix. Here we generate some random data.using CSV, GLM, DataFrames, Distributions\nn = 100\np = 3\nd = 4\n\nsrand(13)\nm = 1+rand(Poisson(5),n)\ncovars = rand(n,p)\nηfn(vi) = exp.([0 + i*sum(vi) for i=1:d])\nq = [ηfn(covars[i,:]) for i=1:n]\nscale!.(q,ones(n)./sum.(q))\ncounts = convert(SparseMatrixCSC{Float64,Int},hcat(broadcast((qi,mi)->rand(Multinomial(mi, qi)),q,m)...)\')\ncovarsdf = DataFrame(covars,[:vy, :v1, :v2])"
},

{
    "location": "#HurdleDMR.DCR",
    "page": "Home",
    "title": "HurdleDMR.DCR",
    "category": "type",
    "text": "Abstract Distributed Counts Regression (DCR) returned object\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.DMR",
    "page": "Home",
    "title": "HurdleDMR.DMR",
    "category": "type",
    "text": "Abstract DMR returned object\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.DMRCoefs",
    "page": "Home",
    "title": "HurdleDMR.DMRCoefs",
    "category": "type",
    "text": "Relatively light object used to return DMR results when we only care about estimated coefficients.\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.DMRPaths",
    "page": "Home",
    "title": "HurdleDMR.DMRPaths",
    "category": "type",
    "text": "Relatively heavy object used to return DMR results when we care about the regulatrization paths.\n\n\n\n\n\n"
},

{
    "location": "#Distributions.ncategories-Tuple{DCR}",
    "page": "Home",
    "title": "Distributions.ncategories",
    "category": "method",
    "text": "Number of categories (terms/words/phrases) used for DMR estimation\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.dmr-Union{Tuple{V}, Tuple{T}, Tuple{AbstractArray{T,2},AbstractArray{V,2}}} where V where T<:AbstractFloat",
    "page": "Home",
    "title": "HurdleDMR.dmr",
    "category": "method",
    "text": "Shorthand for fit(DMR,covars,counts). See also fit(::DMR)\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.dmrpaths-Union{Tuple{T}, Tuple{AbstractArray{T,2},AbstractArray{T,2} where T}} where T<:AbstractFloat",
    "page": "Home",
    "title": "HurdleDMR.dmrpaths",
    "category": "method",
    "text": "Shorthand for fit(DMRPaths,covars,counts). See also fit(::DMRPaths)\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.hasintercept-Tuple{DCR}",
    "page": "Home",
    "title": "HurdleDMR.hasintercept",
    "category": "method",
    "text": "Whether the model includes an intercept in each independent counts (e.g. hurdle) regression\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.ncoefs-Tuple{DMR}",
    "page": "Home",
    "title": "HurdleDMR.ncoefs",
    "category": "method",
    "text": "Number of coefficient potentially including intercept used for each independent Poisson regression\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.ncovars-Tuple{DMR}",
    "page": "Home",
    "title": "HurdleDMR.ncovars",
    "category": "method",
    "text": "Number of covariates used for DMR estimation\n\n\n\n\n\n"
},

{
    "location": "#StatsBase.coef-Tuple{DMRCoefs}",
    "page": "Home",
    "title": "StatsBase.coef",
    "category": "method",
    "text": "coef(m::DMRCoefs)\n\nReturns the AICc optimal coefficients matrix fitted with DMR.\n\nExample:\n\n  m = fit(DMR,covars,counts)\n  coef(m)\n\nKeywords\n\nselect=:AICc only supports AICc criterion. To get other segments see coef(::DMRPaths).\n\n\n\n\n\n"
},

{
    "location": "#StatsBase.coef-Tuple{DMRPaths}",
    "page": "Home",
    "title": "StatsBase.coef",
    "category": "method",
    "text": "coef(m::DMRPaths; select=:AICc)\n\nReturns all or selected coefficients matrix fitted with DMR.\n\nExample:\n\n  m = fit(DMRPaths,covars,counts)\n  coef(m; select=:CVmin)\n\nKeywords\n\nselect=:AICc See coef(::RegularizationPath).\n\n\n\n\n\n"
},

{
    "location": "#StatsBase.fit-Union{Tuple{D}, Tuple{T}, Tuple{Type{D},AbstractArray{T,2},AbstractArray{T,2} where T}} where D<:DMR where T<:AbstractFloat",
    "page": "Home",
    "title": "StatsBase.fit",
    "category": "method",
    "text": "fit(DMR,covars,counts; <keyword arguments>)\ndmr(covars,counts; <keyword arguments>)\n\nFit a Distributed Multinomial Regression (DMR) of counts on covars.\n\nDMR fits independent poisson gamma lasso regressions to each column of counts to approximate a multinomial, picks the minimum AICc segement of each path, and returns a coefficient matrix (wrapped in DMRCoefs) representing point estimates for the entire multinomial (includes the intercept if one was included).\n\nExample:\n\n  m = fit(DMR,covars,counts)\n\nArguments\n\ncovars n-by-p matrix of covariates\ncounts n-by-d matrix of counts (usually sparse)\n\nKeywords\n\nintercept::Bool=false include an intercept in each poisson\nparallel::Bool=true parallelize the poisson fits\nlocal_cluster::Bool=true use local_cluster mode that shares memory across   parallel workers that is appropriate on a single multicore machine, or   remote cluster mode that is more appropriate when distributing across machines   for which sharing memory is costly.\nverbose::Bool=true\nshowwarnings::Bool=false\nkwargs... additional keyword arguments passed along to fit(GammaLassoPath,...)\n\n\n\n\n\n"
},

{
    "location": "#StatsBase.fit-Union{Tuple{T}, Tuple{Type{DMRPaths},AbstractArray{T,2},AbstractArray{T,2} where T}} where T<:AbstractFloat",
    "page": "Home",
    "title": "StatsBase.fit",
    "category": "method",
    "text": "fit(DMRPaths,covars,counts; <keyword arguments>)\ndmrpaths(covars,counts; <keyword arguments>)\n\nFit a Distributed Multinomial Regression (DMR) of counts on covars, and returns the entire regulatrization paths, which may be useful for plotting or picking coefficients other than the AICc optimal ones. Same arguments as fit(::DMR).\n\n\n\n\n\n"
},

{
    "location": "#StatsBase.fit-Union{Tuple{T}, Tuple{Type{T},Model,AbstractDataFrame,AbstractArray{T,2} where T}} where T<:DMR",
    "page": "Home",
    "title": "StatsBase.fit",
    "category": "method",
    "text": "fit(DMR,@model(c ~ x1*x2),df,counts; <keyword arguments>)\n\nFits a DMR but takes a model formula and dataframe instead of the covars matrix. See also fit(::DMR).\n\nc must be specified on the lhs to indicate the model for counts.\n\n\n\n\n\n"
},

{
    "location": "#StatsBase.nobs-Tuple{DCR}",
    "page": "Home",
    "title": "StatsBase.nobs",
    "category": "method",
    "text": "Number of observations used. May be lower than provided after removing all zero obs.\n\n\n\n\n\n"
},

{
    "location": "#StatsBase.predict-Union{Tuple{T}, Tuple{DMRPaths,AbstractArray{T,2}}} where T<:AbstractFloat",
    "page": "Home",
    "title": "StatsBase.predict",
    "category": "method",
    "text": "predict(m,newcovars; <keyword arguments>)\n\nPredict counts using a fitted DMRPaths object and given newcovars.\n\nExample:\n\n  m = fit(DMRPaths,covars,counts)\n  newcovars = covars[1:10,:]\n  countshat = predict(m, newcovars; select=:AICc)\n\nArguments\n\nm::DMRPaths fitted DMRPaths model (DMRCoefs currently not supported)\nnewcovars n-by-p matrix of covariates of same dimensions used to fit m.\n\nKeywords\n\nselect=:AICc See coef(::RegularizationPath).\nkwargs... additional keyword arguments passed along to predict() for each category j=1..size(counts,2)\n\n\n\n\n\n"
},

{
    "location": "#Distributed-Multinomial-Regression-(DMR)-1",
    "page": "Home",
    "title": "Distributed Multinomial Regression (DMR)",
    "category": "section",
    "text": "The Distributed Multinomial Regression (DMR) model of Taddy (2015) is a highly scalable approximation to the Multinomial using distributed (independent, parallel) Poisson regressions, one for each of the d categories (columns) of a large counts matrix, on the covars.To fit a DMR:m = dmr(covars, counts)or with a dataframe and formulamf = @model(c ~ vy + v1 + v2)\nm = fit(DMR, mf, covarsdf, counts)in either case we can get the coefficients matrix for each variable + intercept as usual withcoef(m)By default we only return the AICc maximizing coefficients. To also get back the entire regulatrization paths, runpaths = fit(DMRPaths, mf, covarsdf, counts)we can now select, for example the coefficients that minimize CV mse (takes a while)coef(paths; select=:CVmin)Modules = [HurdleDMR]\nOrder   = [:macro, :type, :function]\nPages   = [\"src/dmr.jl\"]\nPrivate = false"
},

{
    "location": "#HurdleDMR.HDMR",
    "page": "Home",
    "title": "HurdleDMR.HDMR",
    "category": "type",
    "text": "Abstract HDMR returned object\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.HDMRCoefs",
    "page": "Home",
    "title": "HurdleDMR.HDMRCoefs",
    "category": "type",
    "text": "Relatively light object used to return HDMR results when we only care about estimated coefficients.\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.HDMRPaths",
    "page": "Home",
    "title": "HurdleDMR.HDMRPaths",
    "category": "type",
    "text": "Relatively heavy object used to return HDMR results when we care about the regulatrization paths.\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.hdmr-Union{Tuple{V}, Tuple{T}, Tuple{AbstractArray{T,2},AbstractArray{V,2}}} where V where T<:AbstractFloat",
    "page": "Home",
    "title": "HurdleDMR.hdmr",
    "category": "method",
    "text": "Shorthand for fit(HDMR,covars,counts). See also fit(::HDMR)\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.hdmrpaths-Union{Tuple{T}, Tuple{AbstractArray{T,2},AbstractArray{T,2} where T}} where T<:AbstractFloat",
    "page": "Home",
    "title": "HurdleDMR.hdmrpaths",
    "category": "method",
    "text": "Shorthand for fit(HDMRPaths,covars,counts). See also fit(::HDMRPaths)\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.ncoefspos-Tuple{HDMR}",
    "page": "Home",
    "title": "HurdleDMR.ncoefspos",
    "category": "method",
    "text": "Number of coefficient potentially including intercept used by model for positives\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.ncoefszero-Tuple{HDMR}",
    "page": "Home",
    "title": "HurdleDMR.ncoefszero",
    "category": "method",
    "text": "Number of coefficient potentially including intercept used by model for zeros\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.ncovarspos-Tuple{HDMR}",
    "page": "Home",
    "title": "HurdleDMR.ncovarspos",
    "category": "method",
    "text": "Number of covariates used for HDMR estimation of positives model\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.ncovarszero-Tuple{HDMR}",
    "page": "Home",
    "title": "HurdleDMR.ncovarszero",
    "category": "method",
    "text": "Number of covariates used for HDMR estimation of zeros model\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.posindic-Tuple{SparseArrays.SparseMatrixCSC}",
    "page": "Home",
    "title": "HurdleDMR.posindic",
    "category": "method",
    "text": "Sparse version simply replaces all the non-zero values with ones.\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.posindic-Union{Tuple{AbstractArray{T,N} where N}, Tuple{T}} where T",
    "page": "Home",
    "title": "HurdleDMR.posindic",
    "category": "method",
    "text": "posindic(A)\n\nReturns an array of the same dimensions of indicators for positive entries in A.\n\n\n\n\n\n"
},

{
    "location": "#StatsBase.coef-Tuple{HDMRCoefs}",
    "page": "Home",
    "title": "StatsBase.coef",
    "category": "method",
    "text": "coef(m::HDMRCoefs)\n\nReturns the AICc optimal coefficient matrices fitted with HDMR.\n\nExample:\n\n  m = fit(HDMR,covars,counts)\n  coefspos, coefszero = coef(m)\n\nKeywords\n\nselect=:AICc only supports AICc criterion. To get other segments see coef(::HDMRPaths).\n\n\n\n\n\n"
},

{
    "location": "#StatsBase.coef-Tuple{HDMRPaths}",
    "page": "Home",
    "title": "StatsBase.coef",
    "category": "method",
    "text": "coef(m::HDMRPaths; select=:all)\n\nReturns all or selected coefficient matrices fitted with HDMR.\n\nExample:\n\n  m = fit(HDMRPaths,covars,counts)\n  coefspos, coefszero = coef(m; select=:CVmin)\n\nKeywords\n\nselect=:AICc See coef(::RegularizationPath).\n\n\n\n\n\n"
},

{
    "location": "#StatsBase.fit-Union{Tuple{H}, Tuple{T}, Tuple{Type{H},AbstractArray{T,2},AbstractArray{T,2} where T}} where H<:HDMR where T<:AbstractFloat",
    "page": "Home",
    "title": "StatsBase.fit",
    "category": "method",
    "text": "fit(HDMR,covars,counts; <keyword arguments>)\nhdmr(covars,counts; <keyword arguments>)\n\nFit a Hurdle Distributed Multiple Regression (HDMR) of counts on covars.\n\nHDMR fits independent hurdle lasso regressions to each column of counts to approximate a multinomial, picks the minimum AICc segement of each path, and returns a coefficient matrix (wrapped in HDMRCoefs) representing point estimates for the entire multinomial (includes the intercept if one was included).\n\nExample:\n\n  m = fit(HDMR,covars,counts)\n\nArguments\n\ncovars n-by-p matrix of covariates\ncounts n-by-d matrix of counts (usually sparse)\n\nKeywords\n\ninpos=1:p indices of covars columns included in model for positive counts\ninzero=1:p indices of covars columns included in model for zero counts\nintercept::Bool=false include a intercepts in each hurdle regression\nparallel::Bool=true parallelize the poisson fits\nlocal_cluster::Bool=true use local_cluster mode that shares memory across   parallel workers that is appropriate on a single multicore machine, or   remote cluster mode that is more appropriate when distributing across machines   for which sharing memory is costly.\nverbose::Bool=true\nshowwarnings::Bool=false\nkwargs... additional keyword arguments passed along to fit(Hurdle,...)\n\n\n\n\n\n"
},

{
    "location": "#StatsBase.fit-Union{Tuple{T}, Tuple{Type{HDMRPaths},AbstractArray{T,2},AbstractArray{T,2} where T}} where T<:AbstractFloat",
    "page": "Home",
    "title": "StatsBase.fit",
    "category": "method",
    "text": "fit(HDMRPaths,covars,counts; <keyword arguments>)\nhdmrpaths(covars,counts; <keyword arguments>)\n\nFit a Hurdle Distributed Multiple Regression (HDMR) of counts on covars, and returns the entire regulatrization paths, which may be useful for plotting or picking coefficients other than the AICc optimal ones. Same arguments as fit(::HDMR).\n\n\n\n\n\n"
},

{
    "location": "#StatsBase.fit-Union{Tuple{T}, Tuple{Type{T},Model,AbstractDataFrame,AbstractArray{T,2} where T,Vararg{Any,N} where N}} where T<:HDMR",
    "page": "Home",
    "title": "StatsBase.fit",
    "category": "method",
    "text": "fit(HDMR,@model(h ~ x1 + x2, c ~ x1),df,counts; <keyword arguments>)\n\nFits a HDMR but takes a model formula and dataframe instead of the covars matrix. See also fit(::HDMR).\n\nh and c on the lhs indicate the model for zeros and positives, respectively.\n\n\n\n\n\n"
},

{
    "location": "#StatsBase.predict-Union{Tuple{T}, Tuple{HDMRPaths,AbstractArray{T,2}}} where T<:AbstractFloat",
    "page": "Home",
    "title": "StatsBase.predict",
    "category": "method",
    "text": "predict(m,newcovars; <keyword arguments>)\n\nPredict counts using a fitted HDMRPaths object and given newcovars.\n\nExample:\n\n  m = fit(HDMRPaths,covars,counts)\n  newcovars = covars[1:10,:]\n  countshat = predict(m, newcovars; select=:AICc)\n\nArguments\n\nm::HDMRPaths fitted DMRPaths model (HDMRCoefs currently not supported)\nnewcovars n-by-p matrix of covariates of same dimensions used to fit m.\n\nKeywords\n\nselect=:AICc See coef(::RegularizationPath).\nkwargs... additional keyword arguments passed along to predict() for each category j=1..size(counts,2)\n\n\n\n\n\n"
},

{
    "location": "#Hurdle-Distributed-Multiple-Regression-(HDMR)-1",
    "page": "Home",
    "title": "Hurdle Distributed Multiple Regression (HDMR)",
    "category": "section",
    "text": "For highly sparse counts, as is often the case with text that is selected for various reasons, the Hurdle Distributed Multiple Regression (HDMR) model of Kelly, Manela, and Moreira (2018), may be superior to the DMR. It approximates a higher dispersion Multinomial using distributed (independent, parallel) Hurdle regressions, one for each of the d categories (columns) of a large counts matrix, on the covars. It allows a potentially different sets of covariates to explain category inclusion (h=1c0), and repetition (c0).Both the model for zeroes and for positive counts are regularized by default, using GammaLassoPath, picking the AICc optimal segment of the regularization path.HDMR can be fitted:m = hdmr(covars, counts; inpos=1:2, inzero=1:3)or with a dataframe and formulamf = @model(h ~ vy + v1 + v2, c ~ vy + v1)\nm = fit(HDMR, mf, covarsdf, counts)where the h ~ equation is the model for zeros (hurdle crossing) and c ~ is the model for positive countsin either case we can get the coefficients matrix for each variable + intercept as usual withcoefspos, coefszero = coef(m)By default we only return the AICc maximizing coefficients. To also get back the entire regulatrization paths, runpaths = fit(HDMRPaths, mf, covarsdf, counts)\n\ncoef(paths; select=:all)Syntax:Modules = [HurdleDMR]\nOrder   = [:macro, :type, :function]\nPages   = [\"src/hdmr.jl\"]\nPrivate = false"
},

{
    "location": "#HurdleDMR.srproj",
    "page": "Home",
    "title": "HurdleDMR.srproj",
    "category": "function",
    "text": "srproj calculates the MNIR Sufficient Reduction projection from text counts on to the attribute dimensions of interest (covars in mnlm). In particular, for counts C, with row sums m, and mnlm coefficients φj corresponding to attribute j, zj = C\'φj/m is the SR projection in the direction of j. The MNIR paper explains how V=[v1 ... vK], your original covariates/attributes, are independent of text counts C given SR projections Z=[z1 ... z_K]. dir == nothing returns projections in all directions.\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.srproj-Union{Tuple{D}, Tuple{C}, Tuple{T}, Tuple{C,C,Any,D,D}} where D<:Union{Nothing, Int64} where C<:AbstractArray{T,2} where T",
    "page": "Home",
    "title": "HurdleDMR.srproj",
    "category": "method",
    "text": "srproj for hurdle dmr takes two coefficent matrices coefspos, coefszero, and a two specific directions and returns an n-by-3 matrix Z = [zpos zzero m]. dirpos = 0 omits positive counts projections and dirzero = 0 omits zero counts projections. Setting any of these to nothing will return projections in all directions.\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.srproj-Union{Tuple{D}, Tuple{HDMR,Any}, Tuple{HDMR,Any,D}, Tuple{HDMR,Any,D,D}} where D<:Union{Nothing, Int64}",
    "page": "Home",
    "title": "HurdleDMR.srproj",
    "category": "method",
    "text": "srproj for hurdle dmr takes two coefficent matrices coefspos, coefszero, and a two specific directions and returns an n-by-3 matrix Z = [zpos zzero m]. dirpos = 0 omits positive counts projections and dirzero = 0 omits zero counts projections. Setting any of these to nothing will return projections in all directions.\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.srproj-Union{Tuple{T}, Tuple{AbstractArray{T,2},Any}, Tuple{AbstractArray{T,2},Any,Union{Nothing, Int64}}} where T",
    "page": "Home",
    "title": "HurdleDMR.srproj",
    "category": "method",
    "text": "srproj calculates the MNIR Sufficient Reduction projection from text counts on to the attribute dimensions of interest (covars in mnlm). In particular, for counts C, with row sums m, and mnlm coefficients φj corresponding to attribute j, zj = C\'φj/m is the SR projection in the direction of j. The MNIR paper explains how V=[v1 ... vK], your original covariates/attributes, are independent of text counts C given SR projections Z=[z1 ... z_K]. dir == nothing returns projections in all directions.\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.srproj-Union{Tuple{T}, Tuple{AbstractArray{T,2},SparseMatrixCSC,Int64}} where T",
    "page": "Home",
    "title": "HurdleDMR.srproj",
    "category": "method",
    "text": "Like srproj but efficiently interates over a sparse counts matrix, and only projects in a single direction (dir).\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.srprojX-Union{Tuple{M}, Tuple{T}, Tuple{M,M,Any,Any,Int64}} where M<:AbstractArray{T,2} where T",
    "page": "Home",
    "title": "HurdleDMR.srprojX",
    "category": "method",
    "text": "Builds the design matrix X for predicting covar in direction projdir   hdmr version   Assumes that covars include all variables for both positives and zeros models   and indicates which variables are where with the index arrays inpos and inzero.   inz=[1,2] if both zpos and zzero are included   inz=[2] if zpos is dropped due to collinearity\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.srprojX-Union{Tuple{T}, Tuple{AbstractArray{T,2},Any,Any,Any}} where T",
    "page": "Home",
    "title": "HurdleDMR.srprojX",
    "category": "method",
    "text": "Builds the design matrix X for predicting covar in direction projdir   dmr version   inz=[1] and testrank=false always for dmr, so variables are ignored and only here for convinence     of unified calling function\n\n\n\n\n\n"
},

{
    "location": "#Sufficient-reduction-projection-1",
    "page": "Home",
    "title": "Sufficient reduction projection",
    "category": "section",
    "text": "A sufficient reduction projection summarizes the counts, much like a sufficient statistic, and is useful for reducing the d dimensional counts in a potentially much lower dimension matrix z.To get a sufficient reduction projection in direction of vy for the above examplez = srproj(m,counts,1,1)Here, the first column is the SR projection from the model for positive counts, the second is the the SR projection from the model for hurdle crossing (zeros), and the third is the total count for each observation.Syntax:Modules = [HurdleDMR]\nOrder   = [:macro, :type, :function]\nPages   = [\"src/srproj.jl\"]\nPrivate = false"
},

{
    "location": "#HurdleDMR.CIR",
    "page": "Home",
    "title": "HurdleDMR.CIR",
    "category": "type",
    "text": "Counts inverse regression (CIR) model supports both multinomial and hurdle inverse regressions and holds both the inverse and forward regression model estimates\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.coefbwd-Tuple{CIR}",
    "page": "Home",
    "title": "HurdleDMR.coefbwd",
    "category": "method",
    "text": "Returns coefficients for backward model for counts as function of covariates\n\n\n\n\n\n"
},

{
    "location": "#HurdleDMR.coeffwd-Tuple{CIR}",
    "page": "Home",
    "title": "HurdleDMR.coeffwd",
    "category": "method",
    "text": "Returns coefficients of forward regression model. Set nocounts=true to get coefficients for the benchmark model without counts.\n\n\n\n\n\n"
},

{
    "location": "#StatsBase.fit-Union{Tuple{C}, Tuple{FM}, Tuple{BM}, Tuple{Type{C},Model,AbstractDataFrame,AbstractArray{T,2} where T,Symbol,Vararg{Any,N} where N}} where C<:CIR{BM,FM} where FM<:StatsBase.RegressionModel where BM<:DMR",
    "page": "Home",
    "title": "StatsBase.fit",
    "category": "method",
    "text": "fit(CIR{DMR,FM},m,df,counts,projdir[,fmargs...]; <keyword arguments>)\n\nVersion of fit(CIR{DMR,FM}...) that takes a @model() and a dataframe instead of a covars matrix, and a projdir::Symbol specifies the dependent variable. See also fit(CIR...).\n\nExample:\n\n  m = fit(CIR{DMR,LinearModel}, @model(c~x1+x2), df, counts, :x1; nocounts=true)\n\nwhere c~ is the model for counts. x1 (projdir) is the variable to predict. We can then predict with a dataframe as well\n\n  yhat = predict(m, df, counts)\n  yhatnc = predict(m, df, counts; nocounts=true)\n\n\n\n\n\n"
},

{
    "location": "#StatsBase.fit-Union{Tuple{C}, Tuple{FM}, Tuple{BM}, Tuple{Type{C},Model,AbstractDataFrame,AbstractArray{T,2} where T,Symbol,Vararg{Any,N} where N}} where C<:CIR{BM,FM} where FM<:StatsBase.RegressionModel where BM<:HDMR",
    "page": "Home",
    "title": "StatsBase.fit",
    "category": "method",
    "text": "fit(CIR{HDMR,FM},m,df,counts,projdir[,fmargs...]; <keyword arguments>)\n\nVersion of fit(CIR{HDMR,FM}...) that takes a @model() and a dataframe instead of a covars matrix, and a projdir::Symbol specifies the dependent variable. See also fit(CIR...).\n\nExample:\n\n  m = fit(CIR{HDMR,LinearModel}, @model(h~x1+x2, c~x1), df, counts, :x1; nocounts=true)\n\nwhere h~ is the model for zeros, c~ is the model for positives. x1 (projdir) is the variable to predict. We can then predict with a dataframe as well\n\n  yhat = predict(m, df, counts)\n  yhatnc = predict(m, df, counts; nocounts=true)\n\n\n\n\n\n"
},

{
    "location": "#StatsBase.fit-Union{Tuple{C}, Tuple{FM}, Tuple{BM}, Tuple{V}, Tuple{T}, Tuple{Type{C},AbstractArray{T,2},AbstractArray{V,2},Int64,Vararg{Any,N} where N}} where C<:CIR{BM,FM} where FM<:StatsBase.RegressionModel where BM<:DCR where V where T<:AbstractFloat",
    "page": "Home",
    "title": "StatsBase.fit",
    "category": "method",
    "text": "fit(::CIR{BM,FM},covars,counts,projdir[,fmargs...]; <keyword arguments>)\n\nFit a Counts Inverse Regression (CIR) of covars[:,projdir] ~ counts + covars[:,~projdir].\n\nCIR involves three steps:\n\nFit a backward regression model BM<:DCR: counts ~ covars\nCalculate an sufficient reduction projection in direction projdir\nFit a forward regression model FM<:RegressionModel:\n\ncovars[:,projdir] ~ srproj(counts) + covars[:,~projdir]\n\nExample:\n\n  m = fit(CIR{DMR,LinearModel}, covars, counts, 1; nocounts=true)\n  yhat = predict(m, covars, counts)\n  yhatnc = predict(m, covars, counts; nocounts=true)\n\nArguments\n\ncovars n-by-p matrix of covariates\ncounts n-by-d matrix of counts (usually sparse)\nprojdir index of covars column used as dependent variable in forward model\nfmargs... optional arguments passed along to the forward regression model\n\nKeywords\n\nnocounts::Bool=false whether to also fit a benchmark model without counts\nbmkwargs... keyword arguments passed along to the backward regression model\n\n\n\n\n\n"
},

{
    "location": "#StatsBase.predict-Union{Tuple{MM}, Tuple{M}, Tuple{T}, Tuple{MM,AbstractDataFrame,AbstractArray{T,2} where T}} where MM<:Union{DataFrameRegressionModel{M,T}, CIR} where M<:CIR where T",
    "page": "Home",
    "title": "StatsBase.predict",
    "category": "method",
    "text": "Predict using a fitted Counts inverse regression (CIR) given new covars dataframe and counts. See also predict(::CIR).\n\n\n\n\n\n"
},

{
    "location": "#StatsBase.predict-Union{Tuple{V}, Tuple{T}, Tuple{CIR,AbstractArray{T,2},AbstractArray{V,2}}} where V where T<:AbstractFloat",
    "page": "Home",
    "title": "StatsBase.predict",
    "category": "method",
    "text": "Predict using a fitted Counts inverse regression (CIR) given new covars and counts.\n\nKeywords\n\nSet nocounts=true to predict using a benchmark model without counts.\n\n\n\n\n\n"
},

{
    "location": "#Counts-Inverse-Regression-(CIR)-1",
    "page": "Home",
    "title": "Counts Inverse Regression (CIR)",
    "category": "section",
    "text": "Counts inverse regression allows us to predict a covariate with the counts and other covariates. Here we use hdmr for the backward regression and another model for the forward regression. This can be accomplished with a single command, by fitting a CIR{HDMR,FM} where the forward model is FM <: RegressionModel.cir = fit(CIR{HDMR,LinearModel},mf,covarsdf,counts,:vy; nocounts=true)where the nocounts=true means we also fit a benchmark model without counts.we can get the forward and backward model coefficients withcoefbwd(cir)\ncoeffwd(cir)The fitted model can be used to predict vy with new datayhat = predict(cir, covarsdf[1:10,:], counts[1:10,:])We can also predict only with the other covariates, which in this case is just a linear regressionyhat_nocounts = predict(cir, covarsdf[1:10,:], counts[1:10,:]; nocounts=true)Syntax:Modules = [HurdleDMR]\nOrder   = [:macro, :type, :function]\nPages   = [\"src/invreg.jl\"]\nPrivate = false"
},

{
    "location": "#HurdleDMR.Hurdle",
    "page": "Home",
    "title": "HurdleDMR.Hurdle",
    "category": "type",
    "text": "Hurdle returned object\n\n\n\n\n\n"
},

{
    "location": "#StatsBase.coef-Tuple{Hurdle}",
    "page": "Home",
    "title": "StatsBase.coef",
    "category": "method",
    "text": "coef(m::Hurdle; <keyword arguments>)\n\nReturns the AICc optimal coefficient matrices fitted the Hurdle.\n\nExample:\n\n  m = fit(Hurdle,GeneralizedLinearModel,X,y; Xpos=Xpos)\n  coefspos, coefszero = coef(m)\n\nKeywords\n\nkwargs... are passed along to two coef() calls on the two model parts.\n\n\n\n\n\n"
},

{
    "location": "#StatsBase.fit-Union{Tuple{M}, Tuple{Type{Hurdle},Type{M},Formula,AbstractDataFrame}, Tuple{Type{Hurdle},Type{M},Formula,AbstractDataFrame,Distribution{Univariate,S} where S<:ValueSupport}, Tuple{Type{Hurdle},Type{M},Formula,AbstractDataFrame,Distribution{Univariate,S} where S<:ValueSupport,Distribution{Univariate,S} where S<:ValueSupport}, Tuple{Type{Hurdle},Type{M},Formula,AbstractDataFrame,Distribution{Univariate,S} where S<:ValueSupport,Distribution{Univariate,S} where S<:ValueSupport,Link}, Tuple{Type{Hurdle},Type{M},Formula,AbstractDataFrame,Distribution{Univariate,S} where S<:ValueSupport,Distribution{Univariate,S} where S<:ValueSupport,Link,Link}} where M<:StatsBase.RegressionModel",
    "page": "Home",
    "title": "StatsBase.fit",
    "category": "method",
    "text": "fit(Hurdle,M,f,df; fpos=Xpos, <keyword arguments>)\n\nTakes dataframe and two formulas, one for each model part. Otherwise same arguments as fit(::Hurdle)\n\nExample\n\n  fit(Hurdle,GeneralizedLinearModel,@formula(y ~ x1*x2), df; fpos=@formula(y ~ x1*x2+x3))\n\n\n\n\n\n"
},

{
    "location": "#StatsBase.fit-Union{Tuple{V}, Tuple{T}, Tuple{M}, Tuple{Type{Hurdle},Type{M},AbstractArray{T,2},V}, Tuple{Type{Hurdle},Type{M},AbstractArray{T,2},V,Distribution{Univariate,S} where S<:ValueSupport}, Tuple{Type{Hurdle},Type{M},AbstractArray{T,2},V,Distribution{Univariate,S} where S<:ValueSupport,Distribution{Univariate,S} where S<:ValueSupport}, Tuple{Type{Hurdle},Type{M},AbstractArray{T,2},V,Distribution{Univariate,S} where S<:ValueSupport,Distribution{Univariate,S} where S<:ValueSupport,Link}, Tuple{Type{Hurdle},Type{M},AbstractArray{T,2},V,Distribution{Univariate,S} where S<:ValueSupport,Distribution{Univariate,S} where S<:ValueSupport,Link,Link}} where V<:(AbstractArray{T,1} where T<:AbstractFloat) where T<:AbstractFloat where M<:StatsBase.RegressionModel",
    "page": "Home",
    "title": "StatsBase.fit",
    "category": "method",
    "text": "fit(Hurdle,M,X,y; Xpos=Xpos, <keyword arguments>)\n\nFit a Hurdle (Mullahy, 1986) of count vector y on X with potentially another covariates matrix Xpos used to model positive counts.\n\nExample with GLM:\n\n  m = fit(Hurdle,GeneralizedLinearModel,X,y; Xpos=Xpos)\n  yhat = predict(m, X; Xpos=Xpos)\n\nExample with Lasso regularization:\n\n  m = fit(Hurdle,GammaLassoPath,X,y; Xpos=Xpos)\n  yhat = predict(m, X; Xpos=Xpos, select=:AICc)\n\nArguments\n\nM::RegressionModel\ncounts n-by-d matrix of counts (usually sparse)\ndzero::UnivariateDistribution = Binomial() distribution for zeros model\ndpos::UnivariateDistribution = PositivePoisson() distribution for positives model\nlzero::Link=canonicallink(dzero) link function for zeros model\nlpos::Link=canonicallink(dpos) link function for positives model\n\nKeywords\n\nXpos::Union{AbstractMatrix{T},Nothing} = nothing covariates matrix for positives model or nothing to use X for both parts\ndofit::Bool = true fit the model or just construct its shell\nwts::V = ones(y) observation weights\noffsetzero::AbstractVector = similar(y, 0) offsets for zeros model\noffsetpos::AbstractVector = similar(y, 0) offsets for positives model\noffset::AbstractVector = similar(y, 0) offsets for both model parts\nverbose::Bool=true\nshowwarnings::Bool=false\nfitargs... additional keyword arguments passed along to fit(M,...)\n\n\n\n\n\n"
},

{
    "location": "#StatsBase.predict-Union{Tuple{T}, Tuple{Hurdle,AbstractArray{T,2}}} where T<:AbstractFloat",
    "page": "Home",
    "title": "StatsBase.predict",
    "category": "method",
    "text": "predict(m,X; Xpos=Xpos, <keyword arguments>)\n\nPredict using a fitted Hurdle given new X (and potentially Xpos).\n\nExample with GLM:\n\n  m = fit(Hurdle,GeneralizedLinearModel,X,y; Xpos=Xpos)\n  yhat = predict(m, X; Xpos=Xpos)\n\nExample with Lasso regularization:\n\n  m = fit(Hurdle,GammaLassoPath,X,y; Xpos=Xpos)\n  yhat = predict(m, X; Xpos=Xpos, select=:AICc)\n\nArguments\n\nm::Hurdle fitted Hurdle model\nX n-by-p matrix of covariates of same dimensions used to fit m.\n\nKeywords\n\nXpos::Union{AbstractMatrix{T},Nothing} = nothing covariates matrix for positives model or nothing to use X for both parts\nkwargs... additional keyword arguments passed along to predict() for each of the two model parts.\n\n\n\n\n\n"
},

{
    "location": "#Hurdle-1",
    "page": "Home",
    "title": "Hurdle",
    "category": "section",
    "text": "This package also provides a regularized Hurdle model (Mullahy, 1986) that can be fit using a fast coordinate decent algorithm, or simply by running two fit(GeneralizedLinearModel,...) regressions, one for each of its two parts.Syntax:Modules = [HurdleDMR]\nOrder   = [:macro, :type, :function]\nPages   = [\"src/hurdle.jl\"]\nPrivate = false"
},

{
    "location": "#HurdleDMR.PositivePoisson",
    "page": "Home",
    "title": "HurdleDMR.PositivePoisson",
    "category": "type",
    "text": "PositivePoisson(λ)\n\nA PositivePoisson distribution (aka zero-truncated Poisson, ZTP) descibes the number of independent events occurring within a unit time interval, given the average rate of occurrence λ and, importantly, given that the number is not zero.\n\nP(X = k) = fracλ^kk(1-e^-λ) e^-λ quad text for  k = 12ldots\n\nPositivePoisson()        # PositivePoisson distribution with rate parameter 1\nPositivePoisson(lambda)       # PositivePoisson distribution with rate parameter lambda\n\nparams(d)        # Get the parameters, i.e. (λ,)\nmean(d)          # Get the mean arrival rate, i.e. λ\n\nExternal links:\n\nPositivePoisson distribution on Wikipedia\n\n\n\n\n\n"
},

{
    "location": "#Positive-Poisson-1",
    "page": "Home",
    "title": "Positive Poisson",
    "category": "section",
    "text": "This package also implements the PositivePoisson distribution and the GLM necessary methods to facilitate fit with fit(::GeneralizedLinearModel.Syntax:Modules = [HurdleDMR]\nOrder   = [:macro, :type, :function]\nPages   = [\"src/positive_poisson.jl\"]\nPrivate = false"
},

{
    "location": "#API-/-Index-1",
    "page": "Home",
    "title": "API / Index",
    "category": "section",
    "text": ""
},

{
    "location": "tutorials/#",
    "page": "Tutorials",
    "title": "Tutorials",
    "category": "page",
    "text": ""
},

{
    "location": "tutorials/#Tutorials-1",
    "page": "Tutorials",
    "title": "Tutorials",
    "category": "section",
    "text": "The following tutorials demonstrate how to use this package from different languages.Julia tutorial (source).\nR tutorial (source).\nPython tutorial TBD\nMatlab tutorial TBD"
},

]}
