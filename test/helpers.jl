@testset "helpers" begin

# test convert methods for SharedArray
Random.seed!(1)
A = sprand(100,4,0.3)
sA = convert(SharedArray,A)
@test A == sA

a = view(A,:,2:3)
@test a == getindex(A,:,2:3)
@test a == getindex(sA,:,2:3)

sa = convert(SharedArray,a)
@test a == sa

b = view(sA,:,2:3)
@test a == b

# test counts matrix coverter to float64 entries
fpcounts = HurdleDMR.fpcounts
C = convert(SparseMatrixCSC{Int},counts)
D = Matrix(C)
Cf = fpcounts(C)
Df = fpcounts(D)
@test C == Cf
@test C !== Cf
@test D == Df
@test D !== Df
# new matrices have the correct type
@test Cf isa SparseMatrixCSC{Float64}
@test Df isa Matrix{Float64}
# test no unnecessary covertions
@test Cf === fpcounts(Cf)
@test Df === fpcounts(Df)

# test shifters
shifters = HurdleDMR.shifters

# DMR shifters
tcovars, tcounts, tμ, tn = shifters(DMR, covars, counts, true, nothing)
eμ = vec(log.(sum(tcounts, dims=2)))
@test tμ == eμ
@test tcounts == counts

Cf = convert(Matrix{Float64},counts)
tcovars, tcounts, tμ, tn = shifters(DMR, covars, Cf, true, nothing)
@test tcounts === Cf
@test tμ == eμ

tcovars, tcounts, tμ, tn = shifters(DMR, covars, convert(SparseMatrixCSC{Int},counts), true, nothing)
@test tcounts == counts
@test tcounts !== counts
@test tμ == eμ

tcovars, tcounts, tμ, tn = shifters(DMR, covars, convert(Matrix{Int},counts), true, nothing)
@test tcounts == Cf
@test tcounts !== Cf
@test tμ == eμ

Cf[1,:] .= 0
tcovars, tcounts, tμ, tn = @test_logs (:warn, r"omitting 1 observations with no counts") shifters(DMR, covars, Cf, true, nothing)
@test tcounts != Cf
@test tcounts == Cf[2:end,:]
@test tμ == eμ[2:end]
@test tcovars == covars[2:end,:]

prem = vec(sum(counts, dims=2))
pcovars, pcounts, pμ, pn = shifters(DMR, covars, counts[:,1:3], false, prem)
eμ = log.(prem)
@test pμ == eμ
@test pcounts == counts[:,1:3]

# HDMR/Hurdle shifters
M = Hurdle
tcovars, tcounts, tμpos, tμzero, tn = shifters(M, covars, counts, true, nothing, nothing, nothing)
eμ = vec(log.(sum(tcounts, dims=2)))
@test tμpos == eμ
@test tμzero == eμ
@test tcounts == counts

Cf = convert(Matrix{Float64},counts)
tcovars, tcounts, tμpos, tμzero, tn = shifters(M, covars, Cf, true, nothing, nothing, nothing)
@test tcounts === Cf
@test tμpos == eμ
@test tμzero == eμ

tcovars, tcounts, tμpos, tμzero, tn = shifters(M, covars, convert(SparseMatrixCSC{Int},counts), true, nothing, nothing, nothing)
@test tcounts == counts
@test tcounts !== counts
@test tμpos == eμ
@test tμzero == eμ

tcovars, tcounts, tμpos, tμzero, tn = shifters(M, covars, convert(Matrix{Int},counts), true, nothing, nothing, nothing)
@test tcounts == Cf
@test tcounts !== Cf
@test tμpos == eμ
@test tμzero == eμ

Cf[1,:] .= 0
tcovars, tcounts, tμpos, tμzero, tn = @test_logs (:warn, r"omitting 1 observations with no counts") shifters(M, covars, Cf, true, nothing, nothing, nothing)
@test tcounts != Cf
@test tcounts == Cf[2:end,:]
@test tμpos == eμ[2:end]
@test tμzero == eμ[2:end]
@test tcovars == covars[2:end,:]

prem = vec(sum(counts, dims=2))
pred = vec(sum(posindic(counts), dims=2))
pcovars, pcounts, pμpos, pμzero, pn = @test_logs (:warn, r"Old Hurdle-DMR model does not use prespecified l") shifters(M, covars, counts[:,1:3], true, prem, pred, nothing)
@test pμpos == eμ
@test pμzero == eμ
@test pcounts == counts[:,1:3]

# HDMR/InclusionRepetition shifters
M = InclusionRepetition
tcovars, tcounts, tμpos, tμzero, tn = shifters(M, covars, counts, true, nothing, nothing, nothing)

prem = vec(sum(counts, dims=2))
pred = vec(sum(posindic(counts), dims=2))
preD = size(counts, 2)
eμpos = log.(prem .- pred)
eμzero = log.(pred ./ (d .- pred))
@test tμpos == eμpos
@test tμzero == eμzero
@test tcounts == counts

Cf = convert(Matrix{Float64},counts)
tcovars, tcounts, tμpos, tμzero, tn = shifters(M, covars, Cf, true, nothing, nothing, nothing)
@test tcounts === Cf
@test tμpos == eμpos
@test tμzero == eμzero

tcovars, tcounts, tμpos, tμzero, tn = shifters(M, covars, convert(SparseMatrixCSC{Int},counts), true, nothing, nothing, nothing)
@test tcounts == counts
@test tcounts !== counts
@test tμpos == eμpos
@test tμzero == eμzero

tcovars, tcounts, tμpos, tμzero, tn = shifters(M, covars, convert(Matrix{Int},counts), true, nothing, nothing, nothing)
@test tcounts == Cf
@test tcounts !== Cf
@test tμpos == eμpos
@test tμzero == eμzero

Cf[1,:] .= 0
tcovars, tcounts, tμpos, tμzero, tn = @test_logs (:warn, r"omitting 1 observations with no counts") shifters(M, covars, Cf, true, nothing, nothing, nothing)
@test tcounts != Cf
@test tcounts == Cf[2:end,:]
@test tμpos == eμpos[2:end]
@test tμzero == eμzero[2:end]
@test tcovars == covars[2:end,:]

Cf[1,:] .= 1
tcovars, tcounts, tμpos, tμzero, tn = shifters(M, covars, Cf, true, nothing, nothing, nothing)
@test tcounts == Cf
@test tμpos == [-Inf; eμpos[2:end]]
@test tμzero == [Inf; eμzero[2:end]]
@test tcovars == covars

pcovars, pcounts, pμpos, pμzero, pn = shifters(M, covars, counts[:,1:3], true, prem, pred, preD)
eμpos = log.(prem .- pred)
eμzero = log.(pred ./ (preD .- pred))
@test pμpos == eμpos
@test pμzero == eμzero
@test pcounts == counts[:,1:3]

# test posindic used by srproj
m = rand(Poisson(0.1),30,500)
ms = sparse(m)
Im = posindic(m)
Ims = posindic(ms)
@test Im == Ims
@test eltype(Im) == Int
@test eltype(Ims) == Int

m = float.(m)
ms = sparse(m)
Im = posindic(m)
Ims = posindic(ms)
@test Im == Ims
@test eltype(Im) == Float64
@test eltype(Ims) == Float64

end

@testset "logging" begin
    @testset "getlogger(true) = $(HurdleDMR.getlogger(true))" begin
        @test_logs (:warn, "warn!") with_logger(HurdleDMR.getlogger(true)) do
            @warn "warn!"
        end
    end
    @testset "getlogger(false) = $(HurdleDMR.getlogger(false))" begin
        @test_nowarn with_logger(HurdleDMR.getlogger(false)) do
            @warn "warn!"
        end
    end
end
