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

tcovars, tcounts, tμ, tn = shifters(covars, counts, false)
eμ = vec(log.(sum(tcounts, dims=2)))
@test tμ == eμ
@test tcounts == counts

Cf = convert(Matrix{Float64},counts)
tcovars, tcounts, tμ, tn = shifters(covars, Cf, false)
@test tcounts === Cf
@test tμ == eμ

tcovars, tcounts, tμ, tn = shifters(covars, convert(SparseMatrixCSC{Int64},counts), false)
@test tcounts == counts
@test tcounts !== counts
@test tμ == eμ

tcovars, tcounts, tμ, tn = shifters(covars, convert(Matrix{Int},counts), false)
@test tcounts == Cf
@test tcounts !== Cf
@test tμ == eμ

Cf[1,:] .= 0
tcovars, tcounts, tμ, tn = shifters(covars, Cf, false)
@test tcounts != Cf
@test tcounts == Cf[2:end,:]
@test tμ == eμ[2:end]
@test tcovars == covars[2:end,:]


end
