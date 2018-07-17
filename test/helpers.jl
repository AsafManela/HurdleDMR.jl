@testset "helpers" begin

# test convert methods for SharedArray
srand(1)
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

end
