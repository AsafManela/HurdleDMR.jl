using Base.Test
using HurdleDMR

@testset "crossvalidation" begin

resultstats = [CVStats(Float64), CVStats(Float64)]
@test typeof(resultstats) <: AbstractArray{T} where {T <: CVType}
@test typeof(resultstats[1]) <: CVType{Float64}
@test typeof(DataFrame(resultstats)) <: DataFrame

cvd1 = CVData([[] for i=1:6]...)
cvd2 = CVData([[] for i=1:6]...)
cvd12 = CVData([[] for i=1:6]...)
append!(cvd1,cvd2)
@test hash(cvd1) == hash(cvd12)

cvd1 = CVData([[1.0,2.0] for i=1:6]...)
cvd2 = CVData([[3.0,4.0,5.0] for i=1:6]...)
cvd12 = CVData([[1.0,2.0,3.0,4.0,5.0] for i=1:6]...)
append!(cvd1,cvd2)
@test hash(cvd1) == hash(cvd12)

end
