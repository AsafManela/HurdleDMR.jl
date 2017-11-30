using FactCheck
using HurdleDMR

facts("crossvalidation") do

resultstats = [CVStats(Float64), CVStats(Float64)]
@fact typeof(resultstats) <: AbstractArray{T} where {T <: CVType} --> true
@fact typeof(resultstats[1]) <: HurdleDMR.CVType{Float64} --> true
@fact typeof(DataFrame(resultstats)) <: DataFrame --> true

cvd1 = CVData([[] for i=1:6]...)
cvd2 = CVData([[] for i=1:6]...)
cvd12 = CVData([[] for i=1:6]...)
append!(cvd1,cvd2)
@fact hash(cvd1) --> hash(cvd12)

cvd1 = CVData([[1.0,2.0] for i=1:6]...)
cvd2 = CVData([[3.0,4.0,5.0] for i=1:6]...)
cvd12 = CVData([[1.0,2.0,3.0,4.0,5.0] for i=1:6]...)
append!(cvd1,cvd2)
@fact hash(cvd1) --> hash(cvd12)

end
