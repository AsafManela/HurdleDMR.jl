using HurdleDMR, FactCheck

facts("crossvalidation") do
    resultstats = [CVStats(Float64),CVStats(Float64)]
    @fact typeof(resultstats) <: AbstractArray{T} where {T <: CVType} --> true
    @fact typeof(resultstats[1]) <: HurdleDMR.CVType{Float64} --> true
    @fact typeof(DataFrame(resultstats)) <: DataFrame --> true
end
