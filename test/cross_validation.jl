using Base.Test
using HurdleDMR

@testset "crossvalidation" begin

resultstats = [CVStats(Float64), CVStats(Float64)]
@test typeof(resultstats) <: AbstractArray{T} where {T <: CVType}
@test typeof(resultstats[1]) <: CVType{Float64}
@test typeof(DataFrame(resultstats)) <: DataFrame

cvd0 = CVData(Float64)
cvd1 = CVData{Float64}([[] for i=1:6]...)
@test hash(cvd0) == hash(cvd1)

cvd2 = CVData(Float64)
cvd12 = CVData{Float64}([[] for i=1:6]...)
append!(cvd1,cvd2)
@test hash(cvd1) == hash(cvd12)

cvd1 = CVData{Float64}([[[1.0,2.0]] for i=1:6]...)
cvd2 = CVData{Float64}([[[3.0,4.0,5.0]] for i=1:6]...)
cvd12 = CVData{Float64}([[[1.0,2.0],[3.0,4.0,5.0]] for i=1:6]...)
append!(cvd1,cvd2)
@test hash(cvd1) == hash(cvd12)

cvd1 = CVData{Float64}([[[1.0,2.0,3.0].+0.5ϵ,[4.0,5.0,6.0].+ϵ] for ϵ=(0.1*[0,0,0.5,0.6,0.9,1.0])]...)
ys = collect(1.0:6.0)
insϵs = -0.1*0.5*vcat(0.5*ones(3),ones(3))
insϵs_nocounts = -0.1*0.9*vcat(0.5*ones(3),ones(3))
insmse1 = (0.1*0.5*sqrt(((0.5)^2+1)/2))^2
insmse_nocounts1 = (0.1*0.9*sqrt(((0.5)^2+1)/2))^2
insr21 = 1.0 - sum(abs2,insϵs)/sum(abs2,ys.-mean(ys))
insr2_nocounts1 = 1.0 - sum(abs2,insϵs_nocounts)/sum(abs2,ys.-mean(ys))
insσmse1 = std((0.1*0.5)^2*[0.5^2,1.0])/sqrt(2-1)
insσmse_nocounts1 = std((0.1*0.9)^2*[0.5^2,1.0])/sqrt(2-1)
insσΔmse1 = std((0.1*0.5)^2*[0.5^2,1.0].-(0.1*0.9)^2*[0.5^2,1.0])/sqrt(2-1)
# insσrmse1 = insσmse1 / (2.0*insσmse1)
oosϵs = -0.1*0.6*vcat(0.5*ones(3),ones(3))
oosϵs_nocounts = -0.1*1.0vcat(0.5*ones(3),ones(3))
oosmse1 = (0.1*0.6*sqrt(((0.5)^2+1)/2))^2
oosmse_nocounts1 = (0.1*sqrt(((0.5)^2+1)/2))^2
oosσmse1 = std((0.1*0.6)^2*[0.5^2,1.0])/sqrt(2-1)
oosσmse_nocounts1 = std((0.1*1.0)^2*[0.5^2,1.0])/sqrt(2-1)
oosσΔmse1 = std((0.1*0.6)^2*[0.5^2,1.0].-(0.1*1.0)^2*[0.5^2,1.0])/sqrt(2-1)
oosr21 = 1.0 - sum(abs2,oosϵs)/sum(abs2,ys.-mean(ys))
oosr2_nocounts1 = 1.0 - sum(abs2,oosϵs_nocounts)/sum(abs2,ys.-mean(ys))

@test insϵs ≈ vcat(cvd1.ins_ys...) - vcat(cvd1.ins_yhats...)
@test insϵs_nocounts ≈ vcat(cvd1.ins_ys...) - vcat(cvd1.ins_yhats_nocounts...)
@test oosϵs ≈ vcat(cvd1.ins_ys...) - vcat(cvd1.oos_yhats...)
@test oosϵs_nocounts ≈ vcat(cvd1.ins_ys...) - vcat(cvd1.oos_yhats_nocounts...)

# @enter CVStats(cvd1)
s1 = CVStats(cvd1)
@test s1.oos_mse ≈ oosmse1
@test s1.oos_mse_nocounts ≈ oosmse_nocounts1
@test s1.oos_σmse ≈ oosσmse1
@test s1.oos_σmse_nocounts ≈ oosσmse_nocounts1
@test s1.oos_change_mse ≈ oosmse1 - oosmse_nocounts1
@test s1.oos_σchange_mse ≈ oosσΔmse1
@test s1.ins_mse ≈ insmse1
@test s1.ins_mse_nocounts ≈ insmse_nocounts1
@test s1.ins_σmse ≈ insσmse1
@test s1.ins_σmse_nocounts ≈ insσmse_nocounts1
@test s1.ins_change_mse ≈ insmse1 - insmse_nocounts1
@test s1.ins_σchange_mse ≈ insσΔmse1
@test s1.oos_r2 ≈ oosr21
@test s1.oos_r2_nocounts ≈ oosr2_nocounts1
@test s1.ins_r2 ≈ insr21
@test s1.ins_r2_nocounts ≈ insr2_nocounts1

end
