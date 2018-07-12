using DataStructures
using HurdleDMR

@testset "crossvalidation" begin

skf = SerialKfold(4, 2)
@test collect(skf) == [[3,4],[1,2]]

skf = SerialKfold(5, 2)
@test collect(skf) == [[3,4,5],[1,2,5]]

skf = LeaveOutSample(6, 2)
@test collect(skf) == [[1,2,3]]

skf = LeaveOutSample(5, 2)
@test collect(skf) == [[1,2,3]]

skf = LeaveOutSample(5, 2; forward=false)
@test collect(skf) == [[3,4,5]]

skf = LeaveOutSample(5, 10; testlength=4)
@test collect(skf) == [[1]]

@test_throws ErrorException LeaveOutSample(5, 0; testlength=5)

srand(14)
skf = LeaveOutSample(5, 2; random=true)
@test collect(skf) == [[2,3,5]]

srand(14)
skf = LeaveOutSample(5, 2; testlength=3, random=true)
@test collect(skf) == [[2,5]]

cvd0 = CVData(Float64)
cvd1 = CVData([Float64[] for i=1:6]...)
@test hash(cvd0) != hash(cvd1)

cvd2 = CVData(Float64)
cvd12 = CVData([Float64[] for i=1:6]...)
append!(cvd1,cvd2)
@test hash(cvd1) == hash(cvd12)

cvd1 = CVData([[[1.0,2.0]] for i=1:6]...)
cvd1b = CVData([[1.0,2.0] for i=1:6]...)
@test hash(cvd1) == hash(cvd1b)

cvd2 = CVData([[[3.0,4.0,5.0]] for i=1:6]...)
cvd12 = CVData([[[1.0,2.0],[3.0,4.0,5.0]] for i=1:6]...)
append!(cvd1,cvd2)
@test hash(cvd1) == hash(cvd12)

cvd1 = CVData([[[1.0,2.0,3.0].+0.5ϵ,[4.0,5.0,6.0].+ϵ] for ϵ=(0.1*[0,0,0.5,0.6,0.9,1.0])]...)
ys = collect(1.0:6.0)
insϵs = -0.1*0.5*vcat(0.5*ones(3),ones(3))
insϵs_nocounts = -0.1*0.9*vcat(0.5*ones(3),ones(3))
insrmse1 = (0.1*0.5*sqrt(((0.5)^2+1)/2))
insrmse_nocounts1 = (0.1*0.9*sqrt(((0.5)^2+1)/2))
insmse1 = insrmse1^2
insmse_nocounts1 = insrmse_nocounts1^2
insr21 = 1.0 - sum(abs2,insϵs)/sum(abs2,ys.-mean(ys))
insr2_nocounts1 = 1.0 - sum(abs2,insϵs_nocounts)/sum(abs2,ys.-mean(ys))
insσmse1 = std((0.1*0.5)^2*[0.5^2,1.0])/sqrt(2-1)
insσmse_nocounts1 = std((0.1*0.9)^2*[0.5^2,1.0])/sqrt(2-1)
insσrmse1 = insσmse1 / (2*insrmse1)
insσrmse_nocounts1 = insσmse_nocounts1 / (2*insrmse_nocounts1)
insσΔmse1 = std((0.1*0.5)^2*[0.5^2,1.0].-(0.1*0.9)^2*[0.5^2,1.0])/sqrt(2-1)
insσΔrmse1 = std((0.1*0.5)*[0.5,1.0].-(0.1*0.9)*[0.5,1.0])/sqrt(2-1)
# insσrmse1 = insσmse1 / (2.0*insσmse1)
oosϵs = -0.1*0.6*vcat(0.5*ones(3),ones(3))
oosϵs_nocounts = -0.1*1.0vcat(0.5*ones(3),ones(3))
oosrmse1 = (0.1*0.6*sqrt(((0.5)^2+1)/2))
oosrmse_nocounts1 = (0.1*sqrt(((0.5)^2+1)/2))
oosmse1 = oosrmse1^2
oosmse_nocounts1 = oosrmse_nocounts1^2
oosσmse1 = std((0.1*0.6)^2*[0.5^2,1.0])/sqrt(2-1)
oosσmse_nocounts1 = std((0.1*1.0)^2*[0.5^2,1.0])/sqrt(2-1)
oosσrmse1 = oosσmse1 / (2*oosrmse1)
oosσrmse_nocounts1 = oosσmse_nocounts1 / (2*oosrmse_nocounts1)
oosσΔmse1 = std((0.1*0.6)^2*[0.5^2,1.0].-(0.1*1.0)^2*[0.5^2,1.0])/sqrt(2-1)
oosσΔrmse1 = std((0.1*0.6)*[0.5,1.0].-(0.1*1.0)*[0.5,1.0])/sqrt(2-1)
oosr21 = 1.0 - sum(abs2,oosϵs)/sum(abs2,ys.-mean(ys))
oosr2_nocounts1 = 1.0 - sum(abs2,oosϵs_nocounts)/sum(abs2,ys.-mean(ys))

@test insϵs ≈ vcat(cvd1.ins_ys...) - vcat(cvd1.ins_yhats...)
@test insϵs_nocounts ≈ vcat(cvd1.ins_ys...) - vcat(cvd1.ins_yhats_nocounts...)
@test oosϵs ≈ vcat(cvd1.ins_ys...) - vcat(cvd1.oos_yhats...)
@test oosϵs_nocounts ≈ vcat(cvd1.ins_ys...) - vcat(cvd1.oos_yhats_nocounts...)

s1 = cvstats(OrderedDict,cvd1; stats=[:mse, :r2])
@test s1[:oos_mse] ≈ oosmse1
@test s1[:oos_mse_nocounts] ≈ oosmse_nocounts1
@test s1[:oos_σmse] ≈ oosσmse1
@test s1[:oos_σmse_nocounts] ≈ oosσmse_nocounts1
@test s1[:oos_change_mse] ≈ oosmse1 - oosmse_nocounts1
@test s1[:oos_σchange_mse] ≈ oosσΔmse1
@test s1[:ins_mse] ≈ insmse1
@test s1[:ins_mse_nocounts] ≈ insmse_nocounts1
@test s1[:ins_σmse] ≈ insσmse1
@test s1[:ins_σmse_nocounts] ≈ insσmse_nocounts1
@test s1[:ins_change_mse] ≈ insmse1 - insmse_nocounts1
@test s1[:ins_σchange_mse] ≈ insσΔmse1
@test s1[:oos_r2] ≈ oosr21
@test s1[:oos_r2_nocounts] ≈ oosr2_nocounts1
@test s1[:ins_r2] ≈ insr21
@test s1[:ins_r2_nocounts] ≈ insr2_nocounts1
@test_throws KeyError s1[:oos_rmse]

s1 = cvstats(OrderedDict,cvd1; stats=[:rmse, :r2])
@test s1[:oos_rmse] ≈ oosrmse1
@test s1[:oos_rmse_nocounts] ≈ sqrt(oosmse_nocounts1)
@test s1[:oos_σrmse] ≈ oosσmse1 / (2*sqrt(oosmse1))
@test s1[:oos_σrmse_nocounts] ≈ oosσmse_nocounts1  / (2*sqrt(oosmse_nocounts1))
@test s1[:oos_change_rmse] ≈ sqrt(oosmse1) - sqrt(oosmse_nocounts1)
@test s1[:oos_σchange_rmse] ≈ oosσΔrmse1
@test s1[:ins_rmse] ≈ insrmse1
@test s1[:ins_rmse_nocounts] ≈ insrmse_nocounts1
@test s1[:ins_σrmse] ≈ insσrmse1
@test s1[:ins_σrmse_nocounts] ≈ insσrmse_nocounts1
@test s1[:ins_change_rmse] ≈ insrmse1 - insrmse_nocounts1
@test s1[:ins_σchange_rmse] ≈ insσΔrmse1
@test s1[:oos_r2] ≈ oosr21
@test s1[:oos_r2_nocounts] ≈ oosr2_nocounts1
@test s1[:ins_r2] ≈ insr21
@test s1[:ins_r2_nocounts] ≈ insr2_nocounts1

@test_nowarn DataFrame(s1)

# without specifying return type
s2 = cvstats(cvd1; stats=[:rmse, :r2])
for k in keys(s1)
  @test s1[k] == s2[k]
end

end
