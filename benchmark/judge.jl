# a short exmple script comparing two branches
using BenchmarkTools, PkgBenchmark
import HurdleDMR

cd(@__DIR__)

pkg = joinpath(dirname(pathof(HurdleDMR)),"..")
env = Dict(
        "JULIA_NUM_THREADS" => Sys.CPU_THREADS-2, 
        "JULIA_PROJECT" => pkg
    )

# now switch branches and run
current = benchmarkpkg(pkg, BenchmarkConfig(env = env))
writeresults("results.current.json", current)

# checkout master
baseline = benchmarkpkg(pkg, BenchmarkConfig(id = "master", env = env))
writeresults("results.baseline.json", baseline)
#baseline = readresults("results.baseline.json")

export_markdown("benchmark.md", judge(current, baseline))
