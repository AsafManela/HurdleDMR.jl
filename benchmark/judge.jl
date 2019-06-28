# a short exmple script comparing two branches
using BenchmarkTools, PkgBenchmark

cd(@__DIR__)

# checkout master
baseline = benchmarkpkg("HurdleDMR", "master")
#writeresults("results.master.json", baseline)
#baseline = readresults("results.master.json")

# now switch branches and run
target = benchmarkpkg("HurdleDMR", "typesafe_segselect")

export_markdown("benchmark.md", judge(target, baseline))
