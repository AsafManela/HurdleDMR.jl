# a short exmple script comparing two branches
using BenchmarkTools, PkgBenchmark

cd(@__DIR__)

# checkout master
benchmark = benchmarkpkg("HurdleDMR", "master")
#writeresults("results.master.json", benchmark)
#benchmark = readresults("results.master.json")

# now switch branches and run
target = benchmarkpkg("HurdleDMR", "typesafe_segselect")

export_markdown("benchmark.md", judge(target, benchmark))
