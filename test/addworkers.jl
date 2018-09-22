#BLAS.set_num_threads(1) # this one should have worked but did not
ENV["OPENBLAS_NUM_THREADS"] = 1 # prevents thrashing by pmap + blas

# travis-ci limits to 4 or so
const nw = 4# Sys.CPU_CORES-2
if nworkers() < nw
    if nworkers() > 1
        @info("Removing existing parallel workers for tests...")
        rmprocs(workers())
    end
    @info("Starting $nw parallel workers for tests...")
    addprocs(nw)
    @info("$(nworkers()) parallel workers started")
end
