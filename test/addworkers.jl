#BLAS.set_num_threads(1) # this one should have worked but did not
ENV["OPENBLAS_NUM_THREADS"] = 1 # this prevents thrashing by pmap + blas

const nw = Sys.CPU_CORES-2
if nworkers() > 1
  rmprocs(workers())
end
info("Starting $nw parallel workers for tests...")
addprocs(nw)
