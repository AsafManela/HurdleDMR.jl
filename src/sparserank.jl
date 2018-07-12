# Code from Steve Vavasis 2016 implementing rank(A::SparseMatrixCSC)

module MySparseExtensions

import Base.sparse

if VERSION < v"0.5"
    import Base.SparseMatrix.CHOLMOD.FactorComponent
    import Base.SparseMatrix.CHOLMOD.Factor
    import Base.SparseMatrix.CHOLMOD.CHOLMODException
    import Base.SparseMatrix.CHOLMOD.common
    import Base.SparseMatrix.CHOLMOD.C_Sparse
    import Base.SparseMatrix.CHOLMOD.Sparse
    import Base.SparseMatrix.CHOLMOD.free_sparse!
    import Base.SparseMatrix.CHOLMOD.increment
    import Base.SparseMatrix.CHOLMOD.SuiteSparse_long
    import Base.SparseMatrix.CHOLMOD.defaults
    import Base.SparseMatrix.CHOLMOD.fact_
    import Base.SparseMatrix.CHOLMOD.set_print_level
    import Base.SparseMatrix.CHOLMOD.common_final_ll
    import Base.SparseMatrix.SPQR.ORDERING_DEFAULT
    import Base.SparseMatrix.CHOLMOD.get_perm
else
    import Base.SparseArrays.CHOLMOD.FactorComponent
    import Base.SparseArrays.CHOLMOD.Factor
    import Base.SparseArrays.CHOLMOD.CHOLMODException
    import Base.SparseArrays.CHOLMOD.common
    import Base.SparseArrays.CHOLMOD.C_Sparse
    import Base.SparseArrays.CHOLMOD.Sparse
    import Base.SparseArrays.CHOLMOD.free_sparse!
    import Base.SparseArrays.CHOLMOD.increment
    import Base.SparseArrays.CHOLMOD.SuiteSparse_long
    import Base.SparseArrays.CHOLMOD.defaults
    import Base.SparseArrays.CHOLMOD.fact_
    import Base.SparseArrays.CHOLMOD.set_print_level
    import Base.SparseArrays.CHOLMOD.common_final_ll
    import Base.SparseArrays.SPQR.ORDERING_DEFAULT
    import Base.SparseArrays.CHOLMOD.get_perm
end


## Retrieve PtL factor from sparse Cholesky factorization
## See example below for usage

function sparse{Tv}(FC::FactorComponent{Tv,:PtL})
    F = Factor(FC)
    s = unsafe_load(F.p)
    s.is_ll != 0 || throw(CHOLMODException("sparse: supported for :PtL only on LLt factorizations"))
    s.is_super == 0 || throw(CHOLMODException("sparse: cannot invoke sparse() on supernodal factor; use change_factor! first"))
    s.n == s.minor || throw(CHOLMODException("sparse: cannot invoke sparse() on incomplete factor"))
    nnz = s.nzmax
    is = zeros(Int, nnz)
    js = zeros(Int, nnz)
    for oldcolnum = 1 : s.n
        newcolnum = unsafe_load(s.Perm, oldcolnum) + 1
        estart = unsafe_load(s.p, oldcolnum) + 1
        eend = unsafe_load(s.p, oldcolnum + 1)
        for pos = estart : eend
            js[pos] = newcolnum
            oldrownum = unsafe_load(s.i, pos) + 1
            newrownum = unsafe_load(s.Perm, oldrownum) + 1
            is[pos] = newrownum
        end
    end
    sparse(is, js, pointer_to_array(s.x, nnz, false), s.n, s.n)
end

## Retrieve R and colprm factor from sparse QR.  See below
## for usage.

function qrfact_get_r_colperm(A::SparseMatrixCSC{Float64, Int},
                              tol::Float64,
                              ordering = ORDERING_DEFAULT)
    Aw = Sparse(A,0)
    s = unsafe_load(Aw.p)
    if s.stype != 0
        throw(ArgumentError("stype must be zero"))
    end
    ptr_R = Ref{Ptr{C_Sparse{Float64}}}()
    ptr_E = Ref{Ptr{SuiteSparse_long}}()
    cc = common()
    rk = ccall((:SuiteSparseQR_C, :libspqr), Clong,
               (Cint, #ordering
                Cdouble, #tol
                Clong, #econ
                Cint, #getCTX
                Ptr{C_Sparse{Float64}},  # A
                Ptr{Void}, #Bsparse
                Ptr{Void}, #Bdense
                Ptr{Void}, #Zsparse
                Ptr{Void}, #Zdense
                Ptr{Void}, #R
                Ptr{Void}, #E
                Ptr{Void}, #H
                Ptr{Void}, #HPInv
                Ptr{Void}, #HTau
                Ptr{Void}), #cc
               ordering, #ordering
               tol, #tol
               1000000000, #econ
               0, #getCTX
               get(Aw.p),  # A
               C_NULL, #Bsparse
               C_NULL, #Bdense
               C_NULL, #Zsparse
               C_NULL, #Zdense
               ptr_R, #R
               ptr_E, #E
               C_NULL, #H
               C_NULL, #HPInv
               C_NULL, #HTau
               cc) #cc
    R = ptr_R[]
    E = ptr_E[]
    if E != C_NULL
        colprm = pointer_to_array(E, size(A,2), false) + 1
    else
        colprm = collect(1:size(A,2))
    end
    R1 = unsafe_load(R)
    if R1.stype != 0
        throw(ArgumentError("matrix has stype != 0. Convert to matrix with stype == 0 before converting to SparseMatrixCSC"))
    end
    maxrownum = 0
    for entryind = 1 : R1.nzmax
        maxrownum = max(maxrownum, unsafe_load(R1.i, entryind) + 1)
    end
    R_cvt = SparseMatrixCSC(maxrownum,
                            R1.ncol,
                            increment(pointer_to_array(R1.p, (R1.ncol + 1,), false)),
                            increment(pointer_to_array(R1.i, (R1.nzmax,), false)),
                            copy(pointer_to_array(R1.x, (R1.nzmax,), false)))
    free_sparse!(R)
    ccall((:cholmod_l_free, :libcholmod), Ptr{Void}, (Csize_t, Csize_t, Ptr{Void}, Ptr{Void}),
          size(A,2), sizeof(SuiteSparse_long), E, cc)
    R_cvt, colprm
end

## Obtain right null vector from squeezed R factor

function rightnullvec{Tv}(R::SparseMatrixCSC{Tv,Int},
                          colperm::Array{Int,1})
    m = size(R,1)
    n = size(R,2)
    if m >= n
        error("Right null vector cannot be computed if m>=n")
    end
    # find the first squeezed row
    squeezepos = m + 1
    for i = 1 : m
        if R[i,i] == 0.0
            squeezepos = i
            break
        end
    end
    nullvec = zeros(n)
    nullvec[squeezepos] = 1.0
    b = full(R[:,squeezepos])
    if norm(b[squeezepos:end]) > 0.0
        error("R must be upper triangular")
    end
    # solve for the column of the squeezed row
    # in terms of preceding columns using back
    # substitution.  For efficiency, work directly
    # on the fields of the sparse matrix R.
    for j = squeezepos - 1 : -1 : 1
        startp = R.colptr[j]
        endp = R.colptr[j+1] - 1
        if R.rowval[endp] != j
            error("R must be upper triangular")
        end
        coef = b[j] / R.nzval[endp]
        for pos = startp : endp
            i = R.rowval[pos]
            b[i] -= coef * R.nzval[pos]
        end
        nullvec[j] = -coef
    end
    nullvec_permute = zeros(n)
    nullvec_permute[colperm] = nullvec
    nullvec_permute
end




function cholfactLPs(A::SparseMatrixCSC{Float64, Int}; kws...)
    cm = defaults(common()) # setting the common struct to default values. Should only be done when creating new factorization.
    set_print_level(cm, 0) # no printing from CHOLMOD by default

    # Makes it an LLt
    unsafe_store!(common_final_ll, 1)
    F = fact_(Sparse(A), cm; kws...)
    s = unsafe_load(get(F.p))
    s.minor < size(A, 1) && return spzeros(0,0), Int[], false
    return sparse(F[:L]), get_perm(F), true
end

function forwsub_!(Lis, Ljs, Les, rhs)
    n = length(rhs)
    nnz = length(Lis)
    @assert minimum(Lis - Ljs) == 0
    pos = 1
    for j = 1 : n
        @assert Lis[pos] == j && Ljs[pos] == j
        rhs[j] /= Les[pos]
        pos += 1
        while pos <= nnz && Ljs[pos] == j
            rhs[Lis[pos]] -= rhs[j] * Les[pos]
            pos += 1
        end
    end
    nothing
end


function forwsub(L, rhs)
    x = rhs[:]
    is, js, es = findnz(L)
    forwsub_!(is, js, es, x)
    x
end


function backwsub_!(Lis, Ljs, Les, rhs)
    n = length(rhs)
    nnz = length(Lis)
    @assert minimum(Lis - Ljs) == 0
    pos = nnz
    for j = n : -1 : 1
        t = rhs[j]
        while pos > 0 && Ljs[pos] == j && Lis[pos] > j
            t -= Les[pos] * rhs[Lis[pos]]
            pos -= 1
        end
        @assert Lis[pos] == j && Ljs[pos] == j
        rhs[j] = t / Les[pos]
        pos -= 1
    end
    nothing
end


function cholsolve(L, prm, rhs)
    n = size(L,1)
    is, js, es = findnz(L)
    r2 = rhs[prm]
    forwsub_!(is, js, es, r2)
    backwsub_!(is, js, es, r2)
    sol = zeros(n)
    sol[prm] = r2
    sol
end







function cholfactPtL(A)
    n = size(A,1)
    F = cholfact((A + A') / 2)
    L0 = sparse(F[:L])
    is, js, es = findnz(L0)
    p = get_perm(F)
    sparse(p[is], p[js], es, n, n)
end



function test()
    A = [2.0  1.0  1.0 0.0
         1.0  1.0  0.5 0.0
         1.0  0.5  1.0 0.2
         0.0  0.0  0.2 1.5]
    As = sparse(A)
    F = cholfact(As)
    PtL = sparse(F[:PtL])
    println("norm of difference A - PtL * PtL' = ", norm(As - PtL * PtL',1), " (should be close to 0)")

    L, prm, success = cholfactLPs(As)
    @assert success
    println("prm = ", prm)
    println("L = ", L)
    b = [4.3,-2.2,8.8,7.1]
    x = cholsolve(L, prm, b)
    println("x = ", x, " norm(A*x - b) = ", norm(A * x - b), " (should be close to 0)")
    Ainit = [0.0   6.0  -3.1
             6.0   0.0   2.2
             6.3   0.0   0.0
             2.2   0.0   0.0]
    A = hcat(Ainit, Ainit * [2.0,2.1,2.2], Ainit * [3.0,3.1,3.2])
    As = sparse(A)
    R, colprm = qrfact_get_r_colperm(As, 1.0e-14)
    @assert size(R,1) == 3 && size(R,2) == 5
    v = rightnullvec(R,colprm)
    println("residual norm of right null vec = ", norm(As * v) / norm(v), " (should be close to 0)")
    nothing
end

end

function Base.rank(A::SparseMatrixCSC{Float64, Int}, tol::Float64=1.0e-14)
  R, colprm = MySparseExtensions.qrfact_get_r_colperm(A,tol)
  size(R,1)
end
