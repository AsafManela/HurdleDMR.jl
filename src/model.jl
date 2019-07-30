"""
  Model(expr, parts::Vector{FormulaTerm})

Representation of a multipart model composed of a vector of FormulaTerm's.
It is inspired by the [https://github.com/matthieugomez/FixedEffectModels.jl](FixedEffectModels.jl) package
"""
struct Model
  expr
  parts::Vector{FormulaTerm}
end

"""
    @model(formula1, formula2, ...)

Parses a sequence of comma-separated FormulaTerm's representing a multipart model.
"""
macro model(args...)
  Expr(:call, :model_helper, (esc(Base.Meta.quot(a)) for a in args)...)
end

function model_helper(args...)
  expr = args
  parts = FormulaTerm[]
  for part in args
    push!(parts, @eval(@formula($(part.args[2]) ~ $(part.args[3]))))
  end
  Model(expr,parts)
end

function Base.show(io::IO, m::Model)
  print(io, "$(length(m.parts))-part model: [")
  notfirst = false
  for p in m.parts
    if notfirst
      print(io, ", ")
    end
    print(io, p)
    notfirst = true
  end
  print(io, "]")
  nothing
end

# function mergerhsterms(a::Tuple, b::Tuple)
#   newt = union(a.terms,b.terms)
#   ina = findall((in)(a.terms), terms)
#   inb = findall((in)(b.terms), terms)
#
#   newt, ina, inb
# end

function mergerhsterms(a, b)
  terms = union(a, b)

  ina = findall((in)(a), terms)
  inb = findall((in)(b), terms)

  (terms...,), ina, inb
end

# function mergerhsterms(a, b)
#   terms = union(a.terms,b.terms)
#   eterms = union(a.eterms,b.eterms)
#   factors = falses(length(eterms),length(terms))
#   is_non_redundant = falses(length(eterms),length(terms))
#   for t = 1:length(terms)
#     for s in [a,b]
#       it = something(findfirst(isequal(terms[t]), s.terms), 0)
#       if it > 0
#         for et = 1:length(eterms)
#           iet = something(findfirst(isequal(eterms[et]), s.terms), 0)
#           if iet > 0
#             factors[et,t] = s.factors[iet,it]
#             is_non_redundant[et,t] = s.is_non_redundant[iet,it]
#           end
#         end
#       end
#     end
#   end
#   order = vec(sum(factors, dims=1))
#   response = false
#   intercept = false
#
#   newt = terms(terms, eterms, factors, is_non_redundant, order, response, intercept)
#
#   ina = findall((in)(a.terms), terms)
#   inb = findall((in)(b.terms), terms)
#
#   newt, ina, inb
# end
#
"maps inzero and inpos to ModelMatrix columns (important with factor variables)"
function mapins(inzero, inpos, mm)
  mappedinzero = Int[]
  mappedinpos = Int[]
  for to=1:length(mm.assign)
    from = mm.assign[to]
    if from in inzero
      push!(mappedinzero, to)
    end
    if from in inpos
      push!(mappedinpos, to)
    end
  end
  mappedinzero, mappedinpos
end

"""
  getrhsterms(m, lhs)

Returns the right-hand-side Terms of the part of model `m` with left-hand-side
symbol `lhs`.
"""
function getrhsterms(m::Model, lhs::Symbol)
  ix = findfirst(p->p.lhs==term(lhs),m.parts)
  @assert ix > 0 "The model is missing a formula with $lhs on its left-hand-side."
  f = m.parts[ix]
  f.rhs
end

# Replicates functionality in StatsModels, so if it changes there it would have
# to change here too.
function createmodelmatrix(trms, df, counts, contrasts)
  # StatsModels.drop_intercept(T) && (trms.intercept = true)
  trms.intercept = true
  mf = ModelFrame(trms, df, contrasts=contrasts)
  # StatsModels.drop_intercept(T) && (mf.terms.intercept = false)
  mf.terms.intercept = false
  mm = ModelMatrix(mf)
  if !all(mf.nonmissing)
    counts = counts[mf.nonmissing,:]
  end
  mf, mm, counts
end

# struct NoTerm <: AbstractTerm end

function StatsModels.ModelFrame(f::TupleTerm, data;
                    model::Type{M}=StatisticalModel, contrasts=Dict{Symbol,Any}()) where M

    data = columntable(data)
    data, _ = missing_omit(data, termvars(f))

    sch = schema(f, data, contrasts)
    f = apply_schema(f, sch, M)

    ModelFrame(f, sch, data, model)
end

# function modelcols(as, cols)
# function modelcols()
# end
