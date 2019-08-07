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

function mergerhsterms(a, b)
  terms = union(a, b)

  ina = findall((in)(a), terms)
  inb = findall((in)(b), terms)

  (terms...,), ina, inb
end

"maps inzero and inpos to ModelMatrix columns (important with factor variables)"
function mapins(inzero, inpos, appliedschema)
  mappedinzero = Int[]
  mappedinpos = Int[]
  from = 1
  to = 0
  for t=1:length(appliedschema)
    to = from + width(appliedschema[t]) - 1
    if t in inzero
      [push!(mappedinzero, i) for i in from:to]
    end
    if t in inpos
      [push!(mappedinpos, i) for i in from:to]
    end
    from = to + 1
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

StatsModels.missing_omit(data::T, formula::TupleTerm) where T<:ColumnTable =
    missing_omit(NamedTuple{tuple(termvars(formula)...)}(data))

function StatsModels.modelcols(trms::TupleTerm, data, counts::AbstractMatrix;
    model::Type{M}=DCR, contrasts=Dict{Symbol,Any}()) where M

  cols = columntable(data)
  cols, nonmissing = missing_omit(cols, trms)
  s = schema(trms, cols, contrasts)
  as = apply_schema(trms, s, M)
  modelcols(as, nonmissing, cols, counts)
end

function StatsModels.modelcols(as::TupleTerm, nonmissing, cols, counts::AbstractMatrix)

  covars = modelcols(MatrixTerm(as), cols)

  if !all(nonmissing)
    counts = counts[nonmissing,:]
  end

  covars, counts, as
end

"Similar to StatsModels.TableRegressionModel, but also holds a counts matrix and Model"
struct TableCountsRegressionModel{M,D,C} <: RegressionModel
  model::M    # actual fitted model (result of fit())
  data::D     # data table used
  counts::C   # counts matrix used
  f::Model    # @model specification
  schema      # applied schema
  sprojdir::Union{Nothing,Symbol}   # lhs variable in forward regression
end
TableCountsRegressionModel(model, data, counts, f, schema) = TableCountsRegressionModel(model, data, counts, f, schema, nothing)

function Base.show(io::IO, model::TableCountsRegressionModel)
    println(io, typeof(model))
    println(io)
    println(io, model.f)
end
