# Model here is inspired by matthieugomez/FixedEffectModels.jl package
struct Model
  expr
  parts::Vector{Formula}
end

"@model(formulas...) takes a sequence of formulas and parses them in a n-part model"
macro model(args...)
  Expr(:call, :model_helper, (esc(Base.Meta.quot(a)) for a in args)...)
end

function model_helper(args...)
  expr = args
  parts = Formula[]
  for part in args
    push!(parts, @eval(@formula($(part.args[2]) ~ $(part.args[3]))))
  end
  Model(expr,parts)
end

function Base.show(io::IO, m::Model)
  println(io, "$(length(m.parts))-part model:")
  println.(io, m.parts)
end

function mergerhsterms(a::StatsModels.Terms, b::StatsModels.Terms)
  terms = union(a.terms,b.terms)
  eterms = union(a.eterms,b.eterms)
  factors = falses(length(eterms),length(terms))
  is_non_redundant = falses(length(eterms),length(terms))
  for t = 1:length(terms)
    for s in [a,b]
      it = findfirst(s.terms,terms[t])
      if it > 0
        for et = 1:length(eterms)
          iet = findfirst(s.terms,eterms[et])
          if iet > 0
            factors[et,t] = s.factors[iet,it]
            is_non_redundant[et,t] = s.is_non_redundant[iet,it]
          end
        end
      end
    end
  end
  order = vec(sum(factors,1))
  response = false
  intercept = false

  newt = StatsModels.Terms(terms, eterms, factors, is_non_redundant, order, response, intercept)

  ina = findin(terms,a.terms)
  inb = findin(terms,b.terms)

  newt, ina, inb
end

function getpart(m::Model, lhs::Symbol, clearlhs=true)
  ix = findfirst(p->p.lhs==lhs,m.parts)
  @assert ix > 0 "The model is missing a formula with $lhs on its left-hand-side."
  f = m.parts[ix]
  if clearlhs
    f = copy(f)
    f.lhs = nothing
  end
  f
end

function getformula(f::Formula, lhs::Symbol, clearlhs=true)
  @assert f.lhs == lhs "The model is missing a formula with $lhs on its left-hand-side."
  if clearlhs
    f = copy(f)
    f.lhs = nothing
  end
  f
end

function getformula(m::Model, lhs::Symbol, clearlhs=true)
  ix = findfirst(p->p.lhs==lhs,m.parts)
  @assert ix > 0 "The model is missing a formula with $lhs on its left-hand-side."
  f = m.parts[ix]
  if clearlhs
    f = copy(f)
    f.lhs = nothing
  end
  f
end

function getrhsterms(args...)
  f = getformula(args...)
  StatsModels.Terms(f)
end

function createmodelmatrix(trms, df, contrasts)
  # StatsModels.drop_intercept(T) && (trms.intercept = true)
  trms.intercept = true
  mf = ModelFrame(trms, df, contrasts=contrasts)
  # StatsModels.drop_intercept(T) && (mf.terms.intercept = false)
  mf.terms.intercept = false
  mm = ModelMatrix(mf)
  mf, mm
end
