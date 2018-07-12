rtol=0.05
rdist(x::Number,y::Number) = abs(x-y)/max(abs(x),abs(y))
rdist(x::AbstractArray{T}, y::AbstractArray{S}; norm::Function=vecnorm) where {T<:Number,S<:Number} = norm(x - y) / max(norm(x), norm(y))

testdir = dirname(@__FILE__)
