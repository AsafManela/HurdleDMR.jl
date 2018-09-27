rtol=0.05
rdist(x::Number,y::Number) = abs(x-y)/max(abs(x),abs(y))
rdist(x::AbstractArray{T}, y::AbstractArray{S}) where {T<:Number,S<:Number} = norm(x - y) / max(norm(x), norm(y))

testdir = dirname(@__FILE__)

"Tests whether the output of `show(expr)` converted to String equals `expected`"
macro test_show(expr, expected)
    s = quote
        local o = IOBuffer()
        show(o, $expr)
        String(take!(o))
    end
    :(@test $(esc(s)) == $expected)
end
