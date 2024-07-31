centralize(x::AbstractVector, m::AbstractVector) = (isempty(m) ? x : x - m)
centralize(X::AbstractMatrix, m::AbstractVector) = (isempty(m) ? X : X .- m)

fullmean(d::Int, mv::AbstractVector{T}) where {T} = (isempty(mv) ? zeros(T, d) : mv)