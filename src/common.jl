centralize(x::AbstractVector, m::AbstractVector) = (isempty(m) ? x : x - m)
centralize(X::AbstractMatrix, m::AbstractVector) = (isempty(m) ? X : X .- m)

fullmean(d::Int, mv::AbstractVector{T}) where {T} =
  (isempty(mv) ? zeros(T, d) : mv)

preprocess_mean(X::AbstractMatrix{T}, m; dims=2) where {T<:Real} =
  (m === nothing ? vec(Statistics.mean(X, dims=dims)) : m == 0 ? T[] : m)
