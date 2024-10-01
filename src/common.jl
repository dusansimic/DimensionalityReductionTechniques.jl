centralize(x::AbstractVector, m::AbstractVector) = (isempty(m) ? x : x - m)
centralize(X::AbstractMatrix, m::AbstractVector) = (isempty(m) ? X : X .- m)

fullmean(d::Int, mv::AbstractVector{T}) where {T} =
  (isempty(mv) ? zeros(T, d) : mv)

preprocess_mean(X::AbstractMatrix{T}, m; dims=2) where {T<:Real} =
  (m === nothing ? vec(Statistics.mean(X, dims=dims)) : m == 0 ? T[] : m)


struct TransformDimension
  dimension::Int64
end

struct TransformIterationConfig
  max_iterations::Int64
  tolerance::Float64
end

function transformation(transform_dimension::TransformDimension, transform::Function, X)
  n_features = Tables.schema(X).names |> collect |> length
  dimension = transform_dimension.dimension
  if dimension <= 0 || dimension > n_features
    error("Reduced number of features cannot be less or equal to zero and larger than current number of features.")
  end

  matrix = transform(Tables.matrix(X), dimension)
  Tables.table(matrix)
end

function transformation(transform_dimension::TransformDimension, transform_iteration_config::TransformIterationConfig, transform::Function, X)
  n_features = Tables.schema(X).names |> collect |> length
  dimension = transform_dimension.dimension
  if dimension <= 0 || dimension > n_features
    error("Reduced number of features cannot be less or equal to zero and larger than current number of features.")
  end

  matrix = transform(Tables.matrix(X), dimension, transform_iteration_config.max_iterations, transform_iteration_config.tolerance)
  Tables.table(matrix)
end
