"""
    svd_projection(X::AbstractMatrix, k::Int64)

Return matrix X with reduced dimensions using SVD projection.

The Singular Value Decomposition is done using the standard LinearAlgebra
package and then data is projected from the U factor and singluar values.
"""
function svd_projection(X::AbstractMatrix, k::Int64)
  F = LinearAlgebra.svd(X)
  proj = F.U[:, 1:k] .* F.S[1:k]'
  proj
end

mutable struct SVDProjection <: MLJBase.Unsupervised
  dimension::Int64
end

function MLJBase.fit(transformer::SVDProjection, d::Int64)
  transformer.dimension = d
end

function MLJBase.transform(transformer::SVDProjection, X)
  n_features = Tables.schema(X).names |> collect |> length
  dimension = transformer.dimension
  if dimension == 0 || dimension > n_features
    error("Reduced number of features cannot be zero or larger than current number of features.")
  end

  matrix = svd_projection(Tables.matrix(X), dimension)
  Tables.table(matrix)
end
