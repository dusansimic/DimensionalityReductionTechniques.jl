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
  dimension::TransformDimension
end

function MLJBase.fit(transformer::SVDProjection, d::Int64)
  transformer.dimension = TransformDimension(d)
end

MLJBase.transform(transformer::SVDProjection, X) = transformation(transformer.dimension, svd_projection, X)
