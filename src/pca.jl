"""
    pca_projection(X::AbstractMatrix, k::Int64)

Return matrix X with reduced dimensions using PCA projection.

Covariance method with is usually recognized as a Karhunen-Loève transform
[jolliffe2002principal](@cite) of the matrix X.
"""
function pca_projection(X::AbstractMatrix, k::Int64)
  _, n = size(X)

  # Standardize data
  X_mean = Statistics.mean(X, dims=1)
  X_cent = X .- X_mean

  # Find the covariance matrix
  C = Statistics.cov(X_cent)

  # Find the eigenvectors and eigenvalues of the covariance matrix
  F = LinearAlgebra.eigen(C)
  V = F.vectors
  λ = F.values

  # Sort values to capture most variance
  sorted_λ_indices = sortperm(λ; rev=true)

  # Based on sorted values indices, get vectors that will give the most variance
  # and project the data
  proj = X_cent * V[:, sorted_λ_indices[1:k]]
  proj
end

mutable struct PCAProjection <: MLJBase.Unsupervised
  dimension::TransformDimension
end

function MLJBase.fit(transformer::PCAProjection, d::Int64)
  transformer.dimension = TransformDimension(d)
end

MLJBase.transform(transformer::PCAProjection, X) = transformation(transformer.dimension, pca_projection, X)
