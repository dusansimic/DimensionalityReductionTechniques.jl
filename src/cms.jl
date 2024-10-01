"""
    cms_projection(X::AbstractMatrix, k::Int64)

Return matrix X with reduced dimensions using Classical Multidimensional Scaling
projection.
"""
function cms_projection(X::AbstractMatrix, k::Int64)
  n, _ = size(X)
  D = Distances.pairwise(Distances.Euclidean(), X; dims=1)
  H = LinearAlgebra.I(n) .- (1 / n) * ones(n, n)
  D_squared = D .^ 2
  B = -0.5 * H * D_squared * H

  λ, V = LinearAlgebra.eigen(B)
  λ, V = real(λ), real(V)

  sorted_λ = sortperm(λ, rev=true)

  V[:, sorted_λ[1:k]] * LinearAlgebra.Diagonal(sqrt.(λ[sorted_λ[1:k]]))
end

pcoa_projection = cms_projection

mutable struct CMSProjection <: MLJBase.Unsupervised
  dimension::TransformDimension
end

function MLJBase.fit(transformer::CMSProjection, d::Int64)
  transformer.dimension = TransformDimension(d)
end

MLJBase.transform(transformer::CMSProjection, X) = transformation(transformer.dimension, cms_projection, X)
