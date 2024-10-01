"""
    nmf_projection(X::AbstractMatrix, k::Int64; max_iter::Int64=500, tol::Float64=1e-4)

Return matrix X with reduced dimensions using NMF projection.
"""
function nmf_projection(X::AbstractMatrix, k::Int64; max_iter::Int64=500, tol::Float64=1e-4)
  m, n = size(X)
  W = rand(m, k)
  H = rand(k, n)

  for _ in 1:max_iter
    numerator_H = W' * X
    denominator_H = (W' * W) * H .+ 1e-10
    H .= H .* (numerator_H ./ denominator_H)

    numerator_W = X * H'
    denominator_W = W * (H * H') .+ 1e-10
    W .= W .* (numerator_W ./ denominator_W)

    # Calculate Frobenius norm
    err = LinearAlgebra.norm(X - W * H, 2)

    if err < tol
      break
    end
  end

  W
end

mutable struct NMFProjection <: MLJBase.Unsupervised
  dimension::TransformDimension
  iteration_config::TransformIterationConfig
end

function MLJBase.fit(transformer::NMFProjection, d::Int64; max_iter::Int64=500, tol::Int64=1e-4)
  transformer.dimension = TransformDimension(d)
  transformer.iteration_config = TransformIterationConfig(max_iter, tol)
end

MLJBase.transform(transformer::NMFProjection, X) = transformation(transformer.dimension, transformer.iteration_config, nmf_projection, X)
