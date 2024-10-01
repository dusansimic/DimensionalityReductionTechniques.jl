"""
    mds_projection(X::AbstractMatrix, k::Int64; max_iter::Int64=500, tol::Float64=1e-4)

Return matrix X with reduced dimensions using Multidimensional Scaling
projection.
"""
function mds_projection(X::AbstractMatrix, k::Int64; max_iter::Int64=500, tol::Float64=1e-4)
  function __stress(X_low_flat::AbstractVector, D_high::AbstractMatrix, m::Int64, k::Int64)
    X_low = reshape(X_low_flat, m, k)
    # TODO: Should X_high be used here or X_low?
    D_low = Distances.pairwise(Distances.Euclidean(), X_low, dims=1)

    numerator = 0.0
    denominator = 0.0

    for i ∈ 1:m
      for j ∈ (i+1):m
        numerator += (D_high[i, j] - D_low[i, j])^2
        denominator += D_high[i, j]^2
      end
    end

    sqrt(numerator / denominator)
  end
  D = Distances.pairwise(Distances.Euclidean(), X, dims=1)
  m, _ = size(X)

  X_low = randn(m, k)

  result = Optim.optimize(
    X_low_flat -> __stress(X_low_flat, D, m, k),
    vec(X_low),
    Optim.BFGS(),
    Optim.Options(iterations=max_iter, g_tol=tol, show_trace=true)
  )

  X_reduced = reshape(result.minimizer, m, k)
  X_reduced
end