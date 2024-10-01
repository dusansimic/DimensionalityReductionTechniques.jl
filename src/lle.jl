# TODO
function lle_projection(X::AbstractMatrix, k::Int64, d::Int64)
  n, _ = size(X)

  D = Distances.pairwise(Distances.Euclidean(), X, dims=2)
  neighbors = [partialsortperm(D[i, :], 1:k+1)[2:end] for i in 1:n]

  W = zeros(Float64, n, n)

  for i in 1:n
    Z = X[neighbors[i], :] .- X[i, :]
    C = Z * Z'
    C += 1e-3 * LinearAlgebra.I(k)
    w = C \ ones(k)
    W[i, neighbors[i]] = w / sum(w)
  end

  M = (LinearAlgebra.I(n) .- W)' * (LinearAlgebra.I(n) .- W)

  λ, V = LinearAlgebra.eigen(M)
  sorted_λ = sortperm(λ)
  X_reduced = V[:, sorted_λ[2:d+1]]

  X_reduced
end