"""
    gaussian_random_projection(X::AbstractMatrix, k::Int64)

Return matrix X with reduced dimensions using Gaussian random projection.

For the GRP, the projection matrix ``R_{k \\times d}`` where ``d`` is current vector length in
matrix ``X`` and ``k`` is the target vector length is generated using simple random number generator
and then the produced matrix (list of vectors really) is orthogonalized by computing the QR
factorization. From the resulting matrix, only ``k`` columns are collected so the resulting matrix
is the projection matrix used for the dimensionality reduction.
"""
function gaussian_random_projection(X::AbstractMatrix, k::Int64)
  _, dX = size(X)

  if k > dX
    throw(ArgumentError("target dimension 'k' cannot be larger than current dataset dimension"))
  end

  R = rand(dX, dX)
  R = LinearAlgebra.qr(R).Q
  R = R[:, 1:k]

  X * R
end

function __sparse_matrix_random_projection(d::Distributions.DiscreteNonParametric, X::AbstractMatrix, k::Int64)
  _, dX = size(X)

  if k > dX
    throw(ArgumentError("target dimension 'k' cannot be larger than current dataset dimension"))
  end

  R = rand(d, dX, k)

  X * R
end

"""
    achiloptas_random_projection(X::AbstractMatrix, k::Int64)

Return matrix X with reduced dimensions using the Achiloptas [achlioptas2001database](@cite) sparse
matrix for random projection.

The projection matrix for this method is generated using a very simple and computationally efficient
categorical distribution.

```math
R_{i,j} = \\sqrt{3} \\times
\\begin{cases}
  +1 & & \\frac{1}{6} \\\\
  0 & \\text{with probability } & \\frac{2}{3} \\\\
  -1 & & \\frac{1}{6}
\\end{cases}
```
"""
achiloptas_random_projection(X::AbstractMatrix, k::Int64) = __sparse_matrix_random_projection(
  Distributions.DiscreteNonParametric([-sqrt(3), 0, sqrt(3)], [1 / 6, 1 - 1 / 3, 1 / 6]),
  X,
  k,
)

"""
    sparse_random_projection(X::AbstractMatrix, k::Int64, density::Union{Float64,Nothing}=nothing)

Return matrix X with reduced dimensions using the distribution described on sparse random projection
documentation page of SciKit learn framework [scikit2024sparse](@cite).

The random values for sparse matrix are drawn from the distribution

```math
R_{i,j} = \\sqrt{\\frac{s}{k}} \\times
\\begin{cases}
  +1 & & \\frac{1}{2s} \\\\
  0 & \\text{with probability } & 1 - \\frac{1}{s} \\\\
  -1 & &\\frac{1}{2s}
\\end{cases}
```

where ``k`` is the vector length in the projected matrix and ``s`` is density of non-zero values in
the projection matrix.

By default, density is the minimum recommended density by Ping Li et al. [li2006very](@cite),
``\\frac{1}{d}`` where ``d`` is the number of features in matrix ``X``.
"""
function sparse_random_projection(X::AbstractMatrix, k::Int64, density::Union{Float64,Nothing}=nothing)
  _, dX = size(X)

  if isnothing(density)
    density = 1 / sqrt(dX)
  end
  s = 1 / density


  d = Distributions.DiscreteNonParametric([-sqrt(s / k), 0, sqrt(s / k)], [1 / (2 * s), 1 - 1 / s, 1 / (2 * s)])

  __sparse_matrix_random_projection(d, X, k)
end