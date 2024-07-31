struct PCA{T<:Real} <: LinearDimensionalityReduction
  mean::AbstractVector{T}
  proj::AbstractMatrix{T}
  prinvars::AbstractVector{T}
  tprinvar::T
  tvar::T
end

function PCA(mean::AbstractVector{T}, proj::AbstractMatrix{T}, pvars::AbstractVector{T}, tvar::T) where {T<:Real}
  d, p = size(proj)
  if !(isempty(mean) || length(mean) == d)
    throw(DimensionMismatch("Dimensions of mean and projection matrices need to be compatible."))
  end
  if !(length(pvars) == p)
    throw(DimensionMismatch("Dimensions of projection matrix and principal variables need to be compatible."))
  end

  tpvar = sum(pvars)
  if !(tpvar <= tvar || isapprox(tpvar, tvar))
    throw(ArgumentError("principal variance cannot exceed total variance"))
  end

  PCA(mean, proj, pvars, tpvar, tvar)
end

size(M::PCA) = size(M.proj)

mean(M::PCA) = fullmean(LinearAlgebra.size(M.proj, 1), M.proj)

projection(M::PCA) = M.proj

eigenvecs(M::PCA) = projection(M)

principalvars(M::PCA) = M.prinvars

eigenvals(M::PCA) = principalvars(M)

tprincipalvar(M::PCA) = M.tprinvar

tresidualvar(M::PCA) = M.tvar - M.tprinvar

var(M::PCA) = M.tvar

r2(M::PCA) = M.tprinvar / M.tvar
const principalRatio = r2

predict(M::PCA, x::AbstractVecOrMat{T}) where {T<:Real} = transpose(M.proj) * centralize(x, M.mean)
