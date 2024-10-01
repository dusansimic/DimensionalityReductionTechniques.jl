# Random projection

This technique was inspired by by Johnson-Lindenstrauss lemma [johnson1986extensions](@cite).

```@docs
gaussian_random_projection(X::AbstractMatrix, k::Int64)
```

```@docs
achiloptas_random_projection(X::AbstractMatrix, k::Int64)
```

```@docs
sparse_random_projection(X::AbstractMatrix, k::Int64, density::Union{Float64,Nothing}=nothing)
```
