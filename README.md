# DimensionalityReductionTechniques.jl

This library is intended to add various linear and non-linear dimensionality reduction techniques
that are not currently implemented in other libraries.

Most notable library that has some techniques implemented is
[MultivariateStats.jl](https://docs.juliahub.com/General/MultivariateStats/stable/) however it is
missing some techniques which I've been introduced to during a machine learning course at Faculty
of Sciences, University of Novi Sad.

The whole library is a part of a final project for said course.

## Build docs

Documentation is created using Documenter and also requires DocumenterCitations since there are some
cited reference in the docs. To build it run the `make.jl` file from `docs` directory with the
`--project` option.

```sh
cd docs
julia --project make.jl
```

## License

[BSD 2-clause](./LICENSE)
