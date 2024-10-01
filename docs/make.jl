push!(LOAD_PATH, "../src/")

import Pkg
Pkg.activate("..")
using DimensionalityReductionTechniques

Pkg.activate(".")
using Documenter
using DocumenterCitations

About = "Introduction" => "index.md"
LinearMethods = "Linear methods" => [
  "linearmethods/randomprojection.md",
  "linearmethods/principal.md",
  "linearmethods/singular.md",
  "linearmethods/nonnegative.md",
  "linearmethods/classical.md",
]
NonlinearMethods = "Nonlinear methods" => [
  "nonlinearmethods/multidimensional.md",
]
Bibliography = "Bibliography" => "bibliography.md"

PAGES = [
  About,
  LinearMethods,
  NonlinearMethods,
  Bibliography,
]

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(
  modules=[DimensionalityReductionTechniques],
  sitename="DimensionalityReductionTechniques.jl",
  authors="Dušan Simić",
  remotes=nothing,
  checkdocs=:exports,
  pages=PAGES,
  plugins=[bib],
)
