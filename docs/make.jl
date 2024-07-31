push!(LOAD_PATH, "../src/")

using Documenter
using DocumenterCitations
using DimensionalityReductionTechniques

About = "Introduction" => "index.md"
LinearMethods = "Linear methods" => [
  "linearmethods/randomprojection.md",
]
Bibliography = "Bibliography" => "bibliography.md"

PAGES = [
  About,
  LinearMethods,
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
