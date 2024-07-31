module DimensionalityReductionTechniques

import LinearAlgebra
import StatsAPI
import Distributions

export
  gaussian_random_projection,
  achiloptas_random_projection,
  sparse_random_projection

include("types.jl")
include("common.jl")
include("rp.jl")
include("pca.jl")

end
