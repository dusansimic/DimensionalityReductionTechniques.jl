module DimensionalityReductionTechniques

import LinearAlgebra
import StatsAPI
import Distributions
import Statistics
import Distances
import Optim
import MLJBase
import Tables

export
  gaussian_random_projection,
  achiloptas_random_projection,
  sparse_random_projection,
  pca_projection,
  svd_projection,
  nmf_projection,
  cms_projection,
  pcoa_projection,
  mds_projection,
  lle_projection

include("types.jl")
include("common.jl")
include("rp.jl")
include("pca.jl")
include("svd.jl")
include("nmf.jl")
include("cms.jl")
include("mds.jl")
include("lle.jl")

end
