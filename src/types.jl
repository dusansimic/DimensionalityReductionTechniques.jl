abstract type AbstractDimensionalityReduction <: StatsAPI.RegressionModel end

function size_in(model::AbstractDimensionalityReduction)
  return size_in(model)
end

function size_out(model::AbstractDimensionalityReduction)
  return size_out(model)
end

projection(model::AbstractDimensionalityReduction) = error("'projection' is not defined for $(typeof(model)).")

reconstruct(model::AbstractDimensionalityReduction, y) = error("'reconstruct' is not defined for $(typeof(model)).")

abstract type LinearDimensionalityReduction <: AbstractDimensionalityReduction end

loadings(model::LinearDimensionalityReduction) = error("'loadings' is not defined for $(typeof(model)).")

abstract type NonlinearDimensionalityReduction <: AbstractDimensionalityReduction end
