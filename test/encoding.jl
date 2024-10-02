import Pkg
Pkg.activate(".")

using MLJ
using MLJModels
using DataFrames
using CSV
using ScientificTypes
using Tables
using JLD2

ContinuousEncoder = MLJModels.@load ContinuousEncoder pkg = MLJModels

df = CSV.File("data/student-mat.csv") |> DataFrame

select!(df, Not(:G1))
select!(df, Not(:G2))
X, y = unpack(df, !=(:G3), ==(:G3))

coerce!(X, autotype(X, (:string_to_multiclass, :discrete_to_continuous)))
y = map(ŷ -> string(ŷ), y)

encoder = ContinuousEncoder()
mach = machine(encoder, X) |> fit!
X = MLJModels.transform(mach, X)

save_object("mat_X.jld2", X)
save_object("mat_y.jld2", y)
