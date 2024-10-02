import Pkg

Pkg.activate("..")
push!(LOAD_PATH, "../src/")
using DimensionalityReductionTechniques

Pkg.activate(".")
using DecisionTree
using DataFrames
using MLDataUtils
using Random
using StatisticalMeasures
using Plots

gr()

using Plots.PlotMeasures
using CSV
using Tables

data = CSV.File("data/student_performance.csv") |> Tables.matrix

features = data[:, begin:end-1]
labels = data[:, end]
real_to_str_dict = Dict(0.0 => "A", 1.0 => "B", 2.0 => "C", 3.0 => "D", 4.0 => "F")
labels = map(x -> real_to_str_dict[x], labels)

train_idx, test_idx = splitobs(shuffle(collect(1:size(features, 1))), at=0.8)
y_train, y_test = labels[train_idx], labels[test_idx]
class_names = ["A", "B", "C", "D", "F"]


function fit_pred_score(method::String, features::AbstractMatrix, y_train::AbstractVector, y_test::AbstractVector)
  X_train, X_test = features[train_idx, :], features[test_idx, :]

  model = build_tree(y_train, X_train)
  model = prune_tree(model, 0.9)
  y_pred = apply_tree(model, X_test)

  # cf = DecisionTree.confusion_matrix(y_test, y_pred)

  println('='^length(method))
  println(method)
  println('='^length(method))
  # println(cf)
  print("F1: ")
  println(MulticlassFScore(beta=1, average=NoAvg())(y_pred, y_test))
  print("Accuracy: ")
  println(Accuracy()(y_pred, y_test))
  # print("Confusion matrix: ")
  # println(ConfusionMatrix()(y_pred, y_test))
  println()
  println()
  println()

  # cf_plot = heatmap(class_names, class_names, cf.matrix, c=:blues, xlabel="Predicted", ylabel="True", color=:coolwarm, left_margin=19mm, bottom_margin=7mm)
  # savefig(cf_plot, "plots/Student perf. $method confusion matrix.png")
end

fit_pred_score("Pre-reduction", features, y_train, y_test)

function run_tests(dim::Int64)
  println('='^20)
  println()
  print(' '^18)
  println(dim)
  println()
  println('='^20)

  features_grp = gaussian_random_projection(features, dim)
  fit_pred_score("Gaussian random projection dim $dim", features_grp, y_train, y_test)

  features_ach = achiloptas_random_projection(features, dim)
  fit_pred_score("Achiloptas random projection dim $dim", features_ach, y_train, y_test)

  features_spr = sparse_random_projection(features, dim)
  fit_pred_score("Sparse random projection dim $dim", features_spr, y_train, y_test)

  features_pca = pca_projection(features, dim)
  fit_pred_score("Principal component analysis dim $dim", features_pca, y_train, y_test)

  features_svd = svd_projection(features, dim)
  fit_pred_score("Singular value decomposition dim $dim", features_svd, y_train, y_test)

  features_nmf = nmf_projection(features, dim)
  fit_pred_score("Non-negative matrix factorization dim $dim", features_nmf, y_train, y_test)

  features_cms = cms_projection(features, dim)
  fit_pred_score("Classical multidimensional scaling dim $dim", features_cms, y_train, y_test)

  # features_mds = mds_projection(features, dim)
  # fit_pred_score("Multidimensional scaling dim $dim", features_mds, y_train, y_test)
end

for dim in 2:size(features, 2)-1
  run_tests(dim)
end


