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
using JLD2

X = load_object("mat_X.jld2")
y = load_object("mat_y.jld2")
X = Tables.matrix(X)

train_idx, test_idx = splitobs(shuffle(collect(1:size(X, 1))), at=0.8)
y_train, y_test = y[train_idx], y[test_idx]
class_names = map(y -> string(y), 0:20)

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
  println(MulticlassFScore(beta=1, average=NoAvg(); levels=class_names)(y_pred, y_test))
  print("Accuracy: ")
  println(Accuracy()(y_pred, y_test))
  # print("Confusion matrix: ")
  # println(ConfusionMatrix(; levels=class_names)(y_pred, y_test))
  cm = ConfusionMatrix(; levels=class_names)(y_pred, y_test)
  println()
  println()
  println()

  cf_plot = heatmap(class_names, class_names, ConfusionMatrices.matrix(cm), c=:blues, xlabel="Predicted", ylabel="True", color=:coolwarm, left_margin=19mm, bottom_margin=7mm)
  savefig(cf_plot, "plots/Real math student perf. $method confusion matrix.png")
end


fit_pred_score("Pre-reduction", X, y_train, y_test)

features_grp = gaussian_random_projection(X, 25)
fit_pred_score("Gaussian random projection", features_grp, y_train, y_test)

features_ach = achiloptas_random_projection(X, 25)
fit_pred_score("Achiloptas random projection", features_ach, y_train, y_test)

features_spr = sparse_random_projection(X, 25)
fit_pred_score("Sparse random projection", features_spr, y_train, y_test)

features_pca = pca_projection(X, 25)
fit_pred_score("Principal component analysis", features_pca, y_train, y_test)

features_svd = svd_projection(X, 25)
fit_pred_score("Singular value decomposition", features_svd, y_train, y_test)

features_nmf = nmf_projection(X, 25)
fit_pred_score("Non-negative matrix factorization", features_nmf, y_train, y_test)

features_cms = cms_projection(X, 25)
fit_pred_score("Classical multidimensional scaling", features_cms, y_train, y_test)

features_mds = mds_projection(X, 25)
fit_pred_score("Multidimensional scaling", features_mds, y_train, y_test)
