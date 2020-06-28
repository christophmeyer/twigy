#include <vector>

#include "core/decision_tree.h"
#include "core/random_forest.h"
#include "core/utils.h"

int main() {
  // Read the example data and split it into features and labels
  table data_frame = ReadCsv("../test/testdata/titanic_data.csv");
  std::vector<int> y;
  std::vector<std::vector<double>> X;
  Split_X_y(data_frame, X, y, "Survived");

  // preview of the data
  data_frame.Print(10);

  // parameters for the tree and forest classifiers
  ImpurityMeasure impurity_measure = gini;
  int min_samples_leaf = 1;
  int max_depth = 5;
  int min_samples_split = 2;
  int max_features = 8;
  MaxFeaturesMethod max_features_method = all_method;
  double min_impurity_split = 0.0;

  // train and print a decision tree on the training data
  DecisionTreeClassifier tree_classifier = DecisionTreeClassifier(
      impurity_measure, max_depth, min_samples_split, min_samples_leaf,
      max_features, max_features_method, min_impurity_split);
  tree_classifier.BuildTree(X, y);
  tree_classifier.PrintTree();

  // predict classes of training data and print for first ten samples.
  std::vector<int> predicted_classes;
  predicted_classes = tree_classifier.tree_.PredictClasses(&X);

  for (std::size_t j = 0; j < 10; j++) {
    std::cout << predicted_classes[j] << " ";
  }
  std::cout << std::endl;

  // additional parameters for the random forest
  int n_estimators = 100;
  int max_samples = -1;
  int n_threads = 1;
  max_features_method = log2_method;

  // train random forest on training data
  RandomForestClassifier random_forest_clf = RandomForestClassifier(
      n_estimators, impurity_measure, max_depth, min_samples_split,
      min_samples_leaf, max_features, max_features_method, min_impurity_split,
      n_threads, max_samples);

  random_forest_clf.BuildForest(X, y);

  // predict classes of training data and print for first ten samples.
  std::vector<int> predicted_classes_rf;
  predicted_classes_rf = random_forest_clf.PredictClasses(&X);

  for (std::size_t j = 0; j < 10; j++) {
    std::cout << predicted_classes_rf[j] << " ";
  }
  std::cout << std::endl;
}
