#ifndef TWIGY_CORE_RANDOM_FOREST_H_
#define TWIGY_CORE_RANDOM_FOREST_H_

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/bind.hpp>
#include <iostream>
#include <limits>
#include <vector>

#include "core/decision_tree.h"

class RandomForestClassifier {
 public:
  int n_estimators_;
  std::vector<std::vector<double>> feature_data_;
  std::vector<int> label_data_;
  ImpurityMeasure impurity_measure_;
  int min_samples_leaf_;
  int max_depth_;
  int min_samples_split_;
  int n_labels_;
  int max_samples_;
  int n_threads_;
  std::size_t max_features_;
  MaxFeaturesMethod max_features_method_;
  double min_impurity_split_;
  bool is_built_ = false;

  int n_samples_;
  std::vector<DecisionTreeClassifier> trees_;

  RandomForestClassifier(int n_estimators = 100,
                         ImpurityMeasure impurity_measure = gini,
                         int max_depth = std::numeric_limits<int>::max(),
                         int min_samples_split = 2, int min_samples_leaf = 1,
                         int max_features = -1,
                         MaxFeaturesMethod max_features_method = sqrt_method,
                         double min_impurity_split = 0.0, int n_threads = 1,
                         int max_samples = -1);

  void BuildForest(const std::vector<std::vector<double>> feature_data,
                   const std::vector<int> label_data);
  std::vector<int> PredictClasses(
      const std::vector<std::vector<double>> *feature_data_);
};

#endif  // TWIGY_CORE_RANDOM_FOREST_H_
