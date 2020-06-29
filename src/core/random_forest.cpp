#include "random_forest.h"

RandomForestClassifier::RandomForestClassifier(
    int n_estimators, ImpurityMeasure impurity_measure, int max_depth,
    int min_samples_split, int min_samples_leaf, int max_features,
    MaxFeaturesMethod max_features_method, double min_impurity_split,
    int n_threads, int max_samples) {
  n_estimators_ = n_estimators;
  if (n_estimators_ < 1) {
    throw std::out_of_range(
        "Number of estimators n_estimators must be greater than 0.");
  }
  impurity_measure_ = impurity_measure;
  min_samples_leaf_ = min_samples_leaf;
  max_features_ = max_features;
  max_depth_ = max_depth;
  min_samples_split_ = min_samples_split;
  max_samples_ = max_samples;
  n_threads_ = n_threads;
  max_features_method_ = max_features_method;
  min_impurity_split_ = min_impurity_split;
}

void RandomForestClassifier::BuildForest(
    const std::vector<std::vector<double>> feature_data,
    const std::vector<int> label_data) {
  feature_data_ = feature_data;
  label_data_ = label_data;
  n_samples_ = feature_data_.size();

  CheckNegativeLabels(&label_data_);

  if (max_samples_ <= 0) {
    max_samples_ = n_samples_;
  }

  SetMaxFeatures(max_features_, max_features_method_, &feature_data_);

  n_labels_ = *std::max_element(label_data_.begin(), label_data_.end()) + 1;

  std::vector<int> subset_sample;
  std::mt19937 gen;
  gen.seed(42);

  boost::asio::thread_pool pool(n_threads_);

  for (int i = 0; i < n_estimators_; i++) {
    DecisionTreeClassifier new_tree = DecisionTreeClassifier(
        impurity_measure_, max_depth_, min_samples_split_, min_samples_leaf_,
        max_features_, max_features_method_, min_impurity_split_);
    trees_.push_back(new_tree);
  }

  for (std::size_t i = 0; i < trees_.size(); i++) {
    subset_sample =
        GenerateRandomIntVector(0, n_samples_ - 1, max_samples_, gen);
    boost::asio::post(
        pool, boost::bind(&DecisionTreeClassifier::BuildTree, &(trees_[i]),
                          &feature_data, &label_data, subset_sample, n_labels_,
                          &gen));
  }
  pool.join();
  is_built_ = true;
}

std::vector<int> RandomForestClassifier::PredictClasses(
    const std::vector<std::vector<double>> *feature_data_) {
  if (!is_built_) {
    throw std::logic_error(
        "Random forest must be built with BuildForest method before "
        "predictions can be made.");
  }

  std::size_t n_samples_inference = (*feature_data_).size();
  std::vector<std::vector<double>> accumulated_probabilities;
  std::vector<int> predicted_classes;
  std::vector<double> class_probas;
  accumulated_probabilities.reserve(n_samples_inference);
  predicted_classes.reserve(n_samples_inference);
  class_probas.reserve(n_labels_);

  for (std::size_t i = 0; i < n_samples_inference; i++) {
    class_probas.clear();
    for (int k = 0; k < n_labels_; k++) {
      class_probas.emplace_back(0.0);
    }
    accumulated_probabilities.push_back(class_probas);
  }

  std::vector<std::vector<double>> current_estimator_probs;
  for (int i = 0; i < n_estimators_; i++) {
    current_estimator_probs =
        trees_[i].tree_.PredictClassProbabilities(feature_data_);
    for (std::size_t j = 0; j < n_samples_inference; j++) {
      for (int k = 0; k < n_labels_; k++) {
        accumulated_probabilities[j][k] += current_estimator_probs[j][k];
      }
    }
  }

  for (std::size_t j = 0; j < n_samples_inference; j++) {
    for (int k = 0; k < n_labels_; k++) {
      accumulated_probabilities[j][k] /= n_estimators_;
    }
    predicted_classes.push_back(ArgMax(accumulated_probabilities[j]));
  }
  return predicted_classes;
}
