#ifndef TWIGY_CORE_SPLITTER_H_
#define TWIGY_CORE_SPLITTER_H_

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include "core/criterion.h"
#include "core/utils.h"

struct Split {
  std::size_t feature;
  double threshold;
  int pos;
  double impurity_left;
  double impurity_right;
  std::vector<int> left_value;
  std::vector<int> right_value;
  bool found_split;

  void Print();
};

class Splitter {
 public:
  static constexpr double kMinSplitDiff_ = 1e-8;
  const std::vector<std::vector<double>> *feature_data_;
  const std::vector<int> *label_data_;
  Criterion criterion_;
  int start_;
  int end_;
  int min_samples_leaf_;
  std::size_t max_features_;
  std::mt19937 *gen_;
  std::vector<int> feature_order_;
  bool shuffle_features_;

  std::vector<SampleData> sample_map_;
  std::size_t n_samples_total_;
  std::size_t n_features_;
  Splitter() {}
  Splitter(const std::vector<std::vector<double>> *feature_data,
           const std::vector<int> *label_data, int min_samples_leaf,
           ImpurityMeasure impurity_measure, std::size_t max_features,
           int n_labels, std::mt19937 *gen,
           const std::vector<int> &samples_subset = {});
  void ResetSampleRange(int start, int end);
  void SplitNode(Split &split);
};

#endif  // TWIGY_CORE_SPLITTER_H_
