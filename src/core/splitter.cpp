#include "core/splitter.h"

#include <limits>
#include <vector>

#include "core/utils.h"

Splitter::Splitter(const std::vector<std::vector<double>> *feature_data,
                   const std::vector<int> *label_data, int min_samples_leaf,
                   ImpurityMeasure impurity_measure, std::size_t max_features,
                   int n_labels, std::mt19937 *gen,
                   const std::vector<int> &samples_subset) {
  feature_data_ = feature_data;
  label_data_ = label_data;
  n_features_ = (*feature_data)[0].size();
  min_samples_leaf_ = min_samples_leaf;
  max_features_ = max_features;
  gen_ = gen;

  shuffle_features_ = false;
  if (max_features_ < (*feature_data)[0].size()) {
    shuffle_features_ = true;
  }

  if (samples_subset.size() == 0) {
    n_samples_total_ = (*feature_data).size();
    sample_map_.reserve(n_samples_total_);
    for (std::size_t i = 0; i < n_samples_total_; i++) {
      SampleData sample;
      sample.current_feature_value = 0.0;
      sample.sample_number = i;
      sample_map_.push_back(sample);
    }
  } else {
    n_samples_total_ = samples_subset.size();
    sample_map_.reserve(n_samples_total_);
    for (std::size_t i = 0; i < n_samples_total_; i++) {
      SampleData sample;
      sample.current_feature_value = 0.0;
      sample.sample_number = samples_subset[i];
      sample_map_.push_back(sample);
    }
  }
  criterion_ =
      Criterion(impurity_measure, n_labels, feature_data_, label_data_);

  for (std::size_t i = 0; i < n_features_; i++) {
    feature_order_.emplace_back(i);
  }
}

void Splitter::ResetSampleRange(int start, int end) {
  // start_ inclusive and end_ exclusive
  start_ = start;
  end_ = end;
  criterion_.SetNodeLimits(start_, end_);
}

void Split::Print() {
  std::cout << "Feature: " << feature << std::endl;
  std::cout << "Threshold: " << threshold << std::endl;
  std::cout << "Split position: " << pos << std::endl;
  std::cout << "Impurity left: " << impurity_left << std::endl;
  std::cout << "Impurity right: " << impurity_right << std::endl;
}

void Splitter::SplitNode(Split &split) {
  Split current_split;
  Split best_split;
  double current_impurity_improvement =
      -std::numeric_limits<double>::infinity();
  double best_impurity_improvement = -std::numeric_limits<double>::infinity();
  std::size_t current_feature_num = 0;
  std::size_t current_feature = 0;
  current_split.found_split = false;

  if (shuffle_features_) {
    std::shuffle(feature_order_.begin(), feature_order_.end(), *gen_);
  }

  while (current_feature_num < max_features_) {
    current_feature = feature_order_[current_feature_num];
    current_split.feature = current_feature;
    // set current feature values for the samples in the current node
    for (int i = start_; i < end_; i++) {
      sample_map_[i].current_feature_value =
          (*feature_data_)[sample_map_[i].sample_number][current_feature];
    }
    criterion_.ResetStats();

    // sort sample_map by current feature values
    std::sort(sample_map_.begin() + start_, sample_map_.begin() + end_,
              [](SampleData const &a, SampleData const &b) {
                return a.current_feature_value < b.current_feature_value;
              });

    if (sample_map_[start_].current_feature_value ==
        sample_map_[end_ - 1].current_feature_value) {
      current_feature_num++;
      continue;
    }

    // loop through possible split positions
    int pos = start_ + 1;
    // split left [0, pos - 1] right [pos, end_-1]
    while (pos < end_) {
      // skip criterion evaluation for splits with less that kMinSplitDiff
      // difference across split + kMinSplitDiff_
      while (pos < end_ &&
             sample_map_[pos].current_feature_value <=
                 sample_map_[pos - 1].current_feature_value + kMinSplitDiff_) {
        pos++;
      }
      if (pos == end_) {
        pos++;
        continue;
      }

      // check if split would lead to less than min_samples_leaf samples
      if (!((pos - start_) < min_samples_leaf_ ||
            ((end_ - pos) < min_samples_leaf_))) {
        current_split.pos = pos;
        criterion_.UpdateSplitPos(current_split.pos);
        current_impurity_improvement = criterion_.ImpurityImprovement();

        if (current_impurity_improvement > best_impurity_improvement) {
          best_impurity_improvement = current_impurity_improvement;
          current_split.found_split = true;
          current_split.threshold =
              (sample_map_[pos - 1].current_feature_value +
               sample_map_[pos].current_feature_value) /
              2.0;
          best_split = current_split;
          best_split.left_value = criterion_.label_freqs_left_;
          best_split.right_value = criterion_.label_freqs_right_;
        }
      }
      pos++;
    }

    current_feature_num++;
  }

  if (current_split.found_split) {
    if (best_split.pos < end_) {
      if (current_feature != best_split.feature) {
        int left_pos = start_;
        int right_pos = end_;
        int tmp = 0;

        while (left_pos < right_pos) {
          if ((*feature_data_)[sample_map_[left_pos].sample_number]
                              [best_split.feature] <= best_split.threshold) {
            left_pos++;
          } else {
            right_pos--;
            tmp = sample_map_[left_pos].sample_number;
            sample_map_[left_pos].sample_number =
                sample_map_[right_pos].sample_number;
            sample_map_[right_pos].sample_number = tmp;
          }
        }
      }
    }

    criterion_.ResetStats();
    criterion_.UpdateSplitPos(best_split.pos);
    criterion_.ChildrenImpurities(best_split.impurity_left,
                                  best_split.impurity_right);
    split = best_split;
  } else {
    // passing back split.found_split = false
    split = current_split;
  }
}
