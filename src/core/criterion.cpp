#include "core/criterion.h"

#include <vector>

Criterion::Criterion(ImpurityMeasure impurity_measure, int n_labels,
                     const std::vector<std::vector<double>> *feature_data,
                     const std::vector<int> *label_data) {
  impurity_measure_ = impurity_measure;

  switch (impurity_measure_) {
    case gini: {
      impurity_fn_ = Criterion::GiniCoefficient;
      break;
    }
    case entropy: {
      impurity_fn_ = Criterion::Entropy;
      break;
    }
    default: {
      throw std::invalid_argument("Unknown impurity measure.");
    }
  }

  n_labels_ = n_labels;
  feature_data_ = feature_data;
  label_data_ = label_data;
  pos_ = 0;
}

void Criterion::SetNodeLimits(int start, int end) {
  start_ = start;
  end_ = end;
  n_samples_ = end_ - start_;

  label_freqs_total_.clear();
  label_freqs_left_.clear();
  label_freqs_right_.clear();

  // calculate label frequencies across all samples at current node
  for (int i = 0; i < n_labels_; i++) {
    label_freqs_total_.emplace_back(0);
    label_freqs_left_.emplace_back(0);
    label_freqs_right_.emplace_back(0);
  }

  for (int pos = start_; pos < end_; pos++) {
    // This assumes that the labels are in colum 0 of the data_table_ and
    // take values 0,..., n_labels - 1
    label_freqs_total_[(*label_data_)[(*sample_map_ptr_)[pos].sample_number]] +=
        1;
  }
}

void Criterion::ResetStats() {
  pos_ = start_;
  label_freqs_left_.clear();
  label_freqs_right_.clear();
  for (int i = 0; i < n_labels_; i++) {
    label_freqs_left_.emplace_back(0);
    label_freqs_right_.emplace_back(0);
  }
}

void Criterion::SetData(const std::vector<SampleData> *sample_map_ptr) {
  sample_map_ptr_ = sample_map_ptr;
}

void Criterion::UpdateSplitPos(int new_pos) {
  // calculate label_freqs_left_
  for (int i = pos_; i < new_pos; i++) {
    // This assumes that the labels take values 0,..., n_labels - 1
    label_freqs_left_[(*label_data_)[(*sample_map_ptr_)[i].sample_number]] += 1;
  }

  // calculate label_freqs_right_
  for (int i = 0; i < n_labels_; i++) {
    label_freqs_right_[i] = label_freqs_total_[i] - label_freqs_left_[i];
  }

  pos_ = new_pos;
  n_samples_left_ = pos_ - start_;
  n_samples_right_ = end_ - pos_;
}

double Criterion::ImpurityImprovement() {
  double impurity_left;
  double impurity_right;
  ChildrenImpurities(impurity_left, impurity_right);
  // ignore constant terms
  return (-n_samples_left_ * impurity_left - n_samples_right_ * impurity_right);
}

double Criterion::GiniCoefficient(std::vector<int> &label_freqs, int &n_samples,
                                  int &n_labels) {
  double freq_squares = 0.0;

  for (int i = 0; i < n_labels; i++) {
    freq_squares += label_freqs[i] * label_freqs[i];
  }
  return 1.0 - freq_squares / (n_samples * n_samples);
}

double Criterion::Entropy(std::vector<int> &label_freqs, int &n_samples,
                          int &n_labels) {
  double entropy = 0.0;
  double label_frequency = 0.0;

  for (int i = 0; i < n_labels; i++) {
    label_frequency = label_freqs[i];
    if (label_frequency > 0) {
      label_frequency /= n_samples;
      entropy -= label_frequency * std::log2(label_frequency);
    }
  }
  return entropy;
}

void Criterion::ChildrenImpurities(double &impurity_left,
                                   double &impurity_right) {
  impurity_left = impurity_fn_(label_freqs_left_, n_samples_left_, n_labels_);
  impurity_right =
      impurity_fn_(label_freqs_right_, n_samples_right_, n_labels_);
}

double Criterion::NodeImpurity() {
  return impurity_fn_(label_freqs_total_, n_samples_, n_labels_);
}
