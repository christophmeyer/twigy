#ifndef TWIGY_CORE_TREE_BUILDER_H_
#define TWIGY_CORE_TREE_BUILDER_H_

#include <limits>
#include <vector>

#include "core/splitter.h"

enum MaxFeaturesMethod { sqrt_method, log2_method, all_method };

struct Record {
  int start;
  int end;
  int n_samples;
  int depth;
  int parent_id;
  bool is_left;
  double impurity;
  std::vector<int> value;
};

struct NodeStack {
  std::vector<Record> stack;
  void push(Record record);
  void pop(Record &record);
  bool IsEmpty();
};

struct Node {
  int parent_id;
  int left_child_id;
  int right_child_id;
  bool is_left;
  bool is_leaf;
  double impurity;
  int split_feature;
  double threshold;
  int n_samples;
  std::vector<int> value;

  void Print();
};

void SetMaxFeatures(std::size_t &max_features,
                    MaxFeaturesMethod &max_features_method,
                    const std::vector<std::vector<double>> *feature_data);

struct DecisionTree {
  std::vector<Node> nodes;
  bool is_built = false;
  int AddNode(Node node);
  void Print();
  std::vector<int> GetLeafNodes(
      const std::vector<std::vector<double>> *feature_data_ptr);
  void PopulateChildIds();
  void PredictClassProbabilities();
  std::vector<std::vector<double>> PredictClassProbabilities(
      const std::vector<std::vector<double>> *samples);
  std::vector<int> PredictClasses(
      const std::vector<std::vector<double>> *feature_data_ptr);
};

class DecisionTreeClassifier {
 public:
  Splitter splitter_;
  NodeStack stack_;
  int min_samples_leaf_;
  int max_depth_;
  int min_samples_split_;
  double min_impurity_split_;
  std::size_t n_features_;
  DecisionTree tree_;
  ImpurityMeasure impurity_measure_;
  std::size_t max_features_;
  MaxFeaturesMethod max_features_method_;
  std::mt19937 *gen_;
  std::mt19937 generator;
  std::vector<std::vector<double>> feature_data_;
  std::vector<int> label_data_;

  DecisionTreeClassifier(ImpurityMeasure impurity_measure = gini,
                         int max_depth = std::numeric_limits<int>::max(),
                         int min_samples_split = 2, int min_samples_leaf = 1,
                         int max_features = -1,
                         MaxFeaturesMethod max_features_method = all_method,
                         double min_impurity_split_ = 0.0);

  void BuildTree(const std::vector<std::vector<double>> *feature_data_ptr,
                 const std::vector<int> *label_data_ptr,
                 const std::vector<int> samples_subset, int n_labels,
                 std::mt19937 *gen_);

  void BuildTree(const std::vector<std::vector<double>> feature_data,
                 const std::vector<int> label_data);
  void PrintTree();
  std::vector<int> PredictClasses(
      const std::vector<std::vector<double>> *feature_data_ptr);
};

#endif  // TWIGY_CORE_TREE_BUILDER_H_
