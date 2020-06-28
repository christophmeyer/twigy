#include "core/decision_tree.h"

#include <iostream>
#include <vector>

void NodeStack::push(Record record) { stack.push_back(record); }

bool NodeStack::IsEmpty() { return stack.size() == 0; }

void NodeStack::pop(Record &record) {
  record = stack.back();
  stack.pop_back();
}

void DecisionTree::PopulateChildIds() {
  for (std::size_t i = 1; i < nodes.size(); i++) {
    if (nodes[i].is_left) {
      nodes[nodes[i].parent_id].left_child_id = i;
    } else {
      nodes[nodes[i].parent_id].right_child_id = i;
    }
  }
}

void SetMaxFeatures(std::size_t &max_features,
                    MaxFeaturesMethod &max_features_method,
                    const std::vector<std::vector<double>> *feature_data) {
  std::size_t n_features = (*feature_data)[0].size();
  if (max_features < 1) {
    switch (max_features_method) {
      case log2_method:
        max_features = static_cast<int>(std::log2((*feature_data)[0].size()));
        break;
      case sqrt_method:
        max_features = static_cast<int>(std::sqrt((*feature_data)[0].size()));
        break;
      case all_method:
        max_features = n_features;
        break;
    }
  } else if (max_features > n_features) {
    max_features = n_features;
  }
}

std::vector<std::vector<double>> DecisionTree::PredictClassProbabilities(
    const std::vector<std::vector<double>> *feature_data_) {
  if (!is_built) {
    throw std::logic_error(
        "Decision tree must be built with BuildTree method before predictions "
        "can be made.");
  }
  std::vector<int> leaf_node_ids = DecisionTree::GetLeafNodes(feature_data_);
  std::vector<std::vector<double>> class_probabilities;
  std::vector<double> current_class_probabilities;
  class_probabilities.reserve(leaf_node_ids.size());
  current_class_probabilities.reserve(nodes[0].value.size());
  int current_node_id;
  for (std::size_t n_sample = 0; n_sample < leaf_node_ids.size(); n_sample++) {
    current_class_probabilities.clear();
    current_node_id = leaf_node_ids[n_sample];
    for (std::size_t n_class = 0; n_class < nodes[0].value.size(); n_class++) {
      current_class_probabilities.push_back(
          nodes[current_node_id].value[n_class] /
          static_cast<double>(nodes[current_node_id].n_samples));
    }
    class_probabilities.push_back(current_class_probabilities);
  }
  return class_probabilities;
}

std::vector<int> DecisionTree::PredictClasses(
    const std::vector<std::vector<double>> *feature_data_ptr) {
  if (!is_built) {
    throw std::logic_error(
        "Decision tree must be built with BuildTree method before predictions "
        "can be made.");
  }
  std::vector<int> leaf_node_ids = DecisionTree::GetLeafNodes(feature_data_ptr);
  std::vector<int> class_predictions;
  class_predictions.reserve(leaf_node_ids.size());
  int current_node_id;
  for (std::size_t n_sample = 0; n_sample < leaf_node_ids.size(); n_sample++) {
    current_node_id = leaf_node_ids[n_sample];
    class_predictions.push_back(ArgMax(nodes[current_node_id].value));
    nodes[current_node_id].value;
  }
  return class_predictions;
}

std::vector<int> DecisionTreeClassifier::PredictClasses(
    const std::vector<std::vector<double>> *feature_data_ptr) {
  return tree_.PredictClasses(feature_data_ptr);
}

std::vector<int> DecisionTree::GetLeafNodes(
    const std::vector<std::vector<double>> *feature_data_ptr) {
  std::vector<int> leaf_node_ids;
  for (std::size_t i = 0; i < (*feature_data_ptr).size(); i++) {
    int node_id = 0;
    while (!nodes[node_id].is_leaf) {
      if ((*feature_data_ptr)[i][nodes[node_id].split_feature] <=
          nodes[node_id].threshold) {
        node_id = nodes[node_id].left_child_id;
      } else {
        node_id = nodes[node_id].right_child_id;
      }
    }
    leaf_node_ids.push_back(node_id);
  }
  return leaf_node_ids;
}

DecisionTreeClassifier::DecisionTreeClassifier(
    ImpurityMeasure impurity_measure, int max_depth, int min_samples_split,
    int min_samples_leaf, int max_features,
    MaxFeaturesMethod max_features_method, double min_impurity_split) {
  impurity_measure_ = impurity_measure;
  max_depth_ = max_depth;
  min_samples_split_ = min_samples_split;
  min_samples_leaf_ = min_samples_leaf;
  max_features_ = max_features;
  max_features_method_ = max_features_method;
  min_impurity_split_ = min_impurity_split;
}

int DecisionTree::AddNode(Node node) {
  int node_id = nodes.size();
  nodes.emplace_back(node);
  return node_id;
  void Print();
}

void DecisionTreeClassifier::PrintTree() { tree_.Print(); }

void DecisionTree::Print() {
  if (!is_built) {
    throw std::logic_error(
        "Decision tree must be built before it can be printed.");
  }
  for (std::size_t i = 0; i < nodes.size(); i++) {
    std::cout << "=============================" << std::endl;
    std::cout << "node id: " << i << std::endl;
    nodes[i].Print();
  }
  std::cout << "=============================" << std::endl;
}

void Node::Print() {
  std::cout << "parent id: " << parent_id << std::endl;
  std::cout << "criterion: " << impurity << std::endl;
  std::cout << "n_samples: " << n_samples << std::endl;
  std::cout << "is leaf: " << is_leaf << std::endl;
  std::cout << "is left: " << is_left << std::endl;
  if (!is_leaf) {
    std::cout << "feature: " << split_feature << std::endl;
    std::cout << "threshold: " << threshold << std::endl;
    std::cout << "left child id: " << left_child_id << std::endl;
    std::cout << "right child id: " << right_child_id << std::endl;
  }
  std::cout << "value: [ ";
  for (std::size_t i = 0; i < value.size(); i++) {
    std::cout << value[i] << " ";
  }
  std::cout << "]" << std::endl;
}

void DecisionTreeClassifier::BuildTree(
    const std::vector<std::vector<double>> feature_data,
    const std::vector<int> label_data) {
  feature_data_ = feature_data;
  label_data_ = label_data;
  const std::vector<int> samples_subset = {};
  int n_labels = *std::max_element(label_data_.begin(), label_data_.end()) + 1;
  generator.seed(42);

  this->BuildTree(&feature_data_, &label_data_, samples_subset, n_labels,
                  &generator);
}

void DecisionTreeClassifier::BuildTree(
    const std::vector<std::vector<double>> *feature_data_ptr,
    const std::vector<int> *label_data_ptr,
    const std::vector<int> samples_subset, int n_labels, std::mt19937 *gen_) {
  CheckNegativeLabels(label_data_ptr);
  SetMaxFeatures(max_features_, max_features_method_, feature_data_ptr);

  splitter_ = Splitter(feature_data_ptr, label_data_ptr, min_samples_leaf_,
                       impurity_measure_, max_features_, n_labels, gen_,
                       samples_subset);
  splitter_.criterion_.SetData(&(splitter_.sample_map_));

  // put root node on stack
  Record root_node;
  root_node.start = 0;
  root_node.end = splitter_.sample_map_.size();
  root_node.depth = 0;
  root_node.impurity = 0;
  root_node.n_samples = root_node.end;
  root_node.parent_id = -1;
  root_node.is_left = 0;
  stack_.push(root_node);

  Record current_record;
  Record left_record;
  Record right_record;
  Split current_split;
  bool is_leaf;
  bool is_root_node = true;
  int node_id;
  Node current_node;

  while (!stack_.IsEmpty()) {
    // take next node from stack
    stack_.pop(current_record);

    // evaluate abort criterion
    is_leaf = !(current_record.depth < max_depth_) ||
              current_record.n_samples < min_samples_split_ ||
              current_record.n_samples < 2 * min_samples_leaf_;
    // or current_record.impurity <= 0.0;
    // split unless is_leaf
    if (!is_leaf) {
      splitter_.ResetSampleRange(current_record.start, current_record.end);
      splitter_.SplitNode(current_split);
    }

    if (is_root_node) {
      splitter_.ResetSampleRange(current_record.start, current_record.end);
      splitter_.SplitNode(current_split);
      current_record.impurity = splitter_.criterion_.NodeImpurity();
      current_record.value = splitter_.criterion_.label_freqs_total_;
      is_root_node = false;
    }

    is_leaf = is_leaf || !current_split.found_split ||
              current_record.impurity <= min_impurity_split_;

    current_node.parent_id = current_record.parent_id;
    current_node.impurity = current_record.impurity;
    current_node.is_leaf = is_leaf;
    current_node.is_left = current_record.is_left;
    current_node.n_samples = current_record.n_samples;
    current_node.split_feature = current_split.feature;
    current_node.threshold = current_split.threshold;
    current_node.value = current_record.value;

    node_id = tree_.AddNode(current_node);

    if (!is_leaf) {
      right_record.start = current_split.pos;
      right_record.end = current_record.end;
      right_record.n_samples = current_record.end - current_split.pos;
      right_record.depth = current_record.depth + 1;
      right_record.parent_id = node_id;
      right_record.is_left = false;
      right_record.impurity = current_split.impurity_right;
      right_record.value = current_split.right_value;

      stack_.push(right_record);

      left_record.start = current_record.start;
      left_record.end = current_split.pos;
      left_record.n_samples = current_split.pos - current_record.start;
      left_record.depth = current_record.depth + 1;
      left_record.parent_id = node_id;
      left_record.is_left = true;
      left_record.impurity = current_split.impurity_left;
      left_record.value = current_split.left_value;

      stack_.push(left_record);
    }
  }
  tree_.PopulateChildIds();
  tree_.is_built = true;
}
