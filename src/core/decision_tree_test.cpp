#include "core/decision_tree.h"

#include <gtest/gtest.h>

#include "test/test.h"

class DecisionTreeTest : public ::testing::Test {
 protected:
  DecisionTreeTest() {
    table data_frame = ReadCsv(g_testdata_path + "/titanic_data.csv");
    Split_X_y(data_frame, X, y, "Survived");
  }

  std::vector<int> y;
  std::vector<std::vector<double>> X;
  ImpurityMeasure impurity_measure = gini;
  int min_samples_leaf = 1;
  int max_depth = 5;
  int min_samples_split = 2;
  int max_features = -1;
  MaxFeaturesMethod max_features_method = all_method;
  double min_impurity_split = 0.0;
};

TEST_F(DecisionTreeTest, BuildTreeTestNodes) {
  DecisionTreeClassifier tree_classifier = DecisionTreeClassifier(
      impurity_measure, max_depth, min_samples_split, min_samples_leaf,
      max_features, max_features_method, min_impurity_split);
  tree_classifier.BuildTree(X, y);

  EXPECT_EQ(tree_classifier.tree_.nodes.size(), 53);

  EXPECT_EQ(tree_classifier.tree_.nodes[0].n_samples, 891);
  EXPECT_EQ(tree_classifier.tree_.nodes[0].split_feature, 5);
  EXPECT_EQ(tree_classifier.tree_.nodes[0].threshold, 1.5);
  EXPECT_EQ(tree_classifier.tree_.nodes[0].left_child_id, 1);
  EXPECT_EQ(tree_classifier.tree_.nodes[0].right_child_id, 26);
  EXPECT_FLOAT_EQ(tree_classifier.tree_.nodes[0].impurity, 0.473013);
  EXPECT_EQ(tree_classifier.tree_.nodes[0].value[0], 549);
  EXPECT_EQ(tree_classifier.tree_.nodes[0].value[1], 342);

  EXPECT_EQ(tree_classifier.tree_.nodes[22].n_samples, 32);
  EXPECT_FLOAT_EQ(tree_classifier.tree_.nodes[22].impurity, 0.0);
  EXPECT_EQ(tree_classifier.tree_.nodes[22].value[0], 32);
  EXPECT_EQ(tree_classifier.tree_.nodes[22].value[1], 0);

  EXPECT_EQ(tree_classifier.tree_.nodes[49].n_samples, 21);
  EXPECT_EQ(tree_classifier.tree_.nodes[49].split_feature, 2);
  EXPECT_EQ(tree_classifier.tree_.nodes[49].threshold, 0.5);
  EXPECT_EQ(tree_classifier.tree_.nodes[49].left_child_id, 50);
  EXPECT_EQ(tree_classifier.tree_.nodes[49].right_child_id, 51);
  EXPECT_FLOAT_EQ(tree_classifier.tree_.nodes[49].impurity, 0.17233559);
  EXPECT_EQ(tree_classifier.tree_.nodes[49].value[0], 19);
  EXPECT_EQ(tree_classifier.tree_.nodes[49].value[1], 2);
}

TEST_F(DecisionTreeTest, BuildTreeTestNegMaxDepth) {
  max_depth = -5;
  DecisionTreeClassifier tree_classifier = DecisionTreeClassifier(
      impurity_measure, max_depth, min_samples_split, min_samples_leaf,
      max_features, max_features_method, min_impurity_split);
  tree_classifier.BuildTree(X, y);

  EXPECT_EQ(tree_classifier.tree_.nodes.size(), 1);
}

TEST_F(DecisionTreeTest, BuildTreeTestMaxDepth) {
  max_depth = 1;
  DecisionTreeClassifier tree_classifier = DecisionTreeClassifier(
      impurity_measure, max_depth, min_samples_split, min_samples_leaf,
      max_features, max_features_method, min_impurity_split);
  tree_classifier.BuildTree(X, y);

  EXPECT_EQ(tree_classifier.tree_.nodes.size(), 3);
}

TEST_F(DecisionTreeTest, BuildTreeTestMaxFeatures) {
  max_features = 1;
  DecisionTreeClassifier tree_classifier = DecisionTreeClassifier(
      impurity_measure, max_depth, min_samples_split, min_samples_leaf,
      max_features, max_features_method, min_impurity_split);
  tree_classifier.BuildTree(X, y);

  EXPECT_EQ(tree_classifier.tree_.nodes[0].split_feature, 1);
  EXPECT_EQ(tree_classifier.tree_.nodes[1].split_feature, 7);
  EXPECT_EQ(tree_classifier.tree_.nodes[2].split_feature, 6);
  EXPECT_EQ(tree_classifier.tree_.nodes[3].split_feature, 2);
  EXPECT_EQ(tree_classifier.tree_.nodes[4].split_feature, 7);
}

TEST_F(DecisionTreeTest, BuildTreeTestHighMinImpuritySplit) {
  min_impurity_split = 100;
  DecisionTreeClassifier tree_classifier = DecisionTreeClassifier(
      impurity_measure, max_depth, min_samples_split, min_samples_leaf,
      max_features, max_features_method, min_impurity_split);
  tree_classifier.BuildTree(X, y);

  EXPECT_EQ(tree_classifier.tree_.nodes.size(), 1);
}

TEST_F(DecisionTreeTest, BuildTreeTestNormalMinImpuritySplit) {
  min_impurity_split = 0.1;
  DecisionTreeClassifier tree_classifier = DecisionTreeClassifier(
      impurity_measure, max_depth, min_samples_split, min_samples_leaf,
      max_features, max_features_method, min_impurity_split);
  tree_classifier.BuildTree(X, y);

  EXPECT_EQ(tree_classifier.tree_.nodes.size(), 47);
}
