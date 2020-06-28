#include "core/criterion.h"

#include <gtest/gtest.h>

#include <vector>

#include "test/test.h"

class CriterionTest : public ::testing::Test {
 protected:
  CriterionTest() {
    table data_frame = ReadCsv(g_testdata_path + "/titanic_data.csv");
    Split_X_y(data_frame, X, y, "Survived");

    for (std::size_t i = 0; i < X.size(); i++) {
      SampleData sample;
      sample.current_feature_value = 0.0;
      sample.sample_number = i;
      sample_map.push_back(sample);
    }

    for (int i = 0; i < X.size(); i++) {
      sample_map[i].current_feature_value = X[sample_map[i].sample_number][0];
    }
  }
  std::vector<SampleData> sample_map;
  std::vector<int> y;
  std::vector<std::vector<double>> X;
  int n_labels = 2;
};

TEST_F(CriterionTest, SetNodeLimits) {
  Criterion criterion = Criterion(gini, n_labels, &X, &y);
  criterion.SetData(&sample_map);
  criterion.SetNodeLimits(0, 200);

  EXPECT_EQ(criterion.start_, 0);
  EXPECT_EQ(criterion.end_, 200);
  EXPECT_EQ(criterion.label_freqs_total_[0], 131);
  EXPECT_EQ(criterion.label_freqs_total_[1], 69);
}

TEST_F(CriterionTest, ResetStats) {
  Criterion criterion = Criterion(gini, n_labels, &X, &y);
  criterion.SetData(&sample_map);
  criterion.SetNodeLimits(0, 200);
  criterion.ResetStats();
  EXPECT_EQ(criterion.label_freqs_left_[0], 0);
  EXPECT_EQ(criterion.label_freqs_left_[1], 0);
  EXPECT_EQ(criterion.label_freqs_left_[2], 0);
  EXPECT_EQ(criterion.label_freqs_right_[0], 0);
  EXPECT_EQ(criterion.label_freqs_right_[1], 0);
  EXPECT_EQ(criterion.label_freqs_right_[2], 0);
}

TEST_F(CriterionTest, UpdateSplitPosition) {
  Criterion criterion = Criterion(gini, n_labels, &X, &y);
  criterion.SetData(&sample_map);
  criterion.SetNodeLimits(0, 200);
  criterion.UpdateSplitPos(50);

  EXPECT_EQ(criterion.pos_, 50);

  EXPECT_EQ(criterion.label_freqs_left_[0], 28);
  EXPECT_EQ(criterion.label_freqs_left_[1], 22);
  EXPECT_EQ(criterion.label_freqs_right_[0], 103);
  EXPECT_EQ(criterion.label_freqs_right_[1], 47);
}

TEST_F(CriterionTest, NodeImpurityGini) {
  Criterion criterion = Criterion(gini, n_labels, &X, &y);
  criterion.SetData(&sample_map);
  criterion.SetNodeLimits(0, 891);

  EXPECT_FLOAT_EQ(criterion.NodeImpurity(), 0.47301295);
}

TEST_F(CriterionTest, NodeImpurityEntropy) {
  Criterion criterion = Criterion(entropy, n_labels, &X, &y);
  criterion.SetData(&sample_map);
  criterion.SetNodeLimits(0, 891);

  EXPECT_FLOAT_EQ(criterion.NodeImpurity(), 0.9607079);
}

TEST_F(CriterionTest, ImpurityImprovement) {
  Criterion criterion = Criterion(gini, n_labels, &X, &y);
  criterion.SetData(&sample_map);
  criterion.SetNodeLimits(0, 200);
  criterion.UpdateSplitPos(100);

  EXPECT_FLOAT_EQ(criterion.ImpurityImprovement(), -88.699997);
}

TEST_F(CriterionTest, ChildrenImpurities) {
  double impurity_left = 0.0;
  double impurity_right = 0.0;
  Criterion criterion = Criterion(gini, n_labels, &X, &y);
  criterion.SetData(&sample_map);
  criterion.SetNodeLimits(0, 200);
  criterion.UpdateSplitPos(100);
  criterion.ChildrenImpurities(impurity_left, impurity_right);
  EXPECT_FLOAT_EQ(impurity_left, 0.48379999);
  EXPECT_FLOAT_EQ(impurity_right, 0.4032);
}

TEST_F(CriterionTest, GiniCoefficient) {
  std::vector<int> label_freqs = {20, 80};
  int n_samples = 100;
  int n_labels = 2;
  EXPECT_FLOAT_EQ(Criterion::GiniCoefficient(label_freqs, n_samples, n_labels),
                  0.31999999);
}

TEST_F(CriterionTest, Entropy) {
  std::vector<int> label_freqs = {20, 80};
  int n_samples = 100;
  int n_labels = 2;
  EXPECT_FLOAT_EQ(Criterion::Entropy(label_freqs, n_samples, n_labels),
                  0.72192812);
}
