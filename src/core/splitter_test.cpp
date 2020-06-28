#include "core/splitter.h"

#include <gtest/gtest.h>

#include "test/test.h"

class SplitterTest : public ::testing::Test {
 protected:
  SplitterTest() {
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
  ImpurityMeasure impurity_measure = gini;
  int min_samples_leaf = 1;
  int max_features = -1;
  std::vector<int> samples_subset = {};
};

TEST_F(SplitterTest, SetNodeLimits) {
  std::mt19937 generator;
  generator.seed(42);
  Splitter splitter =
      Splitter(&X, &y, min_samples_leaf, impurity_measure, max_features,
               n_labels, &generator, samples_subset);

  splitter.criterion_.SetData(&sample_map);
  splitter.ResetSampleRange(50, 100);

  EXPECT_EQ(splitter.criterion_.start_, 50);
  EXPECT_EQ(splitter.criterion_.end_, 100);
}