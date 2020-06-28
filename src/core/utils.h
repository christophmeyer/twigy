#ifndef TWIGY_CORE_UTILS_H_
#define TWIGY_CORE_UTILS_H_

#include <algorithm>
#include <random>
#include <string>
#include <vector>

struct SampleData {
  int sample_number;
  double current_feature_value;
};

class table {
 public:
  std::vector<std::vector<double>> data_;
  std::vector<std::string> headers_;
  std::vector<int> col_widths_;

  void AddRow(std::vector<double> row);

  void SetHeaders(std::vector<std::string> h);

  void Print(size_t num_rows);
};

table ReadCsv(const std::string &input_path);

std::vector<int> GenerateRandomIntVector(int a, int b, int n_samples,
                                         std::mt19937 &gen);

int ArgMax(std::vector<double> &vector);
int ArgMax(std::vector<int> &vector);

void Split_X_y(table &data_frame, std::vector<std::vector<double>> &X,
               std::vector<int> &y, std::string feature);

void CheckNegativeLabels(const std::vector<int> *label_data);

#endif  // TWIGY_CORE_UTILS_H_
