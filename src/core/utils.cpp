#include "core/utils.h"

#include <array>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "csv/parser.hpp"

void Split_X_y(table &data_frame, std::vector<std::vector<double>> &X,
               std::vector<int> &y, std::string feature) {
  auto it = std::find(data_frame.headers_.begin(), data_frame.headers_.end(),
                      feature);
  std::size_t label_idx = std::distance(data_frame.headers_.begin(), it);
  for (std::size_t row_idx = 0; row_idx < data_frame.data_.size(); row_idx++) {
    std::vector<double> current_row;
    for (std::size_t col_idx = 0; col_idx < data_frame.headers_.size();
         col_idx++) {
      if (col_idx != label_idx) {
        current_row.push_back(data_frame.data_[row_idx][col_idx]);
      }
    }
    X.push_back(current_row);
    y.push_back(data_frame.data_[row_idx][label_idx]);
  }
}

void table::AddRow(std::vector<double> row) { data_.push_back(row); }

void table::SetHeaders(std::vector<std::string> h) { headers_ = h; }

void table::Print(size_t num_rows) {
  for (size_t j = 0; j < headers_.size(); j++) {
    col_widths_.push_back(std::max(static_cast<int>(headers_[j].length()), 6));
  }
  size_t total_width = headers_.size();
  for (auto &n : col_widths_) {
    total_width += n;
  }

  for (size_t j = 0; j < total_width; j++) {
    std::cout << "-";
  }

  std::cout << std::endl;

  for (size_t j = 0; j < headers_.size(); j++) {
    std::cout << std::setw(col_widths_[j]) << headers_[j] << "|";
  }

  std::cout << std::endl;
  for (size_t j = 0; j < total_width; j++) {
    std::cout << "-";
  }

  std::cout << std::endl;
  for (size_t i = 0; i < num_rows; i++) {
    for (size_t j = 0; j < data_[i].size(); j++) {
      std::cout << std::setw(col_widths_[j]) << data_[i][j] << "|";
    }
    std::cout << std::endl;
  }
}

table ReadCsv(const std::string &input_path) {
  std::ifstream input_file(input_path);
  aria::csv::CsvParser parser = aria::csv::CsvParser(input_file);

  bool first_row = true;
  table data;
  std::vector<double> data_row;
  std::vector<std::string> headers;

  for (auto &row : parser) {
    if (first_row) {
      first_row = false;
      for (auto &field : row) {
        headers.push_back(field);
      }
      data.SetHeaders(headers);
      continue;
    }

    for (auto &field : row) {
      data_row.push_back(std::stod(field));
    }
    data.AddRow(data_row);
    data_row.clear();
  }
  return data;
}

std::vector<int> GenerateRandomIntVector(int a, int b, int n_samples,
                                         std::mt19937 &gen) {
  std::uniform_int_distribution<> distrib(a, b);
  std::vector<int> rand_vector;
  for (int i = 0; i < n_samples; i++) {
    rand_vector.push_back(distrib(gen));
  }
  return rand_vector;
}

void CheckNegativeLabels(const std::vector<int> *label_data) {
  for (std::size_t i = 0; i < label_data->size(); i++) {
    if ((*label_data)[i] < 0) {
      throw std::invalid_argument(
          "Negative class label given. The labels must be given by 0,1, ... "
          "n_labels - 1.");
    }
  }
}

int ArgMax(std::vector<double> &vector) {
  std::size_t i = 0;
  std::size_t max_pos = 0;
  while (i < vector.size()) {
    if (vector[max_pos] < vector[i]) {
      max_pos = i;
    }
    i++;
  }
  return max_pos;
}

int ArgMax(std::vector<int> &vector) {
  std::size_t i = 0;
  std::size_t max_pos = 0;
  while (i < vector.size()) {
    if (vector[max_pos] < vector[i]) {
      max_pos = i;
    }
    i++;
  }
  return max_pos;
}
