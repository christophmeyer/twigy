#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <vector>

#include "./core/decision_tree.h"
#include "./core/random_forest.h"

namespace py = pybind11;

PYBIND11_MODULE(twigy, m) {
  py::class_<Node>(m, "Node")
      .def_readwrite("parent_id", &Node::parent_id)
      .def_readwrite("left_child_id", &Node::left_child_id)
      .def_readwrite("right_child_id", &Node::right_child_id)
      .def_readwrite("is_leaf", &Node::is_leaf)
      .def_readwrite("impurity", &Node::impurity)
      .def_readwrite("split_feature", &Node::split_feature)
      .def_readwrite("threshold", &Node::threshold)
      .def_readwrite("n_samples", &Node::n_samples)
      .def_readwrite("value", &Node::value);

  py::class_<DecisionTree>(m, "DecisionTree")
      .def_readwrite("nodes", &DecisionTree::nodes);

  py::enum_<ImpurityMeasure>(m, "ImpurityMeasure")
      .value("gini", ImpurityMeasure::gini)
      .value("enthropy", ImpurityMeasure::entropy);

  py::enum_<MaxFeaturesMethod>(m, "MaxFeaturesMethod")
      .value("sqrt_method", MaxFeaturesMethod::sqrt_method)
      .value("log2_method", MaxFeaturesMethod::log2_method)
      .value("all_method", MaxFeaturesMethod::all_method);

  py::class_<RandomForestClassifier>(m, "RandomForestClassifier")
      .def(py::init<int, ImpurityMeasure, int, int, int, int, MaxFeaturesMethod,
                    double, int, int>(),
           py::arg("n_estimators") = 100, py::arg("impurity_measure") = gini,
           py::arg("max_depth") = std::numeric_limits<int>::max(),
           py::arg("min_samples_split") = 2, py::arg("min_samples_leaf") = 1,
           py::arg("max_features") = -1,
           py::arg("max_features_method") = sqrt_method,
           py::arg("min_impurity_split") = 0.0, py::arg("n_threads") = 1,
           py::arg("max_samples") = -1)
      .def("build_forest", &RandomForestClassifier::BuildForest)
      .def("predict_classes", &RandomForestClassifier::PredictClasses);

  py::class_<DecisionTreeClassifier>(m, "DecisionTreeClassifier")
      .def(py::init<ImpurityMeasure, int, int, int, int, MaxFeaturesMethod,
                    double>(),
           py::arg("impurity_measure") = gini,
           py::arg("max_depth") = std::numeric_limits<int>::max(),
           py::arg("min_samples_split") = 2, py::arg("min_samples_leaf") = 1,
           py::arg("max_features") = -1,
           py::arg("max_features_method") = sqrt_method,
           py::arg("min_impurity_split") = 0.0)
      .def_readwrite("tree", &DecisionTreeClassifier::tree_)
      .def("build_tree", (void (DecisionTreeClassifier::*)(
                             const std::vector<std::vector<double>>,
                             const std::vector<int>)) &
                             DecisionTreeClassifier::BuildTree)
      .def("print_tree", &DecisionTreeClassifier::PrintTree)
      .def("predict_classes", &DecisionTreeClassifier::PredictClasses);
}
