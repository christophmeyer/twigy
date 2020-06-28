cmake_minimum_required(VERSION 3.10)

project(twigy VERSION 0.1 DESCRIPTION "Fast Random Forest in CPP")

add_library(twigycpp STATIC
 src/core/random_forest.cpp
 src/core/utils.cpp
 src/core/criterion.cpp
 src/core/splitter.cpp
 src/core/decision_tree.cpp)
 target_include_directories(twigycpp PUBLIC ./src)

add_executable(example example.cpp)
target_include_directories(example PUBLIC ./src)
target_compile_options(example PUBLIC -std=c++11 -O3 -Wall)
target_link_libraries(example PUBLIC twigycpp pthread boost_system)
INSTALL(TARGETS example DESTINATION bin)

add_subdirectory(pybind11)
pybind11_add_module(twigy ./src/python_bindings.cpp src/core/random_forest.cpp
 src/core/utils.cpp
 src/core/criterion.cpp
 src/core/splitter.cpp
 src/core/decision_tree.cpp)
target_compile_options(twigy PUBLIC -std=c++11 -O3 -Wall)
target_include_directories(twigy PUBLIC ./src)
target_link_libraries(twigy PRIVATE pthread boost_system)

add_executable(test_runner ./test/test.cpp 
./src/core/decision_tree_test.cpp
./src/core/criterion_test.cpp
./src/core/splitter_test.cpp)
target_include_directories(test_runner PUBLIC ./src ./)
target_link_libraries(test_runner gtest twigycpp pthread boost_system)
INSTALL(TARGETS test_runner DESTINATION test/bin COMPONENT tests)