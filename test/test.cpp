#include "test/test.h"

#include <unistd.h>

#include "gtest/gtest.h"

std::string g_testdata_path = "";

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  int opt;
  while ((opt = getopt(argc, argv, "p:")) != -1) {
    switch (opt) {
      case 'p':
        g_testdata_path = optarg;
        break;
    }
  }
  if (g_testdata_path == "") {
    throw std::invalid_argument(
        "Path of test data must be given with -p option.");
  }

  return RUN_ALL_TESTS();
}
