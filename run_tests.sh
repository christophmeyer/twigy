# build test_runner and install to ./test/bin
cd ./build
cmake .. -DCMAKE_INSTALL_PREFIX=../
cmake --build . --target test_runner
cmake -DCOMPONENT="tests" -P cmake_install.cmake
cd ..

# run test_runner
./test/bin/test_runner -p ./test/testdata

# install python package from wheels in docker image
# and run python tests. 
docker run -it \
 -v "$PWD/build:/build" \
 -v "$PWD/test:/test" cmeyr/tests_python:latest \
 bash /test/install_and_run_python_tests.sh

