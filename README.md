cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/bs_tests       # run all unit tests
./build/bs_demo        # run the demo