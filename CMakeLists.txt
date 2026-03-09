cmake_minimum_required(VERSION 3.16)
project(BlackScholesEngine CXX)

# ─────────────────────────────────────────────
#  C++ standard & compiler flags
# ─────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /WX /O2)
else()
    add_compile_options(
        -Wall -Wextra -Wpedantic -Werror
        -O2 -march=native
        -fno-math-errno          # safe; we validate inputs before calling libm
    )
endif()

# ─────────────────────────────────────────────
#  Pricing library
# ─────────────────────────────────────────────
add_library(bs_engine STATIC
    src/black_scholes.cpp
)

target_include_directories(bs_engine
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link math library on Linux/macOS
target_link_libraries(bs_engine PUBLIC m)

# ─────────────────────────────────────────────
#  Optional: OpenMP (used later by MC engine)
# ─────────────────────────────────────────────
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bs_engine PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found — parallel pricing enabled")
else()
    message(WARNING "OpenMP not found — single-threaded build")
endif()

# ─────────────────────────────────────────────
#  Executable demo (optional smoke test)
# ─────────────────────────────────────────────
add_executable(bs_demo src/main.cpp)
target_link_libraries(bs_demo PRIVATE bs_engine)

# ─────────────────────────────────────────────
#  Unit tests (Google Test)
# ─────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
# Prevent GTest from overriding our compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(bs_tests
    tests/test_black_scholes.cpp
)

target_link_libraries(bs_tests
    PRIVATE bs_engine GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(bs_tests)

# ─────────────────────────────────────────────
#  Build summary
# ─────────────────────────────────────────────
message(STATUS "")
message(STATUS "──────────────────────────────────────")
message(STATUS "  Black-Scholes Engine")
message(STATUS "  Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ std    : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenMP     : ${OpenMP_CXX_FOUND}")
message(STATUS "──────────────────────────────────────")
message(STATUS "")