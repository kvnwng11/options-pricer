cmake_minimum_required(VERSION 3.16)
project(OptionsPricingEngine CXX)

# ─────────────────────────────────────────────
#  C++ standard & compiler flags
# ─────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /WX /O2 /openmp)
else()
    add_compile_options(
        -Wall -Wextra -Wpedantic -Werror
        -O2 -march=native
        -fno-math-errno
    )
endif()

# ─────────────────────────────────────────────
#  OpenMP (required for MC parallelism)
# ─────────────────────────────────────────────
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found — parallel MC enabled")
else()
    message(WARNING "OpenMP not found — MC will run single-threaded")
endif()

# ─────────────────────────────────────────────
#  Black-Scholes library
# ─────────────────────────────────────────────
add_library(bs_engine STATIC
    src/black_scholes.cpp
)
target_include_directories(bs_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bs_engine PUBLIC m)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bs_engine PUBLIC OpenMP::OpenMP_CXX)
endif()

# ─────────────────────────────────────────────
#  American Options Tree library
# ─────────────────────────────────────────────
add_library(american_engine STATIC
    src/american_options.cpp
)
target_include_directories(american_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(american_engine PUBLIC bs_engine)

# ─────────────────────────────────────────────
#  Monte Carlo library
# ─────────────────────────────────────────────
add_library(mc_engine STATIC
    src/monte_carlo.cpp
)
target_include_directories(mc_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mc_engine PUBLIC bs_engine)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mc_engine PUBLIC OpenMP::OpenMP_CXX)
endif()

# ─────────────────────────────────────────────
#  Demo executable
# ─────────────────────────────────────────────
add_executable(pricing_demo src/main.cpp)
target_link_libraries(pricing_demo PRIVATE bs_engine american_engine mc_engine)

# ─────────────────────────────────────────────
#  Unit tests (Google Test)
# ─────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(bs_tests tests/test_black_scholes.cpp)
target_link_libraries(bs_tests PRIVATE bs_engine GTest::gtest_main)

add_executable(american_tests tests/test_american_options.cpp)
target_link_libraries(american_tests PRIVATE american_engine GTest::gtest_main)

add_executable(mc_tests tests/test_monte_carlo.cpp)
target_link_libraries(mc_tests PRIVATE mc_engine GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(bs_tests)
gtest_discover_tests(american_tests)
gtest_discover_tests(mc_tests)

# ─────────────────────────────────────────────
#  Build summary
# ─────────────────────────────────────────────
message(STATUS "")
message(STATUS "──────────────────────────────────────────────────")
message(STATUS "  Options Pricing Engine")
message(STATUS "  Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ std    : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenMP     : ${OpenMP_CXX_FOUND}")
message(STATUS "  Targets    : bs_engine, american_engine, mc_engine")
message(STATUS "──────────────────────────────────────────────────")
message(STATUS "")