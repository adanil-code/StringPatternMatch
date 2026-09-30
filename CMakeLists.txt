cmake_minimum_required(VERSION 3.10)
project(StringPatternMatchTest CXX)

# Enforce C++20 as required by the application
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Disable the noisy -Wpsabi warning for GCC/Clang when returning SIMD vectors
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wno-psabi)
endif()

# Ensure the output binary is placed in a /bin folder inside the build directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Locate threading libraries for cross-platform thread pooling
find_package(Threads REQUIRED)

# Define the executable using the updated test folder path
add_executable(string_pattern_match_test
    test/string_pattern_match_test.cpp
)

# Include both the library and test folders so all headers can be resolved
target_include_directories(string_pattern_match_test PRIVATE 
    string_pattern_match
    test
)

# Link threads
target_link_libraries(string_pattern_match_test PRIVATE Threads::Threads)
