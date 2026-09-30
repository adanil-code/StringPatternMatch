#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Default configuration values
BUILD_TYPE="Release"
COMPILER="g++"
CLEAN_BUILD=false

# Capture the exact directory where this script resides (sample/)
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# 1. Parse Command Line Arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--clean) 
            CLEAN_BUILD=true
            shift 
            ;;
        -t|--type) 
            BUILD_TYPE="$2"
            shift 2 
            ;;
        --compiler) 
            COMPILER="$2"
            shift 2 
            ;;
        -h|--help)
            echo "Usage: ./build.sh [OPTIONS]"
            echo "Options:"
            echo "  -c, --clean     Wipe bin directory before starting"
            echo "  -t, --type      Build type (Release/Debug) [Default: Release]"
            echo "  --compiler      Compiler choice (g++, clang++) [Default: g++]"
            echo "  -h, --help      Display this help message"
            exit 0
            ;;
        *) 
            echo "Error: Unknown parameter: $1"
            exit 1 
            ;;
    esac
done

# Validate build type
if [[ "$BUILD_TYPE" != "Release" && "$BUILD_TYPE" != "Debug" ]]; then
    echo "Error: Invalid build type '$BUILD_TYPE'. Use 'Release' or 'Debug'."
    exit 1
fi

BIN_DIR="$SCRIPT_DIR/build/bin"

# 2. Wipe old build folder if a clean build was requested
if [ "$CLEAN_BUILD" = true ]; then
    echo "Wiping build directory..."
    rm -rf "$SCRIPT_DIR/build"
fi

echo "========================================="
echo " Building unicode_ci_matching            "
echo " Build Type:   $BUILD_TYPE               "
echo " Compiler:     $COMPILER                 "
echo "========================================="

mkdir -p "$BIN_DIR"

# 3. Configure compiler flags for speed
# -std=c++20: Required for string_view and modern language features
# -I: Tells the compiler where to find the header files
CXX_FLAGS="-std=c++20 -I$SCRIPT_DIR/../string_pattern_match -I$SCRIPT_DIR"

if [ "$BUILD_TYPE" = "Release" ]; then
    # Use -flto=auto for GCC to parallelize LTRANS jobs across available CPU cores;
    # Clang uses standard -flto.
    if [[ "$COMPILER" == *"clang"* ]]; then
        LTO_FLAG="-flto"
    else
        LTO_FLAG="-flto=auto"
    fi

    # Maximum speed: -O3, native vectorization, Link-Time Optimization
    CXX_FLAGS="$CXX_FLAGS -O3 -march=native $LTO_FLAG -fomit-frame-pointer -DNDEBUG"
else
    CXX_FLAGS="$CXX_FLAGS -g -O0 -D_DEBUG"
fi

# 4. Compile directly (No CMake)
echo "-> Compiling with portable speed optimizations..."
$COMPILER $CXX_FLAGS "$SCRIPT_DIR/unicode_ci_matching.cpp" -o "$BIN_DIR/unicode_ci_matching"

echo "========================================="
echo " Build successful!                       "
echo " Run via: ./build/bin/unicode_ci_matching"
echo "========================================="