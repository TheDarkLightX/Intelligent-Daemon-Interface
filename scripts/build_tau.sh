#!/bin/bash
# Build tau-lang from source for benchmarking
# Prerequisites: cmake, g++ (13+), libcvc5-dev, libboost-all-dev

set -e

TAU_LANG_DIR="${TAU_LANG_DIR:-/home/trevormoc/Downloads/tau-lang-latest}"
BUILD_TYPE="${1:-Release}"

echo "=== Tau-lang Build Script ==="
echo "Source: $TAU_LANG_DIR"
echo "Build type: $BUILD_TYPE"

# Check dependencies
check_dep() {
    if ! command -v $1 &> /dev/null; then
        echo "ERROR: $1 is required but not found"
        exit 1
    fi
}

check_dep cmake
check_dep g++

# Check for cvc5 dev headers
if ! dpkg -l | grep -q libcvc5-dev; then
    echo "WARNING: libcvc5-dev not found. Install with: sudo apt install libcvc5-dev"
    echo "Continuing anyway..."
fi

cd "$TAU_LANG_DIR"

# Initialize submodules
git submodule update --init --recursive

# Build
echo "Building tau-lang ($BUILD_TYPE)..."
./scripts/build.sh $BUILD_TYPE -DTAU_MEASURE=ON

# Check result
if [ -f "build-$BUILD_TYPE/tau" ]; then
    echo "SUCCESS: tau binary built at build-$BUILD_TYPE/tau"
    echo "Version: $(./build-$BUILD_TYPE/tau --version 2>&1 | head -1)"
else
    echo "ERROR: Build failed - tau binary not found"
    exit 1
fi

echo "=== Build Complete ==="

