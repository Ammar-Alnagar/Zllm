#!/bin/bash
# =============================================================================
# build.sh - Build Script
# =============================================================================
# Build the C++/CUDA extensions.
# Usage: ./scripts/build.sh [debug|release]
# =============================================================================

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Build type
BUILD_TYPE="${1:-Release}"
echo "Build type: $BUILD_TYPE"

# Create build directory
BUILD_DIR="$PROJECT_DIR/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "=== Configuring ==="
cmake "$PROJECT_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -GNinja

# Build
echo "=== Building ==="
ninja -j$(nproc)

echo "=== Build Complete ==="

# Run tests
if [[ "$2" == "--test" ]]; then
    echo "=== Running Tests ==="
    ctest --output-on-failure
fi
