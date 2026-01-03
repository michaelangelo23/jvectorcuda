#!/bin/bash
# JVectorCUDA - JCuda Setup Script (Linux/macOS)
# Downloads JCuda 12.0.0 JARs for CUDA 11.8 compatibility

set -e

LIBS_DIR="${1:-libs}"
JCUDA_VERSION="12.0.0"
BASE_URL="https://repo1.maven.org/maven2/org/jcuda"

# Detect OS for native classifier
case "$(uname -s)" in
    Linux*)  OS_CLASSIFIER="linux-x86_64";;
    Darwin*) OS_CLASSIFIER="apple-x86_64";;
    *)       OS_CLASSIFIER="linux-x86_64";;
esac

# Required JCuda modules
MODULES=(
    "jcuda"
    "jcuda-natives"
    "jcublas"
    "jcublas-natives"
    "jcurand"
    "jcurand-natives"
    "jcufft"
    "jcufft-natives"
    "jcusparse"
    "jcusparse-natives"
    "jcusolver"
    "jcusolver-natives"
)

echo "========================================"
echo " JVectorCUDA - JCuda Setup Script"
echo "========================================"
echo ""
echo "JCuda Version: $JCUDA_VERSION"
echo "Platform: $OS_CLASSIFIER"
echo "Target Directory: $LIBS_DIR"
echo ""

# Create libs directory
mkdir -p "$LIBS_DIR"

# Download function
download_jar() {
    local module=$1
    local classifier=$2
    local jar_name
    local url
    
    if [ -n "$classifier" ]; then
        jar_name="${module}-${JCUDA_VERSION}-${classifier}.jar"
        url="${BASE_URL}/${module}/${JCUDA_VERSION}/${jar_name}"
    else
        jar_name="${module}-${JCUDA_VERSION}.jar"
        url="${BASE_URL}/${module}/${JCUDA_VERSION}/${jar_name}"
    fi
    
    local target_path="${LIBS_DIR}/${jar_name}"
    
    if [ -f "$target_path" ]; then
        echo "  [SKIP] $jar_name (already exists)"
        return
    fi
    
    echo "  [DOWNLOAD] $jar_name..."
    if curl -fsSL -o "$target_path" "$url"; then
        echo "  [OK] $jar_name"
    else
        echo "  [FAIL] $jar_name"
    fi
}

echo "Downloading JCuda JARs..."
echo ""

for module in "${MODULES[@]}"; do
    if [[ "$module" == *"-natives" ]]; then
        # Native JARs need platform classifier
        download_jar "$module" "$OS_CLASSIFIER"
    else
        # Pure Java JARs
        download_jar "$module" ""
    fi
done

echo ""
echo "========================================"
echo " Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Ensure CUDA Toolkit 11.8+ is installed"
echo "  2. Run: ./gradlew build"
echo "  3. Run: ./gradlew test"
echo ""
echo "Troubleshooting:"
echo "  - CUDA_ERROR_NO_DEVICE: No NVIDIA GPU detected"
echo "  - CUDA_ERROR_INVALID_PTX: CUDA version mismatch"
echo "  - UnsatisfiedLinkError: Missing CUDA runtime libraries"
echo ""
