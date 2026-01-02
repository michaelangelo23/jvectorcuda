# JVectorCUDA

[![CI Build](https://github.com/michaelangelo23/jvectorcuda/actions/workflows/ci.yml/badge.svg)](https://github.com/michaelangelo23/jvectorcuda/actions/workflows/ci.yml)
[![CodeQL](https://github.com/michaelangelo23/jvectorcuda/actions/workflows/codeql.yml/badge.svg)](https://github.com/michaelangelo23/jvectorcuda/actions/workflows/codeql.yml)
[![License: Apache 2.0](https://img.shields.io/github/license/michaelangelo23/jvectorcuda)](LICENSE)

![Java](https://img.shields.io/badge/Java-17--25-ED8B00?style=flat&logo=openjdk&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-76B900?style=flat&logo=nvidia&logoColor=white)
![Gradle](https://img.shields.io/badge/Gradle-9.0+-02303A?style=flat&logo=gradle&logoColor=white)
![JCuda](https://img.shields.io/badge/JCuda-12.0.0-76B900?style=flat)

**CUDA-accelerated vector similarity search for Java.** 5x+ speedup for batch queries on NVIDIA GPUs, with automatic CPU fallback.

- GPU-accelerated nearest neighbor search using CUDA
- Intelligent routing: single queries → CPU, batch queries → GPU
- Multiple distance metrics: Euclidean, Cosine, Inner Product
- Works without GPU (falls back to CPU)

## Requirements

- Java 17+
- NVIDIA GPU with driver 11.8+ (run `nvidia-smi` to check)
- No CUDA Toolkit needed, just the driver

**Supported GPUs:** GTX 1060+ / RTX series / Tesla P4+ (Compute Capability 6.1+)

## Quick Start

```bash
git clone https://github.com/michaelangelo23/jvectorcuda.git
cd jvectorcuda

# Download JCuda dependencies
.\scripts\setup-jcuda.ps1   # Windows
./scripts/setup-jcuda.sh    # Linux/macOS

# Build and test
./gradlew build test
```

## Usage

```java
import com.vindex.jvectorcuda.*;

// Recommended: Hybrid index (auto-routes to CPU or GPU)
try (VectorIndex index = VectorIndexFactory.hybrid(384)) {
    index.add(embeddings);
    
    // Single query → CPU (low latency)
    SearchResult result = index.search(query, 10);
    
    // Batch queries → GPU (5x+ speedup)
    List<SearchResult> results = index.searchBatch(queries, 10);
}
```

**Other options:**
```java
VectorIndexFactory.auto(384);   // GPU if available, else CPU
VectorIndexFactory.cpu(384);    // Force CPU
VectorIndexFactory.gpu(384);    // Force GPU
```

## Performance

Tested on GTX 1080 Max-Q (384 dimensions, 50K vectors):

| Mode | GPU vs CPU |
|------|------------|
| Single query | 0.5x (CPU wins) |
| Batch 100 queries | **5.5x speedup** |

GPU wins when you batch queries or keep data in GPU memory.

## Thread Safety

Not thread-safe by default. For concurrent access:

```java
VectorIndex index = VectorIndexFactory.hybridThreadSafe(384);
```

## License

Apache 2.0
