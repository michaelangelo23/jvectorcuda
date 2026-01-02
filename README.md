# JVectorCUDA

[![CI Build](https://github.com/michaelangelo23/jvectorcuda/actions/workflows/ci.yml/badge.svg)](https://github.com/michaelangelo23/jvectorcuda/actions/workflows/ci.yml)
[![CodeQL](https://github.com/michaelangelo23/jvectorcuda/actions/workflows/codeql.yml/badge.svg)](https://github.com/michaelangelo23/jvectorcuda/actions/workflows/codeql.yml)
[![License: Apache 2.0](https://img.shields.io/github/license/michaelangelo23/jvectorcuda)](LICENSE)

![Java](https://img.shields.io/badge/Java-17--25-ED8B00?style=flat&logo=openjdk&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-76B900?style=flat&logo=nvidia&logoColor=white)
![Gradle](https://img.shields.io/badge/Gradle-9.0+-02303A?style=flat&logo=gradle&logoColor=white)
![JVector](https://img.shields.io/badge/JVector-3.0.6-blue?style=flat)
![JCuda](https://img.shields.io/badge/JCuda-12.0.0-76B900?style=flat)

> **CUDA-powered vector similarity search for Java applications (requires NVIDIA GPU).**  
> Delivers 5x+ speedup for batch queries with persistent GPU memory. CPU fallback remains available for single-query scenarios or non-GPU systems.

This project was originally prototyped during the creation of "JavaLlama".

## Features

- **CUDA-accelerated brute-force vector search** (requires NVIDIA GPU)
- **Automatic CPU fallback** when GPU is unavailable or for single-query low-latency scenarios
- **Multiple distance metrics:** Euclidean, Cosine Similarity, Inner Product
- **Persistent GPU memory mode** for batch queries — upload once, query many times for 5x+ speedup
- **Portable benchmarking tool** to validate hardware performance and find break-even points

## When to Use JVectorCUDA

**Best suited for you if you have:**
- NVIDIA GPU with CUDA support (GTX 1060 or newer)
- Batch query workloads that reuse the same dataset (persistent GPU memory)
- Java applications where embedding search latency and throughput are critical

**Consider a CPU-only library instead if:**
- You're running single, one-off queries where upload latency will dominate
- Your environment uses non-NVIDIA GPUs (Intel/AMD) or lacks CUDA drivers
- You need the lowest possible memory footprint

## Requirements

- Java 17-25 (tested with OpenJDK Temurin)
- Gradle 9.0+
- NVIDIA GPU with CUDA Compute 6.1+ (GTX 1060 or newer)
- CUDA Toolkit 11.8+

## Quick Start

### 1. Clone and Setup JCuda Dependencies

```bash
git clone https://github.com/michaelangelo23/jvectorcuda.git
cd jvectorcuda
```

**Windows (PowerShell):**
```powershell
.\scripts\setup-jcuda.ps1
```

**Linux/macOS:**
```bash
chmod +x scripts/setup-jcuda.sh
./scripts/setup-jcuda.sh
```

This downloads JCuda 12.0.0 JARs to the `libs/` directory.

### 2. Install CUDA Toolkit

Download [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) (recommended for GTX 10xx/20xx/30xx compatibility).

### 3. Build and Test

```bash
./gradlew build
./gradlew test
```

## Installation (Maven/Gradle)

> **Note:** Not yet published to Maven Central. Use local build for now.

```gradle
dependencies {
    implementation 'com.vindex:jvectorcuda:1.0.0'
}
```

## Usage

### Single Query

```java
import com.vindex.jvectorcuda.*;

// Auto-detect GPU, fallback to CPU
try (VectorIndex index = VectorIndexFactory.auto(384)) {
    index.add(vectors);
    SearchResult result = index.search(query, 10);
}

// With specific distance metric
try (VectorIndex index = VectorIndexFactory.auto(384, DistanceMetric.COSINE)) {
    index.add(vectors);
    SearchResult result = index.search(query, 10);
}
```

### Batch Queries (Recommended for GPU)

For multiple queries, use `searchBatch()` to amortize GPU kernel launch overhead:

```java
// Single queries (inefficient on GPU - kernel launched per query)
for (float[] query : queries) {
    SearchResult result = index.search(query, 10);
}

// Batch queries (efficient - single kernel launch)
float[][] queries = ...; // Shape: [numQueries][dimensions]
List<SearchResult> results = index.searchBatch(queries, 10);
```

**GPU Performance Impact:**
- Single query: ~1-2ms latency per search (kernel launch overhead dominates)
- Batch query: ~0.1-0.2ms per query (overhead amortized across batch)
- **Speedup: 5-10x for batches of 10+ queries**

See [PERFORMANCE.md](PERFORMANCE.md) for detailed benchmarks.

## Important Limitations

### Thread Safety
**VectorIndex implementations are NOT thread-safe.** For concurrent access:
- Create separate index instances per thread
- Use external synchronization (locks)
- Avoid shared mutable state

Future versions may include a thread-safe wrapper.

### Algorithm Differences

CPU and GPU use different algorithms:

| Implementation | Algorithm | Search Type | Complexity |
|----------------|-----------|-------------|------------|
| **CPU** | JVector HNSW graph | Approximate | O(log n) |
| **GPU** | CUDA brute-force | Exact | O(n) |

**What this means:**
- CPU provides approximate results (99%+ recall with default settings)
- GPU guarantees exact nearest neighbors (100% recall)
- For most applications, approximate is sufficient and faster at scale

See [PERFORMANCE.md](PERFORMANCE.md) for detailed analysis.

## Build

```bash
./gradlew build
./gradlew test
```

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `CUDA_ERROR_NO_DEVICE` | No NVIDIA GPU found | Install NVIDIA drivers, check GPU in Device Manager |
| `CUDA_ERROR_INVALID_PTX` | CUDA version mismatch | Use CUDA 11.8 (not 12.x or 13.x for older GPUs) |
| `UnsatisfiedLinkError` | Missing JCuda natives | Run setup script: `.\scripts\setup-jcuda.ps1` |
| `ClassNotFoundException: jcuda` | JARs not in libs/ | Run setup script or manually download JCuda 12.0.0 |
| Build fails on GitHub Actions | Expected - JCuda JARs local only | CI uses `continue-on-error`, tests run locally |

## Benchmarking

Run portable GPU benchmark:

```bash
./gradlew benchmark
```

Generates `benchmark-report.md` with system info and performance results.

## Performance

Tested on GTX 1080 Max-Q (384 dimensions):

| Mode | Dataset | Queries | CPU | GPU | Speedup |
|------|---------|---------|-----|-----|---------|
| Single Query | 50K | 1 | 89ms | 175ms | 0.51x |
| Persistent | 50K | 100 | 2890ms | 523ms | **5.52x** |

GPU wins with persistent memory (upload once, query many). See [PERFORMANCE.md](PERFORMANCE.md) for comprehensive benchmarks and break-even analysis.

## Project Structure

```
src/main/java/com/vindex/jvectorcuda/
├── VectorIndex.java          # Core interface
├── VectorIndexFactory.java   # Auto GPU/CPU selection
├── DistanceMetric.java       # Euclidean, Cosine, Inner Product
├── cpu/CPUVectorIndex.java   # CPU implementation
├── gpu/GPUVectorIndex.java   # CUDA implementation
└── benchmark/                # Benchmarking framework

src/main/resources/kernels/
├── euclidean_distance.cu/.ptx
├── cosine_similarity.cu/.ptx
└── inner_product.cu/.ptx
```

## Status

In development.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for benchmark submission guidelines.

## License

Apache 2.0 - See [LICENSE](LICENSE)
