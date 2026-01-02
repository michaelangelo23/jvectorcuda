# JVectorCUDA
A Java library for GPU-accelerated vector similarity search with automatic CPU fallback.

```
This library idea was created during the creation of "JavaLlama", a school project for OOP.
```

## Overview
JVectorCUDA provides CUDA-accelerated vector search for Java applications. When a GPU is unavailable, it automatically falls back to CPU-based search using JVector.

## Features
- GPU acceleration using JCUDA
- Automatic CPU fallback
- Single JAR deployment
- JavaFX 3D visualization
- Multiple distance metrics (Euclidean, Cosine, Inner Product)

## Quick Start

### Installation
**Gradle:**
```gradle
dependencies {
    implementation 'com.vindex:jvectorcuda:1.0.0'
}
```

**Maven:**
```xml
<dependency>
    <groupId>com.vindex</groupId>
    <artifactId>jvectorcuda</artifactId>
    <version>1.0.0</version>
</dependency>
```

### Basic Usage
```java
import com.vindex.jvectorcuda.VectorIndexFactory;
import com.vindex.jvectorcuda.VectorIndex;
import com.vindex.jvectorcuda.SearchResult;
import com.vindex.jvectorcuda.DistanceMetric;

// Create index (auto-detects GPU)
try (VectorIndex index = VectorIndexFactory.auto(384)) {
    // Add vectors (uploads to GPU once)
    float[][] vectors = ...;
    index.add(vectors);

    // Search (runs on GPU, 5x faster than CPU)
    float[] query = ...;
    SearchResult result = index.search(query, 10);
}

// With distance metric
try (VectorIndex index = VectorIndexFactory.auto(384, DistanceMetric.COSINE)) {
    // Cosine similarity for text embeddings
}
```

### Visualization
```java
import com.vindex.jvectorcuda.visualization.Visualizer;
Visualizer.show3D(index, result);
```

## Requirements
- Java 21+ (tested on Java 25)
- For GPU: NVIDIA GPU with CUDA Compute 6.1+ (GTX 1060 or newer)
- CUDA Toolkit 11.8+ (for GPU support)
- For CPU fallback: Any x64 processor

## Current Status

**Phase:** Phase 5 Complete - Developer Tools

**Completed Phases:**
- Phase 1: Foundation & Setup ✅
- Phase 2: Proof of Concept (CUDA kernels) ✅
- Phase 3: Core Vector Search (GPU/CPU implementations) ✅
- Phase 4: Distance Metrics (Euclidean, Cosine, Inner Product) ✅
- Phase 5: Developer Tools (Benchmarking Framework) ✅

**Test Suite:** 110 tests passing

**Current Focus:**
- Phase 6: JavaFX 3D Visualization
- Gathering GPU performance data from RTX GPUs

### Performance Benchmarks (GTX 1080 Max-Q)

**Single Query (fresh upload each time):**
| Dataset | CPU | GPU | Speedup | Winner |
|---------|-----|-----|---------|--------|
| 50K vectors × 384D | 28ms | 61ms | 0.46x | CPU |

**Persistent Memory (upload once, query many):**
| Dataset | Queries | CPU | GPU | Speedup | Winner |
|---------|---------|-----|-----|---------|--------|
| 50K vectors × 384D | 100 | 2857ms | 525ms | **5.44x** | GPU |

**Key Finding:** GPU wins with **persistent memory architecture**:
- Upload database to GPU once
- Run many queries without re-uploading
- Achieves 5x+ speedup over CPU

**Benchmarking Framework:** Use `BenchmarkFramework` to measure your own hardware.

### Strategic Direction: Hybrid Architecture

Based on POC results and market research:

**Goal:** Build first Java library with **intelligent CPU/GPU routing**
- Automatically uses GPU when beneficial
- Falls back to CPU when faster
- Learns optimal routing from workload patterns

**Why Hybrid:**
- Works on ALL hardware (not just datacenter GPUs)
- Graceful degradation (always functional)
- Unique market position (no competitors do this)
- Future-proof (better GPUs = automatic benefit)

## Contributing

We need help validating GPU performance on modern hardware!

If you have an RTX 2000/3000/4000 series GPU, please:
1. Run our portable benchmark: `./gradlew benchmark`
2. Copy the generated `benchmark-report.md` and share via GitHub Issues

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions.

**Why Your Help Matters:**
- Determines adaptive routing thresholds
- Validates GPU acceleration viability
- Guides performance optimization priorities

## Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute and run benchmarks

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Transparency

This project uses AI assistance for development with full transparency:
- Performance benchmarks published (even when GPU loses)
- Strategic pivots based on real data, not assumptions

**Current assessment:** GPU acceleration **validated** with persistent memory architecture achieving 5x+ speedup.

## Benchmarking Your Hardware

```java
import com.vindex.jvectorcuda.benchmark.*;

// Quick benchmark
BenchmarkFramework framework = new BenchmarkFramework();
BenchmarkResult result = framework.run(BenchmarkConfig.DEFAULT);
System.out.println(result.toSummaryString());

// Run full suite
List<BenchmarkResult> results = framework.runSuite(
    List.of(10_000, 50_000, 100_000),
    List.of(128, 384, 768)
);
framework.printComparisonTable(results);
```
