# JVectorCUDA
A Java library for GPU-accelerated vector similarity search with automatic CPU fallback.

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
import com.vindex.jvectorcuda.VectorIndex;
import com.vindex.jvectorcuda.SearchResult;

// Create index (auto-detects GPU)
VectorIndex index = VectorIndex.auto(384);

// Add vectors
float[][] vectors = ...;
index.add(vectors);

// Search
float[] query = ...;
SearchResult result = index.search(query, 10);
```

### Visualization
```java
import com.vindex.jvectorcuda.visualization.Visualizer;
Visualizer.show3D(index, result);
```

## Requirements
- Java 21+
- For GPU: NVIDIA GPU with CUDA Compute 6.1+ (GTX 1060 or newer)
- For CPU fallback: Any x64 processor

## Current Status

**Phase:** Proof of Concept - Validation & Data Gathering

**Completed POCs:**
- POC #1: CUDA Detection (GTX 1080 Max-Q validated)
- POC #2: Vector Addition Kernel (infrastructure working)
- POC #3: Euclidean Distance Kernel (performance validated)

**Current Focus:**
- Gathering GPU performance data from RTX GPUs
- Finding break-even points for GPU acceleration
- Designing adaptive CPU/GPU routing strategy

### Performance Benchmarks (GTX 1080 Max-Q)

**POC #2 - Vector Addition:**
| Dataset | CPU | GPU | Speedup | Winner |
|---------|-----|-----|---------|--------|
| 1M elements | 5.49ms | 6.13ms | 0.90x | CPU |

**POC #3 - Euclidean Distance:**
| Dataset | CPU | GPU | Speedup | Winner |
|---------|-----|-----|---------|--------|
| 50K vectors × 384D | 28ms | 61ms | 0.46x | CPU |

**Key Finding:** GTX 1080 Max-Q (mobile GPU) is slower than modern CPUs due to:
- JNI overhead (~10-15ms per call)
- Memory transfer costs (Host to Device copies)
- Thermal throttling (mobile GPU limitation)

**Next Step:** Validate on modern GPUs (RTX 3000/4000 series) to find when GPU wins.

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
1. Run our benchmark suite: `./gradlew test --tests GpuBreakEvenTest`
2. Share results via GitHub Issues

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions.

**Why Your Help Matters:**
- Determines adaptive routing thresholds
- Validates GPU acceleration viability
- Guides performance optimization priorities

## Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md) - How to test and contribute
- `POC_LOG.md` - Detailed POC results and findings
- `PROBLEMS.md` - Issues encountered and solutions
- `task.md` - Current development progress

## License

Apache 2.0 - See LICENSE file for details

## Transparency

This project uses AI assistance for development. All POC results are documented with complete transparency:
- Performance benchmarks published (even when GPU loses)
- Issues and failures logged in PROBLEMS.md
- Strategic pivots based on real data, not assumptions

**Current assessment:** GPU acceleration viable with hybrid approach and cuVS integration (viability: 8.5/10)
