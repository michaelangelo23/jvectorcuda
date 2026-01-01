# JVectorCUDA

A Java library for GPU-accelerated vector similarity search with automatic CPU fallback.

## Overview

JVectorCUDA provides CUDA-accelerated vector search for Java applications. When a GPU is unavailable, it automatically falls back to CPU-based search using JVector.

## Features

- GPU acceleration using JCuda
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

## Status

This library is currently under development. Performance benchmarks and documentation will be provided once the implementation is complete.

## License

MIT License - See [LICENSE](LICENSE)
