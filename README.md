# JVectorCUDA

> This library idea was created during the creation of "JavaLlama", a school project for OOP.

GPU-accelerated vector similarity search for Java with automatic CPU fallback.

## Features

- CUDA-accelerated brute-force vector search
- Automatic CPU fallback when GPU unavailable
- Multiple distance metrics: Euclidean, Cosine Similarity, Inner Product
- Persistent memory mode for batch queries (5x+ GPU speedup)
- Portable benchmarking tool for hardware validation

## Requirements

- Java 21+
- Gradle 8.0+
- NVIDIA GPU with CUDA Compute 6.1+ (GTX 1060 or newer)
- CUDA Toolkit 11.8+

## Installation

```gradle
dependencies {
    implementation 'com.vindex:jvectorcuda:1.0.0'
}
```

## Usage

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

## Build

```bash
./gradlew build
./gradlew test
```

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
| Single Query | 50K | 1 | 97ms | 186ms | 0.52x |
| Persistent | 50K | 100 | 2901ms | 1004ms | 2.89x |

GPU wins with persistent memory (upload once, query many).

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
