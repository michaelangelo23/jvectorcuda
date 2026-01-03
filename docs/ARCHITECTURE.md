# Architecture

This document describes the internal architecture and design decisions of JVectorCUDA.

## Table of Contents

- [Overview](#overview)
- [Core Components](#core-components)
- [Index Implementations](#index-implementations)
- [GPU Architecture](#gpu-architecture)
- [Benchmarking Framework](#benchmarking-framework)
- [Design Decisions](#design-decisions)
- [Performance Considerations](#performance-considerations)

## Overview

JVectorCUDA is a GPU-accelerated vector similarity search library for Java that provides automatic CPU fallback when GPU is unavailable.

### Key Design Principles

1. **Simple API**: Consistent interface across CPU and GPU implementations
2. **Automatic Fallback**: Graceful degradation when GPU is unavailable
3. **Intelligent Routing**: Hybrid index automatically routes queries optimally
4. **Memory Efficiency**: Persistent GPU memory for repeated queries
5. **Type Safety**: Compile-time safety with strong typing

## Core Components

### VectorIndex Interface

The central abstraction for all index implementations:

```java
public interface VectorIndex extends AutoCloseable {
    void add(float[][] vectors);
    SearchResult search(float[] query, int k);
    SearchResult search(float[][] queries, int k);
    int size();
    int getDimensions();
    void close();
}
```

**Design Rationale**: 
- Simple, focused API
- `AutoCloseable` for resource management
- Batch search support for GPU efficiency

### SearchResult

Immutable result container:

```java
public final class SearchResult {
    private final int[] ids;
    private final float[] distances;
}
```

**Design Rationale**:
- Immutable for thread safety
- Parallel arrays for cache locality
- Simple structure for easy serialization

### VectorIndexFactory

Factory for creating indices:

```java
public static VectorIndex createBestAvailable(int dimensions) {
    if (CudaDetector.isAvailable()) {
        return new HybridVectorIndex(dimensions);
    }
    return new CPUVectorIndex(dimensions);
}
```

**Design Rationale**:
- Centralized creation logic
- Automatic GPU detection
- Easy for users to get started

## Index Implementations

### CPUVectorIndex

**Strategy**: Brute-force linear search with JVector backend

**Characteristics**:
- Uses [JVector](https://github.com/jbellis/jvector) for CPU search
- Fast for small datasets (<10,000 vectors)
- Low latency for single queries
- No GPU dependency

**Use Cases**:
- Small datasets
- Single query requests
- CPU-only environments
- Low-latency requirements

**Implementation Details**:
```java
public class CPUVectorIndex implements VectorIndex {
    private final VectorSimilarityFunction similarity;
    private final ListRandomAccessVectorValues<float[]> rav;
    
    @Override
    public SearchResult search(float[] query, int k) {
        // Uses JVector's GraphSearcher for efficient search
        NeighborQueue queue = new NeighborQueue(k, false);
        // ... search implementation
    }
}
```

### GPUVectorIndex

**Strategy**: CUDA-accelerated brute-force search

**Characteristics**:
- Offloads computation to GPU
- Fast for large datasets (>10,000 vectors)
- High throughput for batch queries
- Requires NVIDIA GPU with CUDA

**Use Cases**:
- Large datasets
- Batch query processing
- High throughput requirements
- Available GPU hardware

**Implementation Details**:
```java
public class GPUVectorIndex implements VectorIndex {
    private CUdeviceptr deviceDatabase;
    private CUdeviceptr deviceDistances;
    private CUfunction kernelFunction;
    
    @Override
    public void add(float[][] vectors) {
        // Upload to GPU memory (one-time cost)
        cuMemAlloc(deviceDatabase, size);
        cuMemcpyHtoD(deviceDatabase, hostDatabase, size);
    }
    
    @Override
    public SearchResult search(float[] query, int k) {
        // Launch CUDA kernel
        cuLaunchKernel(kernelFunction, ...);
        // Copy results back
        cuMemcpyDtoH(results, deviceDistances, size);
    }
}
```

**Memory Management**:
- Persistent GPU memory allocation
- Database uploaded once, reused for multiple queries
- Distances buffer reused across searches

### HybridVectorIndex

**Strategy**: Intelligent routing between CPU and GPU

**Characteristics**:
- Automatically selects best backend
- Falls back to CPU when GPU unavailable
- Routes single queries to CPU, batches to GPU
- Adapts based on dataset size

**Routing Logic**:
```java
public SearchResult search(float[] query, int k) {
    if (!gpuAvailable || vectorCount < VECTOR_THRESHOLD) {
        return cpuIndex.search(query, k);
    }
    
    // Single query: CPU is faster (lower latency)
    return cpuIndex.search(query, k);
}

public SearchResult search(float[][] queries, int k) {
    if (!gpuAvailable || vectorCount < VECTOR_THRESHOLD) {
        return cpuIndex.search(queries, k);
    }
    
    // Batch queries: GPU is faster (higher throughput)
    if (queries.length >= BATCH_THRESHOLD) {
        return gpuIndex.search(queries, k);
    }
    
    return cpuIndex.search(queries, k);
}
```

**Thresholds**:
- `BATCH_THRESHOLD = 10`: Minimum queries for GPU batching
- `VECTOR_THRESHOLD = 50,000`: Minimum dataset size for GPU

**Use Cases**:
- Production deployments (automatic fallback)
- Mixed workloads (single + batch queries)
- Uncertain GPU availability

### ThreadSafeVectorIndex

**Strategy**: Synchronized wrapper around any VectorIndex

**Characteristics**:
- Thread-safe access to underlying index
- Simple synchronized methods
- Minimal overhead

**Implementation**:
```java
public class ThreadSafeVectorIndex implements VectorIndex {
    private final VectorIndex delegate;
    
    @Override
    public synchronized SearchResult search(float[] query, int k) {
        return delegate.search(query, k);
    }
}
```

**Design Rationale**:
- Decorator pattern for flexibility
- Explicit opt-in for thread safety
- Can wrap any VectorIndex implementation

## GPU Architecture

### CUDA Kernel Design

The GPU implementation uses custom CUDA kernels for distance computation:

**Euclidean Distance Kernel**:
```cuda
__global__ void euclideanDistanceKernel(
    const float* database,      // All vectors in database
    const float* query,          // Query vector
    float* distances,            // Output distances
    int numVectors,              // Number of database vectors
    int dimensions               // Vector dimensions
) {
    int vectorId = blockIdx.x * blockDim.x + threadIdx.x;
    if (vectorId >= numVectors) return;
    
    float sum = 0.0f;
    for (int d = 0; d < dimensions; d++) {
        float diff = database[vectorId * dimensions + d] - query[d];
        sum += diff * diff;
    }
    
    distances[vectorId] = sum;  // Squared distance
}
```

**Kernel Parameters**:
- Grid size: `(numVectors + 255) / 256` blocks
- Block size: 256 threads per block
- Shared memory: Not currently used (future optimization)

### Memory Layout

**Database Memory**:
```
[v0_d0, v0_d1, ..., v0_dn, v1_d0, v1_d1, ..., v1_dn, ...]
```
- Row-major layout for coalesced memory access
- Contiguous allocation for entire database

**Distance Memory**:
```
[dist_v0, dist_v1, dist_v2, ..., dist_vn]
```
- One distance per vector
- Sorted on CPU after copy back

### GPU Workflow

1. **Initialization**: 
   - Load PTX kernels from resources
   - Initialize CUDA context
   - Allocate GPU memory

2. **Add Vectors**:
   - Allocate device memory
   - Copy vectors to GPU (HtoD)
   - Keep in GPU memory

3. **Search**:
   - Copy query to GPU (HtoD)
   - Launch distance kernel
   - Copy distances back (DtoH)
   - Sort on CPU
   - Return top-k results

4. **Cleanup**:
   - Free device memory
   - Destroy context

### PTX Kernels

Pre-compiled PTX files stored in `src/main/resources/kernels/`:
- `euclidean_distance.ptx`
- `cosine_similarity.ptx`
- `inner_product.ptx`

**Design Rationale**:
- Pre-compiled PTX for cross-platform compatibility
- No runtime compilation overhead
- Works without NVCC on client machines

## Benchmarking Framework

### Architecture

```
BenchmarkConfig
     ↓
BenchmarkFramework ← StandardBenchmarkSuite
     ↓                       ↓
BenchmarkResult → ComprehensiveBenchmarkResult
                           ↓
                  BenchmarkResultExporter
```

### Components

**BenchmarkConfig**: Immutable configuration
- Vector count, dimensions, k
- Warmup and measurement iterations
- Distance metric
- Memory profiling flags

**BenchmarkFramework**: Core benchmark execution
- CPU vs GPU comparison
- Persistent memory mode
- Warmup handling
- Timing measurement

**StandardBenchmarkSuite**: High-level benchmarking
- Dataset management
- Percentile metrics
- Memory tracking
- Comprehensive results

**PercentileMetrics**: Latency analysis
- P50, P95, P99 calculation
- Min/max tracking
- Statistical analysis

**MemoryMetrics**: Memory profiling
- Heap usage tracking
- Off-heap (GPU) memory
- Memory pressure detection

**BenchmarkResultExporter**: Result persistence
- CSV export
- JSON export
- Append mode for continuous monitoring

## Design Decisions

### Why Brute-Force Search?

**Decision**: Use brute-force linear search instead of approximate methods (HNSW, IVF)

**Rationale**:
- Exact results (100% recall)
- Simple implementation
- GPU parallelization is straightforward
- Fast enough for moderate datasets
- Foundation for future optimizations

**Trade-offs**:
- O(n) complexity per query
- Not suitable for very large datasets (>1M vectors)
- Future: Can add approximate methods on top

### Why JVector for CPU Backend?

**Decision**: Use JVector library instead of custom CPU implementation

**Rationale**:
- Mature, well-tested implementation
- Optimized Java code
- Supports multiple distance metrics
- Active maintenance
- Pure Java (no JNI)

**Alternative Considered**: 
- Custom brute-force: More control, but more code to maintain
- FAISS with JNI: More features, but complex dependencies

### Why Separate CPU and GPU Indices?

**Decision**: Separate `CPUVectorIndex` and `GPUVectorIndex` classes

**Rationale**:
- Clear separation of concerns
- Easy to test independently
- Users can choose explicitly
- HybridVectorIndex composes both

**Alternative Considered**:
- Single class with runtime switching: More complex, harder to test

### Why Persistent GPU Memory?

**Decision**: Keep database in GPU memory across searches

**Rationale**:
- Amortizes upload cost
- 5-10x speedup for multiple queries
- Realistic production scenario
- Trades memory for speed

**Trade-offs**:
- Higher memory usage
- Requires explicit close() call
- May not fit in GPU memory

### Why Builder Pattern?

**Decision**: Use builders for `BenchmarkConfig` and `BenchmarkResult`

**Rationale**:
- Many optional parameters
- Type-safe construction
- Immutable results
- Easy to extend

**Alternative Considered**:
- Constructors with many parameters: Less readable
- Mutable objects: Not thread-safe

## Performance Considerations

### When GPU is Faster

GPU shows speedup when:
1. **Large datasets**: >10,000 vectors
2. **Batch queries**: ≥10 queries at once
3. **High dimensions**: >256 dimensions
4. **Persistent memory**: Same dataset, many queries

### When CPU is Faster

CPU shows better performance when:
1. **Small datasets**: <5,000 vectors
2. **Single queries**: One query at a time
3. **Low latency**: Need immediate results
4. **Cold start**: First query on dataset

### Memory Bottlenecks

**PCIe Transfer**:
- Uploading database: Major cost for first query
- Query transfer: Negligible (small size)
- Result transfer: Moderate (k × sizeof(int + float))

**Optimization Strategies**:
- Keep database in GPU (persistent memory)
- Batch queries to amortize transfer
- Use HybridVectorIndex for intelligent routing

### Kernel Optimization

**Current**:
- Simple parallel distance computation
- No shared memory
- No warp-level optimizations

**Future Optimizations**:
- Shared memory for query vector
- Warp shuffle for reductions
- Kernel fusion (distance + top-k)
- Tensor cores for matrix operations

## Testing Strategy

### Unit Tests

- `VectorIndexTest`: Basic functionality
- `ParameterizedVectorIndexTest`: Comprehensive coverage
- `DistanceMetricTest`: Distance computation
- `CudaAvailabilityTest`: GPU detection

### Integration Tests

- `IntegrationTest`: End-to-end workflows
- `HybridVectorIndexTest`: Routing logic
- `ThreadSafeVectorIndexTest`: Concurrency

### Benchmark Tests

- `BenchmarkFrameworkTest`: Benchmark correctness
- `GpuBreakEvenTest`: Performance breakpoints

### Testing Philosophy

1. **Test all paths**: CPU, GPU, Hybrid
2. **Test edge cases**: Empty, single vector, k > count
3. **Test correctness**: CPU vs GPU results match
4. **Test performance**: Benchmarks catch regressions

## Future Enhancements

### Approximate Search

- HNSW graph for CPU
- IVF (Inverted File Index) for GPU
- Trade recall for speed

### Advanced GPU Features

- Multi-GPU support
- Kernel fusion (distance + top-k in one kernel)
- Quantization (int8, binary)
- Batch processing pipeline

### API Improvements

- Async search API
- Streaming results
- Update/delete support
- Metadata filtering

### Persistence

- Save/load index to disk
- Memory-mapped files
- Incremental updates

## See Also

- [BENCHMARKING.md](BENCHMARKING.md) - Performance testing guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guide
- [README.md](README.md) - User guide
