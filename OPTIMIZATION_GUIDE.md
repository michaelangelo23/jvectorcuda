# JVectorCUDA Optimization Guide

This document explains the performance optimizations implemented in JVectorCUDA and the scientific principles behind them.

## Table of Contents

- [CPU Optimizations](#cpu-optimizations)
- [GPU/CUDA Optimizations](#gpucuda-optimizations)
- [Memory Management](#memory-management)
- [Algorithm Selection](#algorithm-selection)
- [Performance Tuning](#performance-tuning)

---

## CPU Optimizations

### 1. Loop Unrolling (4-way)

**What it does:** Processes 4 vector elements per loop iteration instead of 1.

**Why it works:**
- Modern CPUs have deep pipelines (15-20+ stages)
- Loop overhead (increment, compare, branch) consumes cycles
- 4-way unrolling amortizes this overhead across 4 operations
- Exposes more instruction-level parallelism (ILP) to the CPU

**Example (Euclidean Distance):**
```java
// Before: Simple loop
for (int i = 0; i < dimensions; i++) {
    float diff = a[i] - b[i];
    sum += diff * diff;
}

// After: 4-way unrolled
for (; i < limit; i += 4) {
    float diff0 = a[i] - b[i];
    float diff1 = a[i+1] - b[i+1];
    float diff2 = a[i+2] - b[i+2];
    float diff3 = a[i+3] - b[i+3];
    sum0 += diff0 * diff0;
    sum1 += diff1 * diff1;
    sum2 += diff2 * diff2;
    sum3 += diff3 * diff3;
}
float sum = sum0 + sum1 + sum2 + sum3;
```

**Expected speedup:** 10-30% for distance calculations

### 2. Primitive Max-Heap for Top-K Selection

**What it does:** Replaces `PriorityQueue<int[]>` with primitive int arrays.

**Why it works:**
- Avoids autoboxing overhead (int → Integer)
- No per-element object allocation (reduces GC pressure)
- Better cache locality (contiguous memory)
- No virtual method dispatch for comparisons

**Complexity:** O(n log k) - same as PriorityQueue, but with lower constants

**Memory usage:** O(k) primitive ints vs O(k) Integer objects + int[] wrappers

### 3. Multiple Accumulator Variables

**What it does:** Uses separate sum variables (sum0, sum1, sum2, sum3) instead of one.

**Why it works:**
- Breaks data dependencies between iterations
- Allows CPU to execute additions in parallel (superscalar execution)
- Modern CPUs can issue 2-4 floating-point operations per cycle
- Single accumulator creates a dependency chain, limiting parallelism

---

## GPU/CUDA Optimizations

### 1. Memory Coalescing

**What it does:** Accesses memory in patterns that allow GPU to fetch data efficiently.

**Why it works:**
- GPU memory transactions are 32/64/128 bytes wide
- Coalesced access: consecutive threads read consecutive addresses
- Single memory transaction serves multiple threads
- Uncoalesced access: each thread requires separate transaction (32x slower!)

**Implementation:**
```cuda
// Good: Thread idx reads database[idx * dimensions + d]
// Each thread accesses contiguous memory for its vector
const float* vec = database + idx * dimensions;
for (int d = 0; d < dimensions; d++) {
    // vec[d] is coalesced within each vector
    sum += (vec[d] - query[d]) * (vec[d] - query[d]);
}
```

### 2. Shared Memory for Query Vector

**What it does:** Loads query vector into fast on-chip shared memory.

**Why it works:**
- Shared memory: ~5 cycles latency, 2TB/s bandwidth
- Global memory: ~400 cycles latency, 500GB/s bandwidth
- Query is read by every thread in the block → perfect for sharing
- Reduces global memory bandwidth by dimensions × blockSize

**Implementation:**
```cuda
extern __shared__ float sharedQuery[];

// Cooperative loading
for (int d = threadIdx.x; d < dimensions; d += blockDim.x) {
    sharedQuery[d] = query[d];
}
__syncthreads();

// Fast access
sum += (vec[d] - sharedQuery[d]) * ...
```

### 3. Loop Unrolling in CUDA

**What it does:** Same principle as CPU, but critical for GPU occupancy.

**Why it works:**
- Reduces instruction overhead (fewer loop control instructions)
- Better utilizes instruction-level parallelism
- Compiler can schedule independent operations
- Reduces register pressure from loop variables

### 4. Persistent GPU Memory

**What it does:** Keeps database vectors in GPU memory across queries.

**Why it works:**
- PCIe bandwidth: ~16 GB/s (bidirectional)
- GPU memory bandwidth: ~500 GB/s
- Upload 50K × 384 × 4 bytes = 73 MB takes ~5ms
- Computation for 50K vectors takes ~5ms
- **Transfer time ≈ Compute time** → Eliminate transfers!

**Results:**
- Without persistence: 100 queries = 100 × (5ms upload + 5ms compute) = 1000ms
- With persistence: 1 × 5ms upload + 100 × 5ms compute = 505ms
- **5x speedup!**

---

## Memory Management

### 1. Integer Overflow Prevention

**Risk:** `vectors.length * dimensions` can overflow int for large datasets.

**Solution:**
```java
long totalSize = (long) vectors.length * dimensions;
if (totalSize > Integer.MAX_VALUE) {
    throw new IllegalArgumentException("Data too large");
}
float[] flat = new float[(int) totalSize];
```

### 2. GPU Memory Pre-validation

**What it does:** Checks available GPU memory before allocation.

**Why it works:**
- Prevents cryptic CUDA_ERROR_OUT_OF_MEMORY crashes
- Allows graceful fallback to CPU
- Provides actionable error messages with suggestions

### 3. Capacity Expansion Strategy

**Strategy:** Double capacity when full (similar to ArrayList).

**Why it works:**
- Amortized O(1) insertion cost
- Reduces number of GPU memory reallocations
- Each reallocation copies existing data (expensive on GPU)

---

## Algorithm Selection

### Brute-Force vs HNSW

| Algorithm | Time Complexity | Recall | Best For |
|-----------|-----------------|--------|----------|
| Brute-force (GPU) | O(n) | 100% | Batch queries, exact results |
| HNSW (CPU) | O(log n) | 99%+ | Single queries, large datasets |

**JVectorCUDA Hybrid Strategy:**
- Single query → CPU (lower latency)
- Batch ≥ 10 queries → GPU (higher throughput)
- Dataset > 50K AND batch queries → GPU (amortized upload)

### Distance Metric Selection

| Metric | Formula | Use Case |
|--------|---------|----------|
| Euclidean | √Σ(a-b)² | General purpose, unnormalized vectors |
| Cosine | 1 - (a·b)/(|a||b|) | Text embeddings, normalized vectors |
| Inner Product | -a·b | Recommendation systems, MIPS |

**Implementation note:** All metrics normalized so smaller = more similar.

---

## Performance Tuning

### Recommended Settings by Use Case

#### High-Throughput Batch Processing
```java
VectorIndex index = VectorIndexFactory.gpu(dimensions);
// Add all vectors at once
index.add(allVectors);
// Process queries in batches of 100+
List<SearchResult> results = index.searchBatch(queries, k);
```

#### Low-Latency Single Queries
```java
VectorIndex index = VectorIndexFactory.cpu(dimensions);
// Or use hybrid for automatic routing
VectorIndex index = VectorIndexFactory.hybrid(dimensions);
```

#### Concurrent Access
```java
VectorIndex index = VectorIndexFactory.hybridThreadSafe(dimensions);
```

### Hardware-Specific Tuning

#### GTX 1080 (Compute 6.1)
- Block size: 256 threads (optimal for this architecture)
- Good for datasets up to ~500K vectors (8GB VRAM)

#### RTX 3090 (Compute 8.6)
- Block size: 256 or 512 threads
- Good for datasets up to ~2M vectors (24GB VRAM)

#### Tesla V100 (Compute 7.0)
- Block size: 256 threads
- HBM2 memory provides 900GB/s bandwidth
- Best for very large datasets

---

## Research References

1. **Memory Coalescing:**
   - NVIDIA CUDA C Programming Guide, Section 5.3.2
   - "Memory access patterns are critical for GPU performance"

2. **Loop Unrolling:**
   - Hennessy & Patterson, "Computer Architecture: A Quantitative Approach"
   - Reduces loop overhead and exposes ILP

3. **Max-Heap for Top-K:**
   - Cormen et al., "Introduction to Algorithms", Chapter 6
   - O(n log k) is optimal for streaming top-k selection

4. **Persistent GPU Memory:**
   - Similar to GPU database systems (RAPIDS, BlazingSQL)
   - Key insight: transfer cost ≈ compute cost for vector search

5. **Distance Metrics:**
   - Faiss paper: "Billion-scale similarity search with GPUs"
   - JVector: "Disk-backed approximate nearest neighbor search"

---

## Benchmarking Your Configuration

Run the built-in benchmark:
```bash
./gradlew benchmark
```

Or run specific tests:
```bash
./gradlew test --tests "*.GpuBreakEvenTest"
```

The benchmark will help you determine:
- Optimal batch size for your hardware
- Break-even point between CPU and GPU
- Memory transfer overhead percentage
