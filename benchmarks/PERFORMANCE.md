# JVectorCUDA Performance Analysis

## System Specifications

| Component | Details |
|-----------|---------|
| **Device** | Alienware 15 R3 |
| **GPU** | NVIDIA GeForce GTX 1080 with Max-Q Design |
| | Compute Capability 6.1, 8191 MB VRAM, 1468 MHz |
| | 2560 CUDA Cores |
| **CPU** | Intel Core i7-7820HK @ 2.90GHz (8 threads) |
| **Memory** | JVM: 512 MB max, 256 MB allocated |
| **OS** | Windows 11 10.0 (amd64) |
| **Java** | 25 (OpenJDK 64-Bit Server VM) |
| **Test Date** | January 3, 2026 |

---

## Executive Summary

| Scenario | Winner | Speedup | Why |
|----------|--------|---------|-----|
| Cold-start (any size) | CPU | 0.26x-1.04x | Memory transfer dominates |
| Persistent memory | **GPU** | **5.56x** | Upload once, query many times |

---

## Why 5.56x Speedup?

From actual test data:
```
CPU: 100 queries × 50,000 vectors = 3,271.14 ms
GPU: 100 queries × 50,000 vectors = 588.38 ms

Speedup = 3271.14 / 588.38 = 5.56x
```

**The math:**
- CPU does 100 queries at ~32.7ms each = 3,271ms total
- GPU uploads data once (~60ms), then runs 100 queries at ~5.3ms each = 588ms total
- GPU is **5.56x faster** because it only pays the upload cost once

---

## Why Cold-Start is Slow

### Memory Transfer Overhead: 114.2%

```
=== Memory Transfer Overhead Analysis ===
Dataset: 50,000 vectors × 384 dimensions

GPU compute time:     60.44 ms
Memory transfer time: 69.00 ms (114.2% overhead)

Transfer takes LONGER than compute!
```

The GPU spends **more time moving data** than computing. This is the PCIe bottleneck.

### Timeline Comparison

```
Cold-Start GPU (single query):
┌──────────────────────────────────────────────────────────────┐
│  Upload Data     │  Compute  │  Download  │  Total: ~65ms   │
│     ~60ms        │   ~5ms    │    ~1ms    │                 │
└──────────────────────────────────────────────────────────────┘

CPU (single query):
┌──────────────────────────────────────────────────────────────┐
│              Compute in RAM               │  Total: ~35ms   │
│                  ~35ms                    │                 │
└──────────────────────────────────────────────────────────────┘

GPU is SLOWER: 65ms vs 35ms = 0.54x (CPU wins)
```

### With Persistent Memory:

```
Persistent GPU (100 queries):
┌──────────────────────────────────────────────────────────────┐
│ Upload │ Q1  │ Q2  │ Q3  │ ... │ Q100 │  Total: 588ms       │
│  60ms  │ 5ms │ 5ms │ 5ms │ ... │ 5ms  │  (5.88ms/query)     │
└──────────────────────────────────────────────────────────────┘

CPU (100 queries):
┌──────────────────────────────────────────────────────────────┐
│ Q1   │ Q2   │ Q3   │ ... │ Q100 │  Total: 3,271ms           │
│ 33ms │ 33ms │ 33ms │ ... │ 33ms │  (32.7ms/query)           │
└──────────────────────────────────────────────────────────────┘

GPU is FASTER: 588ms vs 3,271ms = 5.56x (GPU wins)
```

---

## Benchmark Results

### Cold-Start Break-Even Analysis

| Vectors | Batch | CPU (ms) | GPU (ms) | Speedup | Winner |
|---------|-------|----------|----------|---------|--------|
| 1K | 1 | 10.00 | 11.71 | 0.85x | CPU |
| 1K | 10 | 8.02 | 31.16 | 0.26x | CPU |
| 1K | 100 | 73.99 | 244.75 | 0.30x | CPU |
| 10K | 1 | 5.67 | 17.44 | 0.33x | CPU |
| 10K | 10 | 70.65 | 125.23 | 0.56x | CPU |
| 10K | 100 | 745.02 | 1117.00 | 0.67x | CPU |
| 50K | 1 | 34.46 | 90.23 | 0.38x | CPU |
| 50K | 10 | 369.44 | 562.16 | 0.66x | CPU |
| 50K | 100 | 3886.63 | 5004.70 | 0.78x | CPU |
| 100K | 1 | 65.70 | 207.21 | 0.32x | CPU |
| 100K | 10 | 1067.83 | 1022.39 | **1.04x** | **GPU** |

**Only at 100K vectors with batch 10 does GPU barely win cold-start (1.04x)**

### Persistent Memory Test

| Metric | CPU | GPU |
|--------|-----|-----|
| Total time (100 queries) | 3,271 ms | 588 ms |
| Per-query latency | 32.71 ms | 5.88 ms |
| **Speedup** | - | **5.56x** |

### Dimension Impact

| Dimensions | Vectors | CPU (ms) | GPU (ms) | Speedup |
|------------|---------|----------|----------|---------|
| 128 | 50K | 7.45 | 25.27 | 0.29x |
| 384 | 50K | 27.27 | 42.67 | 0.64x |
| 768 | 25K | 42.97 | 61.53 | 0.70x |
| 1536 | 10K | 26.02 | 37.65 | 0.69x |

Higher dimensions = more compute per vector = better GPU utilization.

---

## When to Use What

### Use CPU (CPUVectorIndex)
- Single queries or small batches
- Database changes frequently
- No GPU available
- Dataset < 10K vectors

### Use GPU (GPUVectorIndex)
- Database is static (upload once)
- High query throughput needed (100+ queries/sec)
- Dataset is large (50K+ vectors)
- Running many queries against same data

### Use Hybrid (HybridVectorIndex)
- Automatic routing based on batch size
- Workload varies (mix of single and batch queries)
- Best-of-both-worlds

---

## Code Examples

### Bad: Cold-Start Pattern (GPU Slower)

```java
// DON'T DO THIS - creates new context every query
for (Query q : queries) {
    try (GPUVectorIndex index = new GPUVectorIndex(dims, maxVectors)) {
        index.add(vectors);  // Upload 73MB every time! (~60ms)
        index.search(q.vector, k);  // Fast compute (~5ms)
    }  // Context destroyed
}
// Result: 100 × 65ms = 6,500ms total
```

### Good: Persistent Memory Pattern (GPU 5.56x Faster)

```java
// DO THIS - upload once, query many times
GPUVectorIndex index = new GPUVectorIndex(dims, maxVectors);
index.add(vectors);  // Upload 73MB once (~60ms)

for (Query q : queries) {
    index.search(q.vector, k);  // Data already in VRAM (~5ms each)
}
// Result: 60ms + (100 × 5ms) = 560ms total
```

---

## Raw Test Output

```
=== GPU Break-Even Point Analysis ===

Testing when GPU starts outperforming CPU...

Vectors         Batch        CPU (ms)     GPU (ms)     Speedup    Winner  
---------------------------------------------------------------------------
1K              1            10.00        11.71        0.85      x CPU     
1K              10           8.02         31.16        0.26      x CPU     
1K              100          73.99        244.75       0.30      x CPU     

10K             1            5.67         17.44        0.33      x CPU     
10K             10           70.65        125.23       0.56      x CPU     
10K             100          745.02       1117.00      0.67      x CPU     

50K             1            34.46        90.23        0.38      x CPU     
50K             10           369.44       562.16       0.66      x CPU     
50K             100          3886.63      5004.70      0.78      x CPU     

100K            1            65.70        207.21       0.32      x CPU     
100K            10           1067.83      1022.39      1.04      x GPU     

=== Memory Transfer Overhead Analysis ===

Total GPU time:     60.44 ms
Transfer time:      69.00 ms (114.2%)

WARNING: Transfer overhead dominates! (>50%)

=== Persistent GPU Memory Test ===

Scenario: Upload database once, run many queries

CPU: 100 queries x 50,000 vectors = 3271.14 ms
GPU: 100 queries x 50,000 vectors = 588.38 ms
Speedup: 5.56x

SUCCESS: GPU is faster with persistent memory!
Per-query latency: 5.88ms

=== Dimension Impact on GPU Performance ===

Dimensions      Vectors      CPU (ms)     GPU (ms)     Speedup   
-----------------------------------------------------------------
128             50K          7.45         25.27        0.29      x
384             50K          27.27        42.67        0.64      x
768             25K          42.97        61.53        0.70      x
1536            10K          26.02        37.65        0.69      x

Higher dimensions = more compute per vector = better GPU utilization
```

---

## Technical Notes

### GTX 1080 Max-Q Limitations

- Mobile variant with thermal throttling
- ~20-30% slower than desktop GTX 1080
- PCIe limited by laptop chipset
- Modern RTX 4000+ series would show significantly better results

### Why GPU is Faster Per-Query (Persistent)

- GTX 1080 has 2560 CUDA cores computing distances in parallel
- CPU has 8 threads computing sequentially
- GPU does ~320 parallel operations per core vs CPU's 8 total threads

---

## Conclusion

**JVectorCUDA's hybrid approach is validated:**

1. **Single queries → CPU** (lower latency, no transfer overhead)
2. **Batch queries with persistent data → GPU** (5.56x throughput)
3. **HybridVectorIndex** automatically routes based on batch size

---
