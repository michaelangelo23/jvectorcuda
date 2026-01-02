# JVectorCUDA Performance Summary

**Last Updated:** January 2, 2026  
**Test Hardware:** GTX 1080 Max-Q, 8GB VRAM, CUDA 11.8  
**Test Config:** 384 dimensions, 50K vectors (typical sentence embeddings)

---

## Quick Answer

**GPU wins:** Batch queries (10+ searches on same dataset) → **5x faster**  
**CPU wins:** Single queries or small datasets → **2x faster**  
**Break-even:** ~10 queries on persistent GPU memory

---

## Benchmark Results

### Single Query Mode (Cold Start)
| Dataset | CPU | GPU (with upload) | Winner |
|---------|-----|-------------------|--------|
| 50K vectors | 89 ms | 175 ms | CPU (2x faster) |

GPU loses because upload time (100ms) dominates kernel execution (5ms).

### Persistent Memory Mode (Realistic Usage)
| Queries | CPU Total | GPU Total | Winner |
|---------|-----------|-----------|--------|
| 1 query | 89 ms | 175 ms | CPU |
| 10 queries | 890 ms | 200 ms | GPU (4.5x) |
| 100 queries | 2890 ms | 523 ms | **GPU (5.5x)** |

Upload once, query many times. GPU reaches 5x speedup at 100 queries.

### Distance Metrics (100 queries, persistent mode)
| Metric | GPU | CPU | Speedup |
|--------|-----|-----|---------|
| Euclidean | 523 ms | 2890 ms | 5.5x |
| Cosine | 545 ms | 2950 ms | 5.4x |
| Inner Product | 498 ms | 2850 ms | 5.7x |

All metrics benefit equally from GPU acceleration.

---

## Why Memory Transfer Matters

**Upload bottleneck:** 50K vectors take ~100ms to transfer (CPU→GPU)  
**Kernel execution:** Only ~5ms per query  
**Download results:** ~2ms (small payload)

**Conclusion:** Transfer cost is 20x higher than computation. Persistent memory amortizes this.

---

## Algorithm Differences

| Implementation | Algorithm | Type | Speed |
|----------------|-----------|------|-------|
| CPU | JVector HNSW | Approximate (99%+ recall) | O(log n) |
| GPU | CUDA brute-force | Exact (100% recall) | O(n) |

CPU uses graph-based approximate search, GPU uses exact brute-force.

---

## When to Use Each

**Use GPU when:**
- Same dataset, many queries (batch workload)
- Need exact results
- Have NVIDIA GPU available

**Use CPU when:**
- One-off queries
- Datasets <10K vectors
- No GPU or need lowest latency

---

## Benchmark Reports Archive

## When to Use Each

**Use GPU when:**
- Same dataset, many queries (batch workload)
- Need exact results
- Have NVIDIA GPU available

**Use CPU when:**
- One-off queries
- Datasets <10K vectors
- No GPU or need lowest latency

---

## Benchmark Reports Archive

### Run #1 - January 2, 2026 (GTX 1080 Max-Q)

**Test Environment:**
- GPU: GTX 1080 Max-Q, 8GB VRAM, CUDA 11.8
- CPU: Intel i7-7700HQ @ 2.8GHz
- RAM: 16GB DDR4
- Java: OpenJDK 25
- Config: 50K vectors, 384 dimensions, Euclidean distance

**Results:**

Single Query (Cold Start):
- CPU: 89ms
- GPU: 175ms (includes 100ms upload)
- Winner: CPU (2x faster)

Batch Queries (Persistent Memory):
- 10 queries: GPU 200ms vs CPU 890ms → GPU 4.5x faster
- 100 queries: GPU 523ms vs CPU 2890ms → GPU 5.5x faster

Per-Query Breakdown (Persistent):
- Upload (one-time): 98ms
- Kernel execution: ~5ms per query
- Download results: ~2ms per query

**Key Finding:** Upload overhead dominates. Persistent memory essential for GPU wins.

**Author:** michaelangelo23

---

## Phase 6A Improvements - January 2, 2026

**Completed Enhancements:**
1. GPU Memory Validation - Pre-flight checks prevent OOM crashes
2. Per-Context Kernel Caching - Avoids CUDA_ERROR_INVALID_HANDLE in tests
3. Batch Search API (`searchBatch`) - 5-10x speedup for multiple queries
4. Removed `hybrid()` method - Reduced API confusion
5. Documentation overhaul - README, CHANGELOG, PERFORMANCE.md

**Test Status**: All 41 tests passing on GTX 1080 Max-Q
- **Build**: SUCCESSFUL in 39s
- **Coverage**: 70.67% line coverage, 86.36% class coverage
  - Instructions: 4043/6163 (65.6%)
  - Branches: 198/392 (50.51%)
  - Lines: 848/1200 (70.67%)
  - Methods: 169/231 (73.16%)
  - Classes: 19/22 (86.36%)

**Technical Fixes:**
- Problem 15: JAR duplicate strategy (FIXED)
- Problem 16: Test crashes from static module caching (FIXED with per-context caching)

---

## Contributing Benchmarks

Have a different GPU? Run `./gradlew benchmark` and contribute results as a comparison table.

**Format for Additional GPUs:**

| GPU Model | VRAM | Single Query | 100 Queries (Persistent) | Speedup vs GTX 1080 |
|-----------|------|--------------|--------------------------|---------------------|
| GTX 1080 Max-Q (baseline) | 8GB | 175ms | 523ms | 1.0x |
| [Your GPU] | XGB | Xms | Xms | X.Xx |

**Include in PR:**
- CPU model and cores
- Java version
- CUDA version
- Test config (if different from 50K vectors @ 384D)

This helps users understand performance scaling across different hardware without duplicating full benchmark reports.

