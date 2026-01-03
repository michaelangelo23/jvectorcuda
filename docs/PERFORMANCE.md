# JVectorCUDA Performance Analysis

This document provides guidance on GPU vs CPU performance trade-offs. Results vary by hardware - **we encourage contributors to run benchmarks and share results!**

---

## Quick Summary

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Single query (cold-start) | **CPU** | Memory transfer overhead dominates |
| Batch queries (10+) with persistent memory | **GPU** | Transfer cost amortized, 2-10x speedup |

---

## Run Your Own Benchmarks

```bash
# Run all GPU benchmarks
./gradlew test --tests "*GpuBreakEvenTest"

# Run full benchmark suite with report
./gradlew runBenchmark
```

The benchmark will automatically:
- Detect your GPU and print specs
- Scale test sizes to your available VRAM
- Compare CPU vs GPU performance
- Generate a report file

---

## Understanding the Results

### Why Cold-Start is Slow

```
Cold-Start GPU (single query):
┌──────────────────────────────────────────────────────────────┐
│  Upload Data     │  Compute  │  Download  │  Total: ~Xms    │
│   (PCIe bound)   │  (fast)   │   (fast)   │                 │
└──────────────────────────────────────────────────────────────┘

CPU (single query):
┌──────────────────────────────────────────────────────────────┐
│              Compute in RAM               │  Total: ~Yms    │
│            (no transfer needed)           │                 │
└──────────────────────────────────────────────────────────────┘
```

For single queries, the GPU must:
1. Upload database to VRAM (slow - PCIe bottleneck)
2. Compute distances (fast)
3. Download results (fast)

The upload time often exceeds the compute time, making CPU faster for cold-start.

### Why Persistent Memory Wins

```
Persistent GPU (100 queries):
┌──────────────────────────────────────────────────────────────┐
│ Upload │ Q1  │ Q2  │ Q3  │ ... │ Q100 │  Total: much faster │
│ (once) │     │     │     │     │      │                     │
└──────────────────────────────────────────────────────────────┘

CPU (100 queries):
┌──────────────────────────────────────────────────────────────┐
│ Q1   │ Q2   │ Q3   │ ... │ Q100 │  Total: N × single query  │
└──────────────────────────────────────────────────────────────┘
```

By keeping the database in GPU memory:
- Upload cost is paid only once
- Each query only uploads the small query vector
- GPU's parallel compute dominates

---

## When to Use Each Mode

### Use CPU (`VectorIndexFactory.cpu()`)
- Single queries or small batches (< 10)
- Database changes frequently
- No GPU available
- Dataset < 10K vectors

### Use GPU (`VectorIndexFactory.gpu()`)
- Database is static (upload once)
- High query throughput needed (100+ queries/sec)
- Dataset is large (50K+ vectors)
- Running many queries against same data

### Use Hybrid (`VectorIndexFactory.hybrid()`) - Recommended
- Automatic routing based on batch size
- Workload varies (mix of single and batch queries)
- Best-of-both-worlds

---

## Code Patterns

### Bad: Cold-Start Pattern (GPU Slower)

```java
// DON'T DO THIS - creates new context every query
for (Query q : queries) {
    try (VectorIndex index = VectorIndexFactory.gpu(dims)) {
        index.add(vectors);  // Upload to VRAM every time!
        index.search(q.vector, k);
    }  // Context destroyed
}
```

### Good: Persistent Memory Pattern (GPU Faster)

```java
// DO THIS - upload once, query many times
try (VectorIndex index = VectorIndexFactory.gpu(dims)) {
    index.add(vectors);  // Upload once
    
    for (Query q : queries) {
        index.search(q.vector, k);  // Data already in VRAM
    }
}
```

### Best: Use Hybrid for Automatic Routing

```java
// RECOMMENDED - automatic CPU/GPU routing
try (VectorIndex index = VectorIndexFactory.hybrid(dims)) {
    index.add(vectors);
    
    // Single query → routed to CPU
    SearchResult single = index.search(query, k);
    
    // Batch query → routed to GPU
    List<SearchResult> batch = index.searchBatch(queries, k);
}
```

---

## Contributing Benchmarks

We welcome benchmark contributions from different hardware configurations!

### How to Contribute

1. Run the benchmark suite:
   ```bash
   ./gradlew runBenchmark
   ```

2. Copy your `benchmark-report.md` output

3. Create a file in `benchmarks/community/` named:
   ```
   benchmarks/community/<GPU_NAME>_<DATE>.md
   ```
   Example: `benchmarks/community/RTX_4090_2026-01-15.md`

4. Submit a PR with your results

### What We're Looking For

- Different GPU generations (Pascal, Turing, Ampere, Ada, Hopper)
- Different VRAM sizes (4GB, 8GB, 16GB, 24GB+)
- Laptop vs desktop comparisons
- Cloud GPU instances (T4, A10, A100, H100)

---

## Performance Factors

### GPU Factors
- **Compute Capability**: Higher = better (6.1 minimum)
- **VRAM**: More = larger datasets
- **Memory Bandwidth**: Higher = faster transfers
- **CUDA Cores**: More = faster parallel compute

### System Factors
- **PCIe Version**: 3.0 vs 4.0 vs 5.0 affects transfer speed
- **PCIe Lanes**: x16 vs x8 vs x4
- **CPU Speed**: Affects comparison baseline
- **System RAM**: Affects CPU-side performance

### Software Factors
- **JVM Warmup**: First few queries may be slower
- **Driver Version**: Newer drivers often faster
- **CUDA Version**: JVectorCUDA targets 11.8+

---

## Expected Results by GPU Tier

| GPU Tier | Example | Expected Persistent Speedup |
|----------|---------|----------------------------|
| Entry Gaming | GTX 1060, RTX 3050 | 2-4x |
| Mid Gaming | GTX 1080, RTX 3070 | 3-6x |
| High Gaming | RTX 3090, RTX 4080 | 5-10x |
| Data Center | T4, A10, A100 | 5-15x+ |

*These are rough estimates. Actual results depend on workload and configuration.*

---

## Troubleshooting

### GPU Slower Than Expected

1. **Check VRAM usage**: Other apps consuming GPU memory?
   ```bash
   nvidia-smi
   ```

2. **Verify persistent memory pattern**: Are you recreating the index for each query?

3. **Check batch size**: Single queries should use CPU

### Out of Memory Errors

Use the GPU memory calculator:
```
VRAM = vectors × dimensions × 4 bytes × 1.2 (overhead)
```

Or check programmatically:
```java
int maxVectors = VramUtil.getMaxSafeVectorCount(384);
```

### Tests Skipped

If GPU tests are skipped:
- Verify NVIDIA driver is installed: `nvidia-smi`
- Check CUDA availability: Run a simple JCuda test
- Ensure GPU meets minimum requirements (Compute 6.1+)

---

## Further Reading

- [OPTIMIZATION_GUIDE.md](../OPTIMIZATION_GUIDE.md) - Technical optimization details
- [README.md](../README.md) - Getting started guide
- [NVIDIA CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

---

## Benchmark Archive

### [2026-01-03] Full Suite Benchmark

**System:**
- **GPU:** NVIDIA GeForce GTX 1080 with MaxQ (8GB VRAM)
- **CPU:** Intel Core i7-7820HK
- **Java:** 25 (OpenJDK 64-Bit)

**Results (Throughput/QPS):**

| Vectors | Queries | CPU (QPS) | GPU (QPS) | Speedup | Note |
|:-------:|:-------:|:---------:|:---------:|:-------:|:-----|
| 1K | 1 | 545.1 | 9.7 | 0.02x | CPU dominates single tiny queries |
| 10K | 1 | 197.5 | 8.9 | 0.05x | Cold start penalty high on GPU |
| 50K | 1 | 33.8 | 5.9 | 0.17x | Transfer overhead visible |
| 10K | 100 | 217.5 | 1575.0 | **7.24x** | GPU efficiently batches 100 queries |
| 50K | 100 | 47.8 | 153.6 | **3.22x** | **GPU sustains 3x higher throughput** |

### Advanced Analysis (Deep Dive)

**1. Persistent Memory vs Cold Start**
*Test Scenario: 50,000 vectors x 384 dimensions, 100 queries*

| Mode | CPU Latency | GPU Latency | Speedup | Conclusion |
|------|-------------|-------------|---------|------------|
| **Cold Start** | 286 ms | 533 ms | 0.54x | CPU faster (Transfer overhead) |
| **Persistent** | 30.38 ms | **5.94 ms** | **5.11x** | **GPU Faster (Raw Compute)** |

> **Key Takeaway:** You MUST use `VectorIndexFactory.gpu()` or `hybrid()` with persistent vectors to see performance gains. Single-shot `add()+search()` patterns are anti-patterns for GPU.

**2. Memory Transfer Overhead**
*Measurement: 50k vectors upload time vs compute time*
- **Transfer Time:** ~70ms
- **Compute Time:** ~6ms
- **Ratio:** 92% of time is spent moving data!
- **Fix:** We are implementing **Pinned Memory** and **Async Streams** in v1.1.0 to hide this latency.

**3. Dimensionality Scaling**
*Impact of vector size on GPU efficiency (Fixed 50MB dataset)*

| Dims | Vectors | GPU Speedup | Insight |
|------|---------|-------------|---------|
| 128 | 50K | 0.61x | Too small for GPU to shine |
| 384 | 50K | 0.65x | Standard embeddings (e.g. MiniLM) |
| 1536 | 10K | 0.25x | OpenAI embeddings (compute bound? Needs investigation) |

**Key Findings:**
1. **Throughput Wins:** For batch workloads, GPU consistently delivers 3-7x higher Queries Per Second (QPS).
2. **Latency vs Throughput:** CPU has lower latency for single items (ms), but GPU has massive bandwidth (QPS).
3. **Scaling:** At 50K vectors, GPU maintains >150 QPS while CPU drops to <50 QPS.
4. **Conclusion:** Use `HybridVectorIndex` to automatically switch between these modes!
