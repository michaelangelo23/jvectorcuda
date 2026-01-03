# JVectorCUDA Performance Analysis

This document provides guidance on GPU vs CPU performance trade-offs. Results vary by hardware - **we encourage contributors to run benchmarks and share results!**

---

## Quick Summary

| Scenario | Typical Winner | Why |
|----------|----------------|-----|
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
