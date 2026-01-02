# Performance

**Hardware:** GTX 1080 Max-Q, 8GB VRAM  
**Config:** 50K vectors, 384 dimensions

## Results

### Single Query
| | Time | Winner |
|---|------|--------|
| CPU | 89 ms | ✓ |
| GPU | 175 ms | |

### Batch Queries (persistent GPU memory)
| Queries | CPU | GPU | Speedup |
|---------|-----|-----|---------|
| 10 | 890 ms | 200 ms | 4.5x |
| 100 | 2890 ms | 523 ms | 5.5x |

### By Distance Metric (100 queries)
| Metric | GPU | CPU | Speedup |
|--------|-----|-----|---------|
| Euclidean | 523 ms | 2890 ms | 5.5x |
| Cosine | 545 ms | 2950 ms | 5.4x |
| Inner Product | 498 ms | 2850 ms | 5.7x |

## Why

- GPU upload: ~100ms (one-time cost)
- GPU kernel: ~5ms per query
- CPU search: ~89ms per query

Single query = upload dominates, CPU wins.  
Many queries = upload amortized, GPU wins.

## When to use what

**GPU:** Batch queries on same dataset  
**CPU:** One-off queries, small datasets (<10K)
