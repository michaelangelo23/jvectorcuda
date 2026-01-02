# GPU Benchmark Results

Generated: 2026-01-03 00:39:02

## System Information

- **GPU:** NVIDIA GeForce GTX 1080 with MaxQ Design (Compute 6.1, 8191 MB VRAM, 1468 MHz)
- **CPU:** Intel(R) Core(TM) i7-7820HK CPU @ 2.90GHz (8 threads)
- **JVM Memory:** 4056 MB max, 256 MB allocated
- **OS:** Windows 11 10.0 (amd64)
- **Java:** 25 (OpenJDK 64-Bit Server VM)

## Benchmark Results

### Single Query Performance

| Vectors | CPU (ms) | GPU (ms) | Speedup | Winner |
|---------|----------|----------|---------|--------|
| 1,000 | 3.5 | 138.4 | 0.02x | CPU |
| 10,000 | 19.7 | 136.9 | 0.14x | CPU |
| 50,000 | 80.6 | 239.7 | 0.34x | CPU |

### Persistent Memory (Many Queries, Same Dataset)

| Vectors | Queries | CPU (ms) | GPU (ms) | Speedup | Winner |
|---------|---------|----------|----------|---------|--------|
| 1,000 | 10 | 6.8 | 127.2 | 0.05x | CPU |
| 1,000 | 100 | 59.7 | 134.9 | 0.44x | CPU |
| 10,000 | 10 | 62.0 | 138.2 | 0.45x | CPU |
| 10,000 | 100 | 571.9 | 206.6 | 2.77x | GPU |
| 50,000 | 10 | 319.3 | 208.0 | 1.54x | GPU |
| 50,000 | 100 | 2903.1 | 732.8 | 3.96x | GPU |

### Memory Transfer Analysis

- Upload time (50K vectors × 384D): 144.8 ms
- Search time (single query): 6.0 ms
- Transfer overhead: 96%

## Summary

**To contribute:** Copy this entire report and paste it into a GitHub Issue at:
https://github.com/michaelangelo23/jvectorcuda/issues/new
