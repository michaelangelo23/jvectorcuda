# Community Benchmark Contributions

This folder contains benchmark results from community contributors with different hardware configurations.

## How to Contribute

1. Run the benchmark suite:
   ```bash
   ./gradlew runBenchmark
   ```

2. Copy your results from `benchmark-report.md`

3. Create a new file with naming convention:
   ```
   <GPU_MODEL>_<DATE>.md
   ```
   Examples:
   - `RTX_4090_2026-01-15.md`
   - `GTX_1080Ti_2026-01-20.md`
   - `Tesla_T4_2026-02-01.md`

4. Submit a PR!

## Template

Use this template for your submission:

```markdown
# Benchmark: [GPU Model]

**Date:** YYYY-MM-DD  
**Contributor:** @github_username

## System Specs
- **GPU:** [Full GPU name from nvidia-smi]
- **VRAM:** [X GB]
- **CPU:** [Model]
- **RAM:** [X GB]
- **OS:** [Windows/Linux/macOS version]
- **Java:** [Version]
- **Driver:** [NVIDIA driver version]

## Results

### Persistent Memory (100 queries × 50K vectors × 384D)
| Metric | CPU | GPU | Speedup |
|--------|-----|-----|---------|
| Total time | X ms | Y ms | Z.ZZx |
| Per-query latency | X ms | Y ms | |

### Cold-Start Break-Even
| Vectors | Batch | CPU (ms) | GPU (ms) | Winner |
|---------|-------|----------|----------|--------|
| 10K | 1 | | | |
| 10K | 10 | | | |
| 50K | 1 | | | |
| 50K | 10 | | | |

### Memory Transfer Overhead
- Transfer time: X ms (Y%)
- Compute time: X ms (Y%)

## Notes
[Any observations about your results]
```

## Current Contributions

| GPU | VRAM | Persistent Speedup | Contributor | Date |
|-----|------|-------------------|-------------|------|
| *Be the first!* | | | | |

---

Thank you for contributing to JVectorCUDA!
