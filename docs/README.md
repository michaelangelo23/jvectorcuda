# Benchmarks

## Overview

See [PERFORMANCE.md](PERFORMANCE.md) for comprehensive GPU vs CPU performance analysis.

## Key Finding

**GPU is 5.56x faster** when data persists on GPU memory (upload once, query many times).  
**CPU is faster** for cold-start scenarios due to PCIe transfer overhead.

## Run Your Own

```bash
./gradlew benchmark
```

Share your results by opening an issue!
