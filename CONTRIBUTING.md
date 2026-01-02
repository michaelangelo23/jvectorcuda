# Contributing to JVectorCUDA

Thank you for your interest in contributing to JVectorCUDA! This project aims to bring intelligent GPU-accelerated vector search to Java, with automatic CPU/GPU routing based on your hardware and workload.

---

## Project Goal

**Build the first Java vector search library with intelligent CPU/GPU routing** that automatically uses GPU when beneficial, CPU when faster - with zero configuration.

---

## Critical Need: GPU Performance Testing

**We discovered:** GTX 1080 Max-Q (mobile GPU) is **slower** than modern CPUs for vector search due to memory transfer overhead.

**We need YOUR help** to test on modern GPUs to find the break-even points!

---

## How to Contribute: GPU Benchmarking

### If You Have RTX GPU (2000-4000 series)

Your hardware is **critical** for validating GPU performance! Here's how to help:

#### Step 1: Clone and Build

```bash
git clone https://github.com/yourusername/jvectorcuda.git
cd jvectorcuda
./gradlew build
```

#### Step 2: Run GPU Benchmark Suite

```bash
# Run comprehensive GPU vs CPU benchmarks
./gradlew test --tests GpuBreakEvenTest

# Or specific scenarios
./gradlew test --tests "GpuBreakEvenTest.testBreakEvenPoints"
./gradlew test --tests "GpuBreakEvenTest.testMemoryTransferOverhead"
./gradlew test --tests "GpuBreakEvenTest.testPersistentGpuMemory"
```

#### Step 3: Share Your Results

Open a GitHub Issue with this template:

```markdown
**GPU Benchmarking Results**

**Hardware:**
- GPU: [e.g., RTX 4090, RTX 3080, RTX 2060]
- CPU: [e.g., Ryzen 9 5900X, Intel i7-12700K]
- RAM: [e.g., 32GB DDR4-3200]
- OS: [Windows 11 / Ubuntu 22.04]
- CUDA Version: [e.g., 11.8, 12.2]

**Test Results:**
Paste the benchmark output here, including:
- Speedup for different vector counts (1K, 10K, 100K, 1M)
- Speedup for different batch sizes (1, 10, 100, 1000)
- Memory transfer overhead percentage
- Break-even point (where GPU starts winning)

**Logs:**
Attach `build/reports/tests/test/index.html` or paste console output
```

---

## What Makes a Good GPU Benchmark Report

### Include These Metrics:

1. **Speedup by Dataset Size**
   ```
   1K vectors:    CPU vs GPU speedup
   10K vectors:   CPU vs GPU speedup
   100K vectors:  CPU vs GPU speedup
   1M vectors:    CPU vs GPU speedup
   ```

2. **Speedup by Batch Size**
   ```
   Single query:  CPU vs GPU speedup
   10 queries:    CPU vs GPU speedup
   100 queries:   CPU vs GPU speedup
   1000 queries:  CPU vs GPU speedup
   ```

3. **Memory Transfer Analysis**
   ```
   Transfer time: X ms (Y% of total)
   Compute time:  X ms (Y% of total)
   ```

4. **Break-Even Point**
   ```
   "GPU starts winning at: X vectors with Y batch size"
   ```

### Example Good Report:

```
**Hardware:** RTX 4090, Ryzen 9 7950X, 64GB DDR5

**Results:**
- 1K vectors × 1 batch:    0.8x (CPU faster)
- 10K vectors × 10 batch:  1.5x (GPU wins!)
- 100K vectors × 100:      8.2x (GPU dominates)
- 1M vectors × 1000:       25x (GPU crushes CPU)

**Break-even:** ~5K vectors with batch size 5+

**Transfer overhead:** 15% (good!)
```

---

## Advanced: Profiling Contributions

### If you want to dive deeper:

1. **Profile specific operations:**
   ```bash
   # Profile memory transfers
   ./gradlew test --tests "GpuBreakEvenTest.testMemoryTransferOverhead"
   
   # Profile kernel execution
   ./gradlew test --tests "GpuBreakEvenTest.testKernelExecutionTime"
   ```

2. **Test edge cases:**
   - Different vector dimensions (128, 384, 768, 1536)
   - Different distance metrics (Euclidean, Cosine, Dot Product)
   - Concurrent queries
   - Memory-constrained scenarios

3. **Capture NVIDIA profiler data:**
   ```bash
   nvprof --print-gpu-trace ./gradlew test --tests GpuBreakEvenTest
   ```

---

## Testing on Cloud GPUs

Don't have an RTX GPU? Test on cloud for ~$1-3!

### AWS (P3 instances with V100):

```bash
# Launch P3.2xlarge (~$3/hour)
aws ec2 run-instances --image-id ami-xxxxx --instance-type p3.2xlarge

# SSH and run benchmarks
ssh ubuntu@<instance-ip>
git clone https://github.com/yourusername/jvectorcuda.git
cd jvectorcuda
./gradlew test --tests GpuBreakEvenTest
```

### Google Colab (Free T4 GPU):

1. Open [Google Colab](https://colab.research.google.com/)
2. Runtime → Change runtime type → GPU (T4)
3. Run benchmarks:
   ```python
   !git clone https://github.com/yourusername/jvectorcuda.git
   %cd jvectorcuda
   !./gradlew test --tests GpuBreakEvenTest
   ```

---

## Other Ways to Contribute

### 1. Code Contributions

**Current priorities:**
- [ ] cuVS integration (Java bindings)
- [ ] Adaptive routing algorithm
- [ ] Visualization features (3D t-SNE/UMAP)
- [ ] Spring Boot starter
- [ ] Additional distance metrics

**Before contributing code:**
1. Check existing issues or create one describing your idea
2. Fork the repository
3. Follow our testing strategy (see `TESTING_STRATEGY.md`)
4. Run all tests: `./gradlew test`
5. Submit PR with clear description

### 2. Documentation

- Improve README with use cases
- Add tutorials for common scenarios
- Document performance characteristics
- Create integration examples

### 3. Bug Reports

Use this template:

```markdown
**Bug Description:**
Clear description of the bug

**To Reproduce:**
1. Step 1
2. Step 2
3. See error

**Expected Behavior:**
What should happen

**Environment:**
- JVectorCUDA version:
- Java version:
- OS:
- GPU (if applicable):
- CUDA version (if applicable):

**Logs:**
Paste relevant error messages or stack traces
```

---

## Development Setup

### Prerequisites

- **JDK 17+** (for development)
- **CUDA Toolkit 11.8+** (for kernel compilation)
- **Visual Studio 2019 Build Tools** (Windows only, for CUDA)
- **Gradle 8.0+** (wrapper included)

### Build from Source

```bash
# Clone repository
git clone https://github.com/yourusername/jvectorcuda.git
cd jvectorcuda

# Build
./gradlew build

# Run all tests
./gradlew test

# Run specific test
./gradlew test --tests CudaAvailabilityTest
```

### Compile CUDA Kernels

```bash
# Windows (with VS 2019)
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && nvcc -ptx -arch=compute_61 -O3 --use_fast_math src/main/resources/kernels/euclidean_distance.cu -o src/main/resources/kernels/euclidean_distance.ptx'

# Linux
nvcc -ptx -arch=compute_61 -O3 --use_fast_math src/main/resources/kernels/euclidean_distance.cu -o src/main/resources/kernels/euclidean_distance.ptx
```

---

## Code Style

We follow **Google Java Style Guide** + **NVIDIA CUDA Best Practices**:

- SOLID principles
- Max 20 lines per function
- Meaningful names (no `foo`, `bar`, `temp`)
- Comprehensive Javadoc for public APIs
- Edge case testing (null, empty, NaN, infinity)
- Performance testing (document behavior, don't just assert faster)

See `user_global.md` for detailed standards.

---

## Testing Requirements

All contributions MUST include tests that cover:

### 1. Correctness
- [ ] Results match CPU implementation
- [ ] Results match reference implementation (NumPy, JVector)

### 2. Edge Cases
- [ ] Null inputs
- [ ] Empty collections
- [ ] Zero-filled data
- [ ] NaN / Infinity handling
- [ ] Negative values
- [ ] Boundary conditions

### 3. Performance Characteristics
- [ ] CPU vs GPU comparison
- [ ] Document when GPU wins/loses
- [ ] Memory transfer overhead measured
- [ ] NO blind assertions that GPU is faster

See `/test-feature` workflow and `TESTING_STRATEGY.md` for details.

---

## Current Priorities (Help Wanted!)

### High Priority:
1. **GPU benchmarking on RTX 2000-4000 series** [HIGH PRIORITY]
2. cuVS integration validation
3. Adaptive routing algorithm design
4. Break-even point documentation

### Medium Priority:
- Visualization features
- Spring Boot integration
- Additional distance metrics
- Quantization support

### Low Priority (Future):
- Multi-GPU support
- Disk-based indexes
- Distributed search

---

## Communication

- **GitHub Issues:** Bug reports, feature requests, benchmarking results
- **GitHub Discussions:** General questions, architecture discussions
- **Pull Requests:** Code contributions (please discuss first for large changes)

---

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License (same as project).

---

## Special Thanks

Contributors who test on diverse GPUs are **CRITICAL** to this project's success!

Your benchmarking results will:
- Validate GPU acceleration viability
- Determine adaptive routing thresholds
- Guide performance optimization priorities
- Help us document when GPU helps vs hurts

**Thank you for helping make GPU vector search accessible in Java!**
