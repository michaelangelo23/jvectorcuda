# Contributing to JVectorCUDA

We're building intelligent GPU-accelerated vector search for Java. The core idea: use GPU when it's faster, fall back to CPU when it's not, automatically.

## What We Need Most: GPU Benchmarks

Our testing on GTX 1080 Max-Q showed that GPU isn't always faster. Memory transfer overhead can kill performance on smaller datasets. We need data from modern GPUs (RTX 2000-4000 series) to understand where the real break-even points are.

If you have access to an RTX GPU, running our benchmark suite is the most valuable contribution you can make right now.

## Running Benchmarks

**Requirements**
- JDK 21+ ([Eclipse Temurin](https://adoptium.net/) recommended)
- Gradle 8.0+ (or use included `./gradlew` wrapper)
- NVIDIA GPU with CUDA Compute 6.1+ (GTX 1060 or newer)
- CUDA Toolkit 11.8+ ([download](https://developer.nvidia.com/cuda-toolkit))
- Updated NVIDIA drivers

```bash
git clone https://github.com/michaelangelo23/jvectorcuda.git
cd jvectorcuda
./gradlew benchmark
```

This generates `benchmark-report.md` with all system info and results. Copy and paste it into a GitHub Issue.

The report includes:
- Auto-detected GPU/CPU info
- Single query performance (tests JNI overhead)
- Persistent memory performance (tests amortized transfer cost)
- Memory transfer analysis

## Sharing Results

Open a GitHub Issue and paste the contents of `benchmark-report.md`. That's it.

## Development Setup

**Requirements**
- JDK 21+
- CUDA Toolkit 11.8+
- Visual Studio 2019 Build Tools (Windows)

**Build**
```bash
./gradlew build
./gradlew test
```

**Compile CUDA kernels** (if modifying .cu files)
```bash
# Windows
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && nvcc -ptx -arch=compute_61 -O3 --use_fast_math src/main/resources/kernels/euclidean_distance.cu -o src/main/resources/kernels/euclidean_distance.ptx'
```

## Testing Requirements

All code changes need tests covering:
- Correctness (results match CPU reference)
- Edge cases (null, empty, NaN, infinity)
- Performance characteristics (document behavior, don't assume GPU is faster)

## Current Priorities

1. GPU benchmarks on RTX hardware
2. Adaptive routing algorithm
3. JavaFX 3D visualization
4. Spring Boot integration

Check GitHub Issues for specific tasks.

## Acknowledgements

This project benefits from community GPU benchmarking. Every data point helps us build better routing logic—whether GPU wins or loses on your hardware, we want to know.
