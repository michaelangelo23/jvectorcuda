# Changelog

All notable changes to JVectorCUDA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-03

### Added
- **HybridVectorIndex** - Intelligent CPU/GPU routing for optimal performance
  - Automatically routes single queries to CPU (lowest latency)
  - Automatically routes batch queries (10+) to GPU (5x+ speedup)
  - Configurable thresholds via Builder pattern
  - `getRoutingReport()` for debugging routing decisions
- `VectorIndexFactory.hybrid()` and `hybridThreadSafe()` factory methods
- `VectorIndex.hybrid()` static convenience method
- Comprehensive `HybridVectorIndexTest` with 27 test cases
- `ThreadSafeVectorIndex` wrapper class with ReadWriteLock for concurrent access
- Factory methods: `autoThreadSafe()`, `cpuThreadSafe()`, `gpuThreadSafe()` in VectorIndexFactory
- Comprehensive integration test suite (`IntegrationTest.java`) with 21 test cases
- End-to-end workflow tests (CPU, GPU, auto-detection, batch, async)
- Large-scale tests (100K vectors, 1000 batch queries, 1536D high-dimensional)
- Stress tests (rapid open/close cycles, concurrent searches, memory stress)
- Error handling tests (invalid input, null checks, operations after close)
- Distance metric edge cases in `DistanceMetricTest` (cosine negative values, inner product large magnitudes)
- GPU memory validation in `GPUVectorIndex` constructor to prevent out-of-memory crashes
- `searchBatch()` method in both GPU and CPU implementations for multiple queries
- Per-context kernel caching in `GpuKernelLoader` to avoid reloading PTX on every index creation
- Comprehensive `PERFORMANCE.md` with benchmark results and usage guidance
- `CHANGELOG.md` for version tracking
- **Comprehensive Javadoc documentation** for all public APIs
- Javadoc configuration in Gradle with `javadoc`, `javadocJar`, and `sourcesJar` tasks
- `indices()` and `distances()` alias methods in `SearchResult` for cleaner API
- **CUDA driver version validation** - checks for minimum 11.8 at runtime
- `CudaDetector.getDriverVersion()` - returns current CUDA driver version
- `CudaDetector.getCompatibilityReport()` - comprehensive compatibility diagnostics

### Changed
- README simplified to essential information only
- `add(null)` now throws `IllegalArgumentException` instead of silently returning (breaking change)
- Repository description now clearly states NVIDIA GPU requirement
- README messaging focuses on realistic use cases (batch queries)
- Removed emojis from documentation for professional tone
- Benchmark framework now clearly documents cold-start vs persistent modes
- Kernel caching now per-context aware to prevent CUDA_ERROR_INVALID_HANDLE in tests

### Fixed
- **Security (CodeQL):** Command injection vulnerability in BenchmarkRunner - now uses absolute paths with String[] for Runtime.exec()
- **Security (CodeQL):** Missing @Override annotations on `searchBatch()` in CPUVectorIndex and GPUVectorIndex
- **Security (CodeQL):** Uncaught NumberFormatException in BenchmarkFrameworkTest - now wrapped with try-catch
- **Security (CodeQL #9):** Unsafe use of getResource in GpuKernelLoader - now uses class literal instead of getClass()
- Deprecated clockRate field warnings in CudaDetector.java (suppressed with annotation, no JCuda alternative)
- Unused import (ArrayList) removed from IntegrationTest.java
- References to non-existent `STRATEGIC_INSIGHTS.md` replaced with `PERFORMANCE.md`
- JAR duplicate strategy issue causing build failures (added `duplicatesStrategy = DuplicatesStrategy.EXCLUDE`)
- Test failures from static module caching across CUDA contexts (changed to per-context caching)

### Removed
- `hybrid()` factory method stub from `VectorIndexFactory` (was throwing UnsupportedOperationException) - now fully implemented
- Unused kernel files: `vector_add.cu`, `vector_add.ptx` (never integrated into library)
- Orphaned test file: `VectorAdditionTest.java` (tested non-existent functionality)

## [1.0.0-SNAPSHOT] - In Development

### Added
- Core `VectorIndex` interface with GPU/CPU implementations
- CUDA-accelerated brute-force vector search
- Persistent GPU memory mode for batch queries (5x+ speedup)
- Three distance metrics: Euclidean, Cosine Similarity, Inner Product
- Automatic CPU fallback when GPU unavailable
- Comprehensive test suite with edge cases and AI blind spots
- Benchmarking framework with performance metrics
- JaCoCo code coverage integration
- CI/CD with GitHub Actions
- CodeQL security scanning
- Dependabot dependency updates

### Known Limitations
- Not thread-safe (single-threaded access only)
- No incremental updates (must rebuild index after adding vectors)
- No deletion support
- GPU-only brute-force (no approximate methods yet)
- Cold-start penalty for single queries (upload overhead)

[Unreleased]: https://github.com/michaelangelo23/jvectorcuda/compare/v1.0.0-SNAPSHOT...HEAD
[1.0.0-SNAPSHOT]: https://github.com/michaelangelo23/jvectorcuda/releases/tag/v1.0.0-SNAPSHOT
