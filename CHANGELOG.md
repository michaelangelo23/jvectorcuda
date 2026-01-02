# Changelog

All notable changes to JVectorCUDA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GPU memory validation in `GPUVectorIndex` constructor to prevent out-of-memory crashes
- `searchBatch()` method in both GPU and CPU implementations for multiple queries
- Per-context kernel caching in `GpuKernelLoader` to avoid reloading PTX on every index creation
- Comprehensive `PERFORMANCE.md` with benchmark results and usage guidance
- Thread safety warnings in README
- Algorithm differences documentation (CPU uses HNSW, GPU uses brute-force)
- "When to Use JVectorCUDA" section in README
- Prominent CUDA requirement notice in README
- `CHANGELOG.md` for version tracking

### Changed
- Repository description now clearly states NVIDIA GPU requirement
- README messaging focuses on realistic use cases (batch queries)
- Removed emojis from documentation for professional tone
- Benchmark framework now clearly documents cold-start vs persistent modes
- Kernel caching now per-context aware to prevent CUDA_ERROR_INVALID_HANDLE in tests

### Removed
- `hybrid()` factory method from `VectorIndexFactory` (was throwing UnsupportedOperationException)

### Fixed
- References to non-existent `STRATEGIC_INSIGHTS.md` replaced with `PERFORMANCE.md`
- JAR duplicate strategy issue causing build failures (added `duplicatesStrategy = DuplicatesStrategy.EXCLUDE`)
- Test failures from static module caching across CUDA contexts (changed to per-context caching)

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
