package com.vindex.jvectorcuda;

import com.vindex.jvectorcuda.cpu.CPUVectorIndex;
import com.vindex.jvectorcuda.gpu.GPUVectorIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Factory for creating {@link VectorIndex} instances with automatic GPU/CPU detection.
 *
 * <p>This factory provides multiple strategies for creating vector indices:
 * <ul>
 *   <li><b>Hybrid (Recommended):</b> Intelligent routing between GPU and CPU based on workload</li>
 *   <li><b>Auto:</b> Automatically selects GPU if CUDA is available, falls back to CPU</li>
 *   <li><b>CPU:</b> Explicitly creates a CPU-based index using JVector HNSW</li>
 *   <li><b>GPU:</b> Explicitly creates a GPU-based index using CUDA</li>
 *   <li><b>Thread-safe:</b> Wraps any index with {@link ThreadSafeVectorIndex}</li>
 * </ul>
 *
 * <h2>Basic Usage</h2>
 * <pre>{@code
 * // Recommended: Hybrid index with intelligent routing
 * VectorIndex index = VectorIndexFactory.hybrid(384);
 *
 * // Auto-detect best available backend (GPU preferred)
 * VectorIndex index = VectorIndexFactory.auto(384);
 *
 * // Force CPU (HNSW approximate search)
 * VectorIndex cpuIndex = VectorIndexFactory.cpu(768);
 *
 * // Force GPU with custom distance metric
 * VectorIndex gpuIndex = VectorIndexFactory.gpu(1536, DistanceMetric.COSINE);
 * }</pre>
 *
 * <h2>Thread-Safe Usage</h2>
 * <pre>{@code
 * // Thread-safe wrapper for concurrent applications
 * VectorIndex threadSafe = VectorIndexFactory.autoThreadSafe(384);
 *
 * // Use from multiple threads safely
 * ExecutorService executor = Executors.newFixedThreadPool(4);
 * for (int i = 0; i < 100; i++) {
 *     executor.submit(() -> threadSafe.search(queryVector, 10));
 * }
 * }</pre>
 *
 * <h2>Distance Metrics</h2>
 * <p>Supported distance metrics:
 * <ul>
 *   <li>{@link DistanceMetric#EUCLIDEAN} - L2 distance (default)</li>
 *   <li>{@link DistanceMetric#COSINE} - Cosine similarity (1 - cosine)</li>
 *   <li>{@link DistanceMetric#INNER_PRODUCT} - Negative inner product</li>
 * </ul>
 *
 * @see VectorIndex
 * @see ThreadSafeVectorIndex
 * @see DistanceMetric
 * @since 1.0.0
 */
public final class VectorIndexFactory {

    private static final Logger logger = LoggerFactory.getLogger(VectorIndexFactory.class);
    
    private VectorIndexFactory() {
        // Utility class - prevent instantiation
    }

    // ===========================================
    // Hybrid Index (Recommended)
    // ===========================================

    /**
     * Creates a hybrid index with intelligent CPU/GPU routing using Euclidean distance.
     *
     * <p><b>This is the recommended default for most applications.</b> The hybrid index
     * automatically routes queries to the optimal backend based on workload characteristics:
     * <ul>
     *   <li>Single queries → CPU (lowest latency)</li>
     *   <li>Batch queries (10+) → GPU (5x+ speedup)</li>
     *   <li>Large datasets (50K+) with persistent GPU memory → GPU</li>
     * </ul>
     *
     * @param dimensions the dimensionality of vectors to be indexed
     * @return a new HybridVectorIndex instance
     * @throws IllegalArgumentException if dimensions &le; 0
     * @see HybridVectorIndex
     */
    public static VectorIndex hybrid(int dimensions) {
        return hybrid(dimensions, DistanceMetric.EUCLIDEAN);
    }

    /**
     * Creates a hybrid index with intelligent CPU/GPU routing using the specified metric.
     *
     * @param dimensions the dimensionality of vectors to be indexed
     * @param metric the distance metric to use for similarity calculations
     * @return a new HybridVectorIndex instance
     * @throws IllegalArgumentException if dimensions &le; 0 or metric is null
     * @see HybridVectorIndex
     */
    public static VectorIndex hybrid(int dimensions, DistanceMetric metric) {
        validateDimensions(dimensions);
        validateMetric(metric);
        logger.info("Creating hybrid index: {} dimensions, {} (intelligent routing)", dimensions, metric);
        return new HybridVectorIndex(dimensions, metric);
    }

    /**
     * Creates a thread-safe hybrid index with intelligent CPU/GPU routing.
     *
     * <p>Combines the benefits of hybrid routing with thread-safe concurrent access.
     *
     * @param dimensions the dimensionality of vectors to be indexed
     * @return a thread-safe HybridVectorIndex instance
     * @throws IllegalArgumentException if dimensions &le; 0
     * @see HybridVectorIndex
     * @see ThreadSafeVectorIndex
     */
    public static VectorIndex hybridThreadSafe(int dimensions) {
        return hybridThreadSafe(dimensions, DistanceMetric.EUCLIDEAN);
    }

    /**
     * Creates a thread-safe hybrid index with intelligent CPU/GPU routing.
     *
     * @param dimensions the dimensionality of vectors to be indexed
     * @param metric the distance metric to use
     * @return a thread-safe HybridVectorIndex instance
     * @throws IllegalArgumentException if dimensions &le; 0 or metric is null
     */
    public static VectorIndex hybridThreadSafe(int dimensions, DistanceMetric metric) {
        VectorIndex baseIndex = hybrid(dimensions, metric);
        return new ThreadSafeVectorIndex(baseIndex);
    }

    // ===========================================
    // Auto Detection (GPU preferred)
    // ===========================================

    /**
     * Creates an index with automatic GPU/CPU detection using Euclidean distance.
     *
     * <p>Attempts GPU first (5x+ speedup for batch queries), falls back to CPU
     * if CUDA is unavailable or GPU initialization fails.
     *
     * @param dimensions the dimensionality of vectors to be indexed
     * @return a new VectorIndex instance (GPU or CPU)
     * @throws IllegalArgumentException if dimensions &le; 0
     */
    public static VectorIndex auto(int dimensions) {
        return auto(dimensions, DistanceMetric.EUCLIDEAN);
    }

    /**
     * Creates an index with automatic GPU/CPU detection using the specified distance metric.
     *
     * <p>Attempts GPU first (5x+ speedup for batch queries), falls back to CPU
     * if CUDA is unavailable or GPU initialization fails.
     *
     * @param dimensions the dimensionality of vectors to be indexed
     * @param metric the distance metric to use for similarity calculations
     * @return a new VectorIndex instance (GPU or CPU)
     * @throws IllegalArgumentException if dimensions &le; 0 or metric is null
     */
    public static VectorIndex auto(int dimensions, DistanceMetric metric) {
        validateDimensions(dimensions);
        validateMetric(metric);

        // Try GPU first - persistent memory architecture makes it 5x faster
        if (CudaDetector.isAvailable()) {
            try {
                VectorIndex gpuIndex = gpu(dimensions, metric);
                logger.info("Auto-selected GPU index ({} dimensions, {}) - 5x speedup expected", 
                    dimensions, metric);
                return gpuIndex;
            } catch (RuntimeException e) {
                logger.warn("GPU initialization failed, falling back to CPU: {}", e.getMessage());
            }
        } else {
            logger.info("CUDA not available, using CPU index");
        }

        // Fallback to CPU
        return cpu(dimensions);
    }

    /**
     * Creates a CPU-only index using JVector's HNSW algorithm.
     *
     * <p>The CPU implementation uses approximate nearest neighbor search (HNSW),
     * which is faster for single queries but less accurate than brute-force.
     *
     * <h3>When to Use CPU</h3>
     * <ul>
     *   <li>Single queries where GPU upload overhead isn't amortized</li>
     *   <li>Systems without NVIDIA GPU</li>
     *   <li>Memory-constrained environments</li>
     *   <li>When approximate results are acceptable</li>
     * </ul>
     *
     * @param dimensions the dimensionality of vectors to be indexed
     * @return a new CPU-based VectorIndex instance
     * @throws IllegalArgumentException if dimensions &le; 0
     */
    public static VectorIndex cpu(int dimensions) {
        validateDimensions(dimensions);
        logger.debug("Creating CPU index with {} dimensions", dimensions);
        return new CPUVectorIndex(dimensions);
    }

    /**
     * Creates a GPU-only index using CUDA brute-force search with Euclidean distance.
     *
     * @param dimensions the dimensionality of vectors to be indexed
     * @return a new GPU-based VectorIndex instance
     * @throws IllegalArgumentException if dimensions &le; 0
     * @throws UnsupportedOperationException if CUDA is not available
     * @throws IllegalStateException if GPU memory is insufficient
     */
    public static VectorIndex gpu(int dimensions) {
        return gpu(dimensions, DistanceMetric.EUCLIDEAN);
    }

    /**
     * Creates a GPU-only index using CUDA brute-force search.
     *
     * <p>The GPU implementation uses exact brute-force search, which is slower
     * for single queries (due to upload overhead) but much faster for batch queries.
     *
     * <h3>When to Use GPU</h3>
     * <ul>
     *   <li>Batch queries (10+ queries at once)</li>
     *   <li>Large datasets (50K+ vectors)</li>
     *   <li>High-dimensional vectors (768+)</li>
     *   <li>When exact results are required</li>
     * </ul>
     *
     * <h3>Requirements</h3>
     * <ul>
     *   <li>NVIDIA GPU with CUDA Compute Capability 3.5+</li>
     *   <li>CUDA 11.8+ runtime installed</li>
     *   <li>JCuda libraries in classpath</li>
     * </ul>
     *
     * @param dimensions the dimensionality of vectors to be indexed
     * @param metric the distance metric to use
     * @return a new GPU-based VectorIndex instance
     * @throws IllegalArgumentException if dimensions &le; 0 or metric is null
     * @throws UnsupportedOperationException if CUDA is not available
     * @throws IllegalStateException if GPU memory is insufficient
     */
    public static VectorIndex gpu(int dimensions, DistanceMetric metric) {
        validateDimensions(dimensions);
        validateMetric(metric);

        if (!CudaDetector.isAvailable()) {
            throw new UnsupportedOperationException("CUDA is not available on this system");
        }

        logger.debug("Creating GPU index with {} dimensions, {} (persistent memory)", dimensions, metric);
        return new GPUVectorIndex(dimensions, metric);
    }

    private static void validateDimensions(int dimensions) {
        if (dimensions <= 0) {
            throw new IllegalArgumentException("Dimensions must be positive, got: " + dimensions);
        }
    }

    /**
     * Creates a thread-safe wrapper around an auto-detected index.
     *
     * <p>Uses {@link java.util.concurrent.locks.ReadWriteLock} for concurrent read
     * access and exclusive write access. Multiple threads can search simultaneously,
     * but writes (add) block all other operations.
     *
     * @param dimensions the dimensionality of vectors to be indexed
     * @return a thread-safe VectorIndex instance
     * @throws IllegalArgumentException if dimensions &le; 0
     * @see ThreadSafeVectorIndex
     */
    public static VectorIndex autoThreadSafe(int dimensions) {
        return autoThreadSafe(dimensions, DistanceMetric.EUCLIDEAN);
    }

    /**
     * Creates a thread-safe wrapper around an auto-detected index with custom metric.
     *
     * @param dimensions the dimensionality of vectors to be indexed
     * @param metric the distance metric to use
     * @return a thread-safe VectorIndex instance
     * @throws IllegalArgumentException if dimensions &le; 0 or metric is null
     * @see ThreadSafeVectorIndex
     */
    public static VectorIndex autoThreadSafe(int dimensions, DistanceMetric metric) {
        VectorIndex baseIndex = auto(dimensions, metric);
        return new ThreadSafeVectorIndex(baseIndex);
    }

    /**
     * Creates a thread-safe wrapper around a CPU index.
     *
     * <p>Recommended for multi-threaded applications using CPU backend.
     *
     * @param dimensions the dimensionality of vectors to be indexed
     * @return a thread-safe CPU VectorIndex instance
     * @throws IllegalArgumentException if dimensions &le; 0
     */
    public static VectorIndex cpuThreadSafe(int dimensions) {
        VectorIndex baseIndex = cpu(dimensions);
        return new ThreadSafeVectorIndex(baseIndex);
    }

    /**
     * Creates a thread-safe wrapper around a GPU index.
     *
     * <p><b>Note:</b> GPU operations are serialized through CUDA, so thread-safe
     * wrapper adds coordination overhead. Consider using CPU for highly concurrent
     * workloads with many small queries.
     *
     * @param dimensions the dimensionality of vectors to be indexed
     * @return a thread-safe GPU VectorIndex instance
     * @throws IllegalArgumentException if dimensions &le; 0
     * @throws UnsupportedOperationException if CUDA is not available
     */
    public static VectorIndex gpuThreadSafe(int dimensions) {
        return gpuThreadSafe(dimensions, DistanceMetric.EUCLIDEAN);
    }

    /**
     * Creates a thread-safe wrapper around a GPU index with custom metric.
     *
     * @param dimensions the dimensionality of vectors to be indexed
     * @param metric the distance metric to use
     * @return a thread-safe GPU VectorIndex instance
     * @throws IllegalArgumentException if dimensions &le; 0 or metric is null
     * @throws UnsupportedOperationException if CUDA is not available
     */
    public static VectorIndex gpuThreadSafe(int dimensions, DistanceMetric metric) {
        VectorIndex baseIndex = gpu(dimensions, metric);
        return new ThreadSafeVectorIndex(baseIndex);
    }

    private static void validateMetric(DistanceMetric metric) {
        if (metric == null) {
            throw new IllegalArgumentException("Distance metric cannot be null");
        }
    }
}
