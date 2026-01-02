package com.vindex.jvectorcuda;

import com.vindex.jvectorcuda.cpu.CPUVectorIndex;
import com.vindex.jvectorcuda.gpu.GPUVectorIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Factory for creating VectorIndex instances with automatic GPU/CPU detection.
 * 
 * <h2>Adaptive Routing Strategy</h2>
 * Based on Phase 2.5 break-even analysis (GTX 1080 Max-Q), the factory uses:
 * <ul>
 *   <li><b>GPU:</b> When CUDA is available (5x speedup with persistent memory)</li>
 *   <li><b>CPU:</b> Fallback when GPU is unavailable or initialization fails</li>
 * </ul>
 * 
 * <h2>Key Findings from Validation:</h2>
 * <ul>
 *   <li>GPU with persistent memory: 5.09x faster than CPU</li>
 *   <li>Single-query fresh upload: CPU is faster (memory transfer overhead)</li>
 *   <li>Batch queries: GPU excels due to amortized memory costs</li>
 * </ul>
 * 
 * <h2>Distance Metrics:</h2>
 * <ul>
 *   <li>EUCLIDEAN - L2 distance (default)</li>
 *   <li>COSINE - 1 - cosine_similarity</li>
 *   <li>INNER_PRODUCT - negative dot product</li>
 * </ul>
 * 
 * <p>The {@link #auto(int)} method creates a GPU index when available, as the
 * GPUVectorIndex uses persistent memory by design (vectors uploaded once,
 * queries streamed).
 *
 * @author JVectorCUDA (AI-assisted, Human-verified)
 * @since 1.0.0
 */
public final class VectorIndexFactory {

    private static final Logger logger = LoggerFactory.getLogger(VectorIndexFactory.class);
    
    private VectorIndexFactory() {
        // Utility class - prevent instantiation
    }

    /**
     * Creates index with automatic GPU detection and CPU fallback.
     * Uses Euclidean distance by default.
     * 
     * @param dimensions number of dimensions per vector
     * @return VectorIndex optimized for the current hardware
     * @throws IllegalArgumentException if dimensions <= 0
     */
    public static VectorIndex auto(int dimensions) {
        return auto(dimensions, DistanceMetric.EUCLIDEAN);
    }

    /**
     * Creates index with automatic GPU detection and CPU fallback.
     * 
     * <p>Routing logic based on Phase 2.5 validation:
     * <ol>
     *   <li>If CUDA available → GPU (persistent memory = 5x speedup)</li>
     *   <li>If GPU init fails → CPU fallback</li>
     *   <li>If no CUDA → CPU</li>
     * </ol>
     * 
     * @param dimensions number of dimensions per vector
     * @param metric distance metric to use
     * @return VectorIndex optimized for the current hardware
     * @throws IllegalArgumentException if dimensions <= 0 or metric is null
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
            } catch (Exception e) {
                logger.warn("GPU initialization failed, falling back to CPU: {}", e.getMessage());
            }
        } else {
            logger.info("CUDA not available, using CPU index");
        }

        // Fallback to CPU
        return cpu(dimensions);
    }

    /**
     * Creates CPU-only index using brute-force search.
     * 
     * <p>Use when:
     * <ul>
     *   <li>CUDA is not available</li>
     *   <li>Dataset is very small (< 1000 vectors)</li>
     *   <li>Testing or development without GPU</li>
     * </ul>
     * 
     * <p>Note: CPU index currently only supports Euclidean distance.
     * 
     * @param dimensions number of dimensions per vector
     * @return CPUVectorIndex instance
     * @throws IllegalArgumentException if dimensions <= 0
     */
    public static VectorIndex cpu(int dimensions) {
        validateDimensions(dimensions);
        logger.debug("Creating CPU index with {} dimensions", dimensions);
        return new CPUVectorIndex(dimensions);
    }

    /**
     * Creates GPU-only index using JCuda with persistent memory.
     * Uses Euclidean distance by default.
     * 
     * @param dimensions number of dimensions per vector
     * @return GPUVectorIndex instance with persistent memory
     * @throws IllegalArgumentException if dimensions <= 0
     * @throws UnsupportedOperationException if CUDA is not available
     */
    public static VectorIndex gpu(int dimensions) {
        return gpu(dimensions, DistanceMetric.EUCLIDEAN);
    }

    /**
     * Creates GPU-only index using JCuda with persistent memory and specified metric.
     * 
     * <p>The GPU index stores vectors in GPU memory once and only transfers
     * query vectors for each search. This eliminates the memory transfer
     * overhead that dominates single-query performance.
     * 
     * <p>Use when:
     * <ul>
     *   <li>CUDA is available and you want guaranteed GPU execution</li>
     *   <li>Processing many queries against a stable dataset</li>
     *   <li>Dataset fits in GPU memory</li>
     * </ul>
     * 
     * @param dimensions number of dimensions per vector
     * @param metric distance metric to use (EUCLIDEAN, COSINE, or INNER_PRODUCT)
     * @return GPUVectorIndex instance with persistent memory
     * @throws IllegalArgumentException if dimensions <= 0 or metric is null
     * @throws UnsupportedOperationException if CUDA is not available
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

    /**
     * Creates a hybrid index that routes queries based on dataset characteristics.
     * 
     * <p>This is a future enhancement that will dynamically choose GPU or CPU
     * based on query patterns and dataset size.
     * 
     * @param dimensions number of dimensions per vector
     * @return HybridVectorIndex (not yet implemented)
     * @throws UnsupportedOperationException always (not yet implemented)
     */
    public static VectorIndex hybrid(int dimensions) {
        validateDimensions(dimensions);
        throw new UnsupportedOperationException("Hybrid index not yet implemented - use auto() instead");
    }

    private static void validateDimensions(int dimensions) {
        if (dimensions <= 0) {
            throw new IllegalArgumentException("Dimensions must be positive, got: " + dimensions);
        }
    }

    private static void validateMetric(DistanceMetric metric) {
        if (metric == null) {
            throw new IllegalArgumentException("Distance metric cannot be null");
        }
    }
}
