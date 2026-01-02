package com.vindex.jvectorcuda;

import com.vindex.jvectorcuda.cpu.CPUVectorIndex;
import com.vindex.jvectorcuda.gpu.GPUVectorIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

// Factory for creating VectorIndex instances with auto GPU/CPU detection.
public final class VectorIndexFactory {

    private static final Logger logger = LoggerFactory.getLogger(VectorIndexFactory.class);
    
    private VectorIndexFactory() {
        // Utility class - prevent instantiation
    }

    public static VectorIndex auto(int dimensions) {
        return auto(dimensions, DistanceMetric.EUCLIDEAN);
    }

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

    public static VectorIndex cpu(int dimensions) {
        validateDimensions(dimensions);
        logger.debug("Creating CPU index with {} dimensions", dimensions);
        return new CPUVectorIndex(dimensions);
    }

    public static VectorIndex gpu(int dimensions) {
        return gpu(dimensions, DistanceMetric.EUCLIDEAN);
    }

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

    private static void validateMetric(DistanceMetric metric) {
        if (metric == null) {
            throw new IllegalArgumentException("Distance metric cannot be null");
        }
    }
}
