package com.vindex.jvectorcuda;

/**
 * Factory for creating VectorIndex instances with automatic GPU/CPU detection.
 *
 * @author JVectorCUDA Team
 * @since 1.0.0
 */
final class VectorIndexFactory {

    private VectorIndexFactory() {
        // Utility class - prevent instantiation
    }

    /**
     * Creates index with automatic GPU detection and CPU fallback.
     */
    static VectorIndex auto(int dimensions) {
        validateDimensions(dimensions);

        // Try GPU first
        if (CudaDetector.isAvailable()) {
            try {
                return gpu(dimensions);
            } catch (Exception e) {
                System.err.println("GPU initialization failed, falling back to CPU: " + e.getMessage());
            }
        }

        // Fallback to CPU
        return cpu(dimensions);
    }

    /**
     * Creates CPU-only index using JVector.
     */
    static VectorIndex cpu(int dimensions) {
        validateDimensions(dimensions);
        throw new UnsupportedOperationException("CPU implementation not yet available");
    }

    /**
     * Creates GPU-only index using JCuda.
     */
    static VectorIndex gpu(int dimensions) {
        validateDimensions(dimensions);

        if (!CudaDetector.isAvailable()) {
            throw new UnsupportedOperationException("CUDA is not available on this system");
        }

        throw new UnsupportedOperationException("GPU implementation not yet available");
    }

    private static void validateDimensions(int dimensions) {
        if (dimensions <= 0) {
            throw new IllegalArgumentException("Dimensions must be positive, got: " + dimensions);
        }
    }
}
