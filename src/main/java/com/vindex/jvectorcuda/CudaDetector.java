package com.vindex.jvectorcuda;

import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;

/**
 * Detects CUDA availability and validates GPU capabilities.
 *
 * @author JVectorCUDA Team
 * @since 1.0.0
 */
public final class CudaDetector {

    private static final int MIN_COMPUTE_MAJOR = 6; // GTX 1060+
    private static final int MIN_COMPUTE_MINOR = 1;

    private static Boolean cudaAvailable = null;

    private CudaDetector() {
        // Utility class
    }

    /**
     * Checks if CUDA is available and meets minimum requirements.
     * Results are cached after first call.
     *
     * @return true if CUDA is available and compatible
     */
    public static synchronized boolean isAvailable() {
        if (cudaAvailable != null) {
            return cudaAvailable;
        }

        try {
            JCuda.setExceptionsEnabled(true);

            int[] deviceCount = new int[1];
            JCuda.cudaGetDeviceCount(deviceCount);

            if (deviceCount[0] == 0) {
                cudaAvailable = false;
                return false;
            }

            // Check compute capability
            cudaDeviceProp prop = new cudaDeviceProp();
            JCuda.cudaGetDeviceProperties(prop, 0);

            boolean compatible = prop.major > MIN_COMPUTE_MAJOR ||
                    (prop.major == MIN_COMPUTE_MAJOR && prop.minor >= MIN_COMPUTE_MINOR);

            if (!compatible) {
                System.err.println(String.format(
                        "GPU compute capability %d.%d is below minimum required %d.%d",
                        prop.major, prop.minor, MIN_COMPUTE_MAJOR, MIN_COMPUTE_MINOR));
                cudaAvailable = false;
                return false;
            }

            cudaAvailable = true;
            return true;

        } catch (Exception e) {
            cudaAvailable = false;
            return false;
        }
    }

    /**
     * Gets GPU information for debugging.
     *
     * @return GPU info string or error message
     */
    public static String getGpuInfo() {
        if (!isAvailable()) {
            return "CUDA not available";
        }

        try {
            cudaDeviceProp prop = new cudaDeviceProp();
            JCuda.cudaGetDeviceProperties(prop, 0);

            return String.format("GPU: %s, Compute: %d.%d, Memory: %d MB",
                    prop.getName(),
                    prop.major,
                    prop.minor,
                    prop.totalGlobalMem / (1024 * 1024));
        } catch (Exception e) {
            return "Error getting GPU info: " + e.getMessage();
        }
    }
}
