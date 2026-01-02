package com.vindex.jvectorcuda;

import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;

// Detects CUDA availability and validates GPU capabilities.
public final class CudaDetector {

    private static final int MIN_COMPUTE_MAJOR = 6; // GTX 1060+
    private static final int MIN_COMPUTE_MINOR = 1;

    private static Boolean cudaAvailable = null;

    private CudaDetector() {
        // Utility class
    }

    // Checks if CUDA is available and meets minimum requirements (cached)
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

    // Returns GPU info string for debugging
    public static String getGpuInfo() {
        if (!isAvailable()) {
            return "CUDA not available";
        }

        try {
            cudaDeviceProp prop = new cudaDeviceProp();
            JCuda.cudaGetDeviceProperties(prop, 0);

            return String.format("%s (Compute %d.%d, %d MB VRAM, %d MHz)",
                    prop.getName(),
                    prop.major,
                    prop.minor,
                    prop.totalGlobalMem / (1024 * 1024),
                    prop.clockRate / 1000);
        } catch (Exception e) {
            return "Error getting GPU info: " + e.getMessage();
        }
    }

    // Returns detailed GPU specs for benchmarking
    public static String getDetailedGpuInfo() {
        if (!isAvailable()) {
            return "CUDA not available";
        }

        try {
            cudaDeviceProp prop = new cudaDeviceProp();
            JCuda.cudaGetDeviceProperties(prop, 0);

            StringBuilder info = new StringBuilder();
            info.append(String.format("%s\n", prop.getName()));
            info.append(String.format("  - Compute Capability: %d.%d\n", prop.major, prop.minor));
            info.append(String.format("  - Total VRAM: %d MB\n", prop.totalGlobalMem / (1024 * 1024)));
            info.append(String.format("  - Clock Rate: %d MHz\n", prop.clockRate / 1000));
            info.append(String.format("  - Multiprocessors: %d\n", prop.multiProcessorCount));
            info.append(String.format("  - Warp Size: %d\n", prop.warpSize));
            info.append(String.format("  - Max Threads per Block: %d\n", prop.maxThreadsPerBlock));
            
            return info.toString();
        } catch (Exception e) {
            return "Error getting GPU info: " + e.getMessage();
        }
    }
}
