package com.vindex.jvectorcuda.gpu;

import com.vindex.jvectorcuda.CudaDetector;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUresult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static jcuda.driver.JCudaDriver.*;

/**
 * Utility class for querying and calculating GPU VRAM requirements.
 * Helps tests and applications determine safe dataset sizes based on available memory.
 */
public final class VramUtil {
    
    private static final Logger logger = LoggerFactory.getLogger(VramUtil.class);
    
    // Memory overhead factor (buffers, kernel state, etc.)
    private static final double OVERHEAD_FACTOR = 1.2;
    
    // Safe usage percentage (leave room for other GPU processes)
    private static final double SAFE_USAGE_PERCENT = 0.7;
    
    private VramUtil() {
        // Utility class
    }
    
    /**
     * Calculates the memory required for a dataset in bytes.
     * Formula: vectors × dimensions × 4 bytes × 1.2 (overhead)
     *
     * @param numVectors Number of vectors
     * @param dimensions Dimensions per vector
     * @return Required memory in bytes (including overhead)
     */
    public static long calculateRequiredBytes(int numVectors, int dimensions) {
        long baseBytes = (long) numVectors * dimensions * Sizeof.FLOAT;
        return (long) (baseBytes * OVERHEAD_FACTOR);
    }
    
    /**
     * Calculates the memory required for a dataset in megabytes.
     *
     * @param numVectors Number of vectors
     * @param dimensions Dimensions per vector
     * @return Required memory in MB (including overhead)
     */
    public static double calculateRequiredMB(int numVectors, int dimensions) {
        return calculateRequiredBytes(numVectors, dimensions) / (1024.0 * 1024.0);
    }
    
    /**
     * Queries the available (free) GPU memory.
     * Returns -1 if CUDA is not available or query fails.
     *
     * @return Free GPU memory in bytes, or -1 if unavailable
     */
    public static long getAvailableVramBytes() {
        if (!CudaDetector.isAvailable()) {
            return -1;
        }
        
        try {
            cuInit(0);
            CUdevice device = new CUdevice();
            cuDeviceGet(device, 0);
            CUcontext context = new CUcontext();
            cuCtxCreate(context, 0, device);
            
            long[] free = new long[1];
            long[] total = new long[1];
            
            int result = cuMemGetInfo(free, total);
            cuCtxDestroy(context);
            
            if (result != CUresult.CUDA_SUCCESS) {
                logger.warn("Failed to query GPU memory: {}", CUresult.stringFor(result));
                return -1;
            }
            
            return free[0];
        } catch (Exception e) {
            logger.warn("Error querying GPU memory: {}", e.getMessage());
            return -1;
        }
    }
    
    /**
     * Queries the total GPU memory.
     * Returns -1 if CUDA is not available or query fails.
     *
     * @return Total GPU memory in bytes, or -1 if unavailable
     */
    public static long getTotalVramBytes() {
        if (!CudaDetector.isAvailable()) {
            return -1;
        }
        
        try {
            cuInit(0);
            CUdevice device = new CUdevice();
            cuDeviceGet(device, 0);
            CUcontext context = new CUcontext();
            cuCtxCreate(context, 0, device);
            
            long[] free = new long[1];
            long[] total = new long[1];
            
            int result = cuMemGetInfo(free, total);
            cuCtxDestroy(context);
            
            if (result != CUresult.CUDA_SUCCESS) {
                return -1;
            }
            
            return total[0];
        } catch (Exception e) {
            return -1;
        }
    }
    
    /**
     * Calculates the maximum safe number of vectors for the current GPU.
     * Uses 70% of available VRAM to leave room for other processes.
     *
     * @param dimensions Dimensions per vector
     * @return Maximum safe vector count, or -1 if GPU unavailable
     */
    public static int getMaxSafeVectorCount(int dimensions) {
        long availableBytes = getAvailableVramBytes();
        if (availableBytes <= 0) {
            return -1;
        }
        
        // Use 70% of available memory
        long safeBytes = (long) (availableBytes * SAFE_USAGE_PERCENT);
        
        // Account for overhead in calculation
        long bytesPerVector = (long) (dimensions * Sizeof.FLOAT * OVERHEAD_FACTOR);
        
        return (int) Math.min(safeBytes / bytesPerVector, Integer.MAX_VALUE);
    }
    
    /**
     * Scales a desired vector count to fit within available GPU memory.
     * Returns the smaller of: requested count or max safe count.
     *
     * @param desiredCount Desired number of vectors
     * @param dimensions Dimensions per vector
     * @return Scaled vector count that fits in VRAM, or desiredCount if GPU unavailable
     */
    public static int scaleToAvailableVram(int desiredCount, int dimensions) {
        int maxSafe = getMaxSafeVectorCount(dimensions);
        
        if (maxSafe <= 0) {
            // GPU unavailable, return desired count (test will be skipped anyway)
            return desiredCount;
        }
        
        if (desiredCount > maxSafe) {
            logger.info("Scaling test from {} to {} vectors to fit in available VRAM ({} MB free)",
                desiredCount, maxSafe, getAvailableVramBytes() / (1024 * 1024));
            return maxSafe;
        }
        
        return desiredCount;
    }
    
    /**
     * Checks if a dataset will fit in available GPU memory.
     *
     * @param numVectors Number of vectors
     * @param dimensions Dimensions per vector
     * @return true if dataset fits, false otherwise (or if GPU unavailable)
     */
    public static boolean willFitInVram(int numVectors, int dimensions) {
        long available = getAvailableVramBytes();
        if (available <= 0) {
            return false;
        }
        
        long required = calculateRequiredBytes(numVectors, dimensions);
        return required <= available * SAFE_USAGE_PERCENT;
    }
    
    /**
     * Prints VRAM status to stdout (useful for debugging/benchmarks).
     */
    public static void printVramStatus() {
        long available = getAvailableVramBytes();
        long total = getTotalVramBytes();
        
        if (available < 0 || total < 0) {
            System.out.println("GPU VRAM: Not available");
            return;
        }
        
        double availableMB = available / (1024.0 * 1024.0);
        double totalMB = total / (1024.0 * 1024.0);
        double usedMB = totalMB - availableMB;
        double usedPercent = (usedMB / totalMB) * 100;
        
        System.out.printf("GPU VRAM: %.0f MB free / %.0f MB total (%.1f%% used)%n",
            availableMB, totalMB, usedPercent);
        System.out.printf("Safe test capacity (70%%): %,d vectors @ 384 dims%n",
            getMaxSafeVectorCount(384));
    }
}
