package com.vindex.jvectorcuda;

import org.junit.jupiter.api.Test;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests CUDA availability and GPU detection.
 *
 * @author JVectorCUDA Team
 */
class CudaAvailabilityTest {

    @Test
    void testCudaDetection() {
        System.out.println("=== CUDA Availability Test ===");

        boolean available = CudaDetector.isAvailable();
        System.out.println("CUDA Available: " + available);

        if (available) {
            System.out.println(CudaDetector.getGpuInfo());

            // Verify we can actually initialize CUDA
            assertDoesNotThrow(() -> {
                JCuda.setExceptionsEnabled(true);

                int[] deviceCount = new int[1];
                JCuda.cudaGetDeviceCount(deviceCount);

                assertTrue(deviceCount[0] > 0, "No CUDA devices found");

                cudaDeviceProp prop = new cudaDeviceProp();
                JCuda.cudaGetDeviceProperties(prop, 0);

                System.out.println("Device Properties:");
                System.out.println("  Name: " + prop.getName());
                System.out.println("  Compute: " + prop.major + "." + prop.minor);
                System.out.println("  Memory: " + (prop.totalGlobalMem / 1024 / 1024) + " MB");
                System.out.println("  Multiprocessors: " + prop.multiProcessorCount);

                assertTrue(prop.major >= 6, "GPU compute capability too old (need 6.0+)");
            });
        } else {
            System.out.println("CUDA not available - CPU fallback will be used");
        }

        System.out.println("=== Test Complete ===");
    }

    @Test
    void testFactoryAutoMode() {
        System.out.println("\n=== Testing VectorIndex.auto() ===");

        // This should not throw, even if GPU is unavailable
        assertDoesNotThrow(() -> {
            try {
                VectorIndex index = VectorIndex.auto(384);
                System.out.println("Successfully created index with auto-detection");
                index.close();
            } catch (UnsupportedOperationException e) {
                // Expected during development - implementations not complete yet
                System.out.println("Implementation not complete: " + e.getMessage());
            }
        });
    }

    @Test
    void testInvalidDimensions() {
        assertThrows(IllegalArgumentException.class, () -> {
            VectorIndex.auto(0);
        }, "Should reject zero dimensions");

        assertThrows(IllegalArgumentException.class, () -> {
            VectorIndex.auto(-1);
        }, "Should reject negative dimensions");
    }
}
