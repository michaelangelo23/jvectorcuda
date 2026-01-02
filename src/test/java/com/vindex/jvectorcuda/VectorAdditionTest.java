package com.vindex.jvectorcuda;

import com.vindex.jvectorcuda.gpu.GpuKernelLoader;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import org.junit.jupiter.api.*;
import java.util.Random;

import static jcuda.driver.JCudaDriver.*;
import static org.junit.jupiter.api.Assertions.*;

// Tests for vector addition CUDA kernel (POC #2)
@DisplayName("Vector Addition Kernel Tests (POC #2)")
class VectorAdditionTest {

    private static CUcontext context;
    private static CUdevice device;

    @BeforeAll
    static void setupCuda() {
        // Initialize CUDA
        cuInit(0);

        device = new CUdevice();
        cuDeviceGet(device, 0);

        context = new CUcontext();
        cuCtxCreate(context, 0, device);
    }

    @AfterAll
    static void teardownCuda() {
        if (context != null) {
            cuCtxDestroy(context);
        }
    }

    @Test
    @DisplayName("Should add two vectors correctly on GPU")
    void testVectorAdditionCorrectness() {
        int n = 1000;
        float[] a = generateRandomArray(n);
        float[] b = generateRandomArray(n);
        float[] result = new float[n];

        // Execute on GPU
        vectorAddGPU(a, b, result, n);

        // Verify against CPU
        for (int i = 0; i < n; i++) {
            assertEquals(a[i] + b[i], result[i], 1e-5f,
                    "Mismatch at index " + i);
        }
    }

    @Test
    @DisplayName("Should handle large vectors (1M elements)")
    void testLargeVectors() {
        int n = 1_000_000;
        float[] a = generateRandomArray(n);
        float[] b = generateRandomArray(n);
        float[] result = new float[n];

        long startTime = System.nanoTime();
        vectorAddGPU(a, b, result, n);
        long gpuTime = System.nanoTime() - startTime;

        // Verify correctness (sample check)
        for (int i = 0; i < 100; i++) {
            int idx = i * (n / 100);
            assertEquals(a[idx] + b[idx], result[idx], 1e-5f);
        }

        System.out.printf("GPU time for 1M elements: %.2f ms%n", gpuTime / 1e6);
    }

    @Test
    @DisplayName("Should demonstrate GPU vs CPU performance characteristics")
    void testPerformanceVsCPU() {
        // Test with 10M elements - large enough for GPU to overcome JNI overhead
        int n = 10_000_000;
        float[] a = generateRandomArray(n);
        float[] b = generateRandomArray(n);
        float[] gpuResult = new float[n];
        float[] cpuResult = new float[n];

        // Warmup (JIT compilation)
        vectorAddCPU(a, b, cpuResult, 1000);
        vectorAddGPU(a, b, gpuResult, 1000);

        // GPU
        long gpuStart = System.nanoTime();
        vectorAddGPU(a, b, gpuResult, n);
        long gpuTime = System.nanoTime() - gpuStart;

        // CPU
        long cpuStart = System.nanoTime();
        vectorAddCPU(a, b, cpuResult, n);
        long cpuTime = System.nanoTime() - cpuStart;

        // Results should match
        assertArrayEquals(cpuResult, gpuResult, 1e-5f, "GPU and CPU results differ");

        float speedup = (float) cpuTime / gpuTime;
        System.out.printf("CPU: %.2f ms, GPU: %.2f ms, Speedup: %.2fx%n",
                cpuTime / 1e6, gpuTime / 1e6, speedup);

        // Note: For smaller datasets (<5M), CPU may be faster due to JNI/transfer overhead.
        // See PERFORMANCE.md for detailed analysis of GPU vs CPU trade-offs.
        System.out.printf("GPU is %s for this dataset size (%,d elements)%n",
                speedup > 1.0f ? "FASTER" : "slower (JNI overhead)", n);
    }

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {

        @Test
        @DisplayName("Should handle zero-filled vectors")
        void testZeroVectors() {
            int n = 100;
            float[] a = new float[n];
            float[] b = new float[n];
            float[] result = new float[n];

            vectorAddGPU(a, b, result, n);

            for (int i = 0; i < n; i++) {
                assertEquals(0.0f, result[i], 1e-10f);
            }
        }

        @Test
        @DisplayName("Should handle negative numbers")
        void testNegativeNumbers() {
            float[] a = { -1.0f, -2.0f, -3.0f };
            float[] b = { 1.0f, 2.0f, 3.0f };
            float[] result = new float[3];

            vectorAddGPU(a, b, result, 3);

            assertArrayEquals(new float[] { 0.0f, 0.0f, 0.0f }, result, 1e-5f);
        }

        @Test
        @DisplayName("Should handle non-multiple-of-blocksize length")
        void testNonMultipleOfBlockSize() {
            int n = 257; // Not a multiple of 256
            float[] a = generateRandomArray(n);
            float[] b = generateRandomArray(n);
            float[] result = new float[n];

            vectorAddGPU(a, b, result, n);

            for (int i = 0; i < n; i++) {
                assertEquals(a[i] + b[i], result[i], 1e-5f);
            }
        }
    }

    // Helper methods

    private void vectorAddGPU(float[] a, float[] b, float[] result, int n) {
        // Allocate GPU memory
        CUdeviceptr d_a = new CUdeviceptr();
        CUdeviceptr d_b = new CUdeviceptr();
        CUdeviceptr d_result = new CUdeviceptr();
        int size = n * Sizeof.FLOAT;

        cuMemAlloc(d_a, size);
        cuMemAlloc(d_b, size);
        cuMemAlloc(d_result, size);

        // Copy data to GPU
        cuMemcpyHtoD(d_a, Pointer.to(a), size);
        cuMemcpyHtoD(d_b, Pointer.to(b), size);

        // Load kernel and execute
        GpuKernelLoader kernel = new GpuKernelLoader("vector_add.ptx", "vectorAdd");

        Pointer kernelParams = Pointer.to(
                Pointer.to(d_a),
                Pointer.to(d_b),
                Pointer.to(d_result),
                Pointer.to(new int[] { n }));

        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        kernel.launch(gridSize, blockSize, kernelParams);

        // Copy result back
        cuMemcpyDtoH(Pointer.to(result), d_result, size);

        // Cleanup
        cuMemFree(d_a);
        cuMemFree(d_b);
        cuMemFree(d_result);
    }

    private void vectorAddCPU(float[] a, float[] b, float[] result, int n) {
        for (int i = 0; i < n; i++) {
            result[i] = a[i] + b[i];
        }
    }

    private float[] generateRandomArray(int n) {
        Random rand = new Random(42); // Fixed seed for reproducibility
        float[] arr = new float[n];
        for (int i = 0; i < n; i++) {
            arr[i] = rand.nextFloat() * 100.0f;
        }
        return arr;
    }
}
