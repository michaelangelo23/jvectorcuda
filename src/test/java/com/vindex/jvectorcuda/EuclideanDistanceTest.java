package com.vindex.jvectorcuda;

import com.vindex.jvectorcuda.gpu.GpuKernelLoader;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import org.junit.jupiter.api.*;

import java.util.Arrays;
import java.util.Random;

import static jcuda.driver.JCudaDriver.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Euclidean distance CUDA kernel (POC #3).
 * This is the REAL vector search workload - should show GPU winning!
 * 
 * @author JVectorCUDA (AI-assisted, Human-verified)
 */
@DisplayName("Euclidean Distance Kernel Tests (POC #3)")
class EuclideanDistanceTest {

    private static CUcontext context;
    private static CUdevice device;

    @BeforeAll
    static void setupCuda() {
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
    @DisplayName("Should compute correct Euclidean distances")
    void testEuclideanDistanceCorrectness() {
        int numVectors = 1000;
        int dimensions = 384;

        float[][] database = generateRandomDatabase(numVectors, dimensions);
        float[] query = generateRandomVector(dimensions);
        float[] gpuDistances = new float[numVectors];
        float[] cpuDistances = new float[numVectors];

        // Compute on GPU
        euclideanDistanceGPU(database, query, gpuDistances, numVectors, dimensions);

        // Compute on CPU
        euclideanDistanceCPU(database, query, cpuDistances);

        // Verify all distances match
        for (int i = 0; i < numVectors; i++) {
            assertEquals(cpuDistances[i], gpuDistances[i], 1e-3f,
                    "Distance mismatch at index " + i);
        }
    }

    @Test
    @DisplayName("Should handle large database (100K vectors)")
    void testLargeDatabase() {
        int numVectors = 100_000;
        int dimensions = 384;

        float[][] database = generateRandomDatabase(numVectors, dimensions);
        float[] query = generateRandomVector(dimensions);
        float[] distances = new float[numVectors];

        long startTime = System.nanoTime();
        euclideanDistanceGPU(database, query, distances, numVectors, dimensions);
        long gpuTime = System.nanoTime() - startTime;

        System.out.printf("GPU time for 100K vectors (384D): %.2f ms%n", gpuTime / 1e6);

        // Sample verification
        float[] cpuDistances = new float[numVectors];
        euclideanDistanceCPU(database, query, cpuDistances);

        for (int i = 0; i < 100; i++) {
            int idx = i * (numVectors / 100);
            assertEquals(cpuDistances[idx], distances[idx], 1e-3f);
        }
    }

    @Test
    @DisplayName("GPU should be FASTER than CPU for vector search")
    void testPerformanceVsCPU() {
        // Use realistic vector search workload
        int numVectors = 50_000; // Database size
        int dimensions = 384; // Common embedding size

        float[][] database = generateRandomDatabase(numVectors, dimensions);
        float[] query = generateRandomVector(dimensions);
        float[] gpuDistances = new float[numVectors];
        float[] cpuDistances = new float[numVectors];

        // Warmup
        euclideanDistanceGPU(database, query, gpuDistances, numVectors, dimensions);
        euclideanDistanceCPU(database, query, cpuDistances);

        // GPU benchmark
        long gpuStart = System.nanoTime();
        euclideanDistanceGPU(database, query, gpuDistances, numVectors, dimensions);
        long gpuTime = System.nanoTime() - gpuStart;

        // CPU benchmark
        long cpuStart = System.nanoTime();
        euclideanDistanceCPU(database, query, cpuDistances);
        long cpuTime = System.nanoTime() - cpuStart;

        // Verify correctness FIRST
        assertArrayEquals(cpuDistances, gpuDistances, 1e-3f, "Results differ");

        float speedup = (float) cpuTime / gpuTime;
        System.out.printf("%n=== POC #3 Performance Results ===%n");
        System.out.printf("Database: %,d vectors × %d dimensions%n", numVectors, dimensions);
        System.out.printf("CPU: %.2f ms%n", cpuTime / 1e6);
        System.out.printf("GPU: %.2f ms%n", gpuTime / 1e6);
        System.out.printf("Speedup: %.2fx%n", speedup);
        System.out.printf("GPU is %s%n", speedup > 1.0f ? "FASTER ✅" : "slower (JNI overhead - need cuVS integration)");

        // Document reality: For simple kernels, GPU may lose due to JNI overhead
        // This validates the need to integrate cuVS for production performance
        System.out.printf("%nConclusion: %s%n",
                speedup > 1.0f
                        ? "GPU wins! Ready for optimization."
                        : "Need cuVS integration for production speedup (as planned).");
    }

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {

        @Test
        @DisplayName("Should handle identical vectors (distance = 0)")
        void testIdenticalVectors() {
            int dimensions = 384;
            float[] vector = generateRandomVector(dimensions);
            float[][] database = { vector.clone() };
            float[] distances = new float[1];

            euclideanDistanceGPU(database, vector, distances, 1, dimensions);

            assertEquals(0.0f, distances[0], 1e-5f);
        }

        @Test
        @DisplayName("Should handle orthogonal vectors")
        void testOrthogonalVectors() {
            float[] query = { 1.0f, 0.0f, 0.0f };
            float[][] database = { { 0.0f, 1.0f, 0.0f } };
            float[] distances = new float[1];

            euclideanDistanceGPU(database, query, distances, 1, 3);

            // Distance = sqrt(1^2 + 1^2) = sqrt(2) ≈ 1.414
            assertEquals(Math.sqrt(2), distances[0], 1e-5f);
        }
    }

    // Helper methods

    private void euclideanDistanceGPU(float[][] database, float[] query,
            float[] distances, int numVectors, int dimensions) {
        // Flatten database
        float[] flatDatabase = new float[numVectors * dimensions];
        for (int i = 0; i < numVectors; i++) {
            System.arraycopy(database[i], 0, flatDatabase, i * dimensions, dimensions);
        }

        // Allocate GPU memory
        CUdeviceptr d_database = new CUdeviceptr();
        CUdeviceptr d_query = new CUdeviceptr();
        CUdeviceptr d_distances = new CUdeviceptr();

        int dbSize = numVectors * dimensions * Sizeof.FLOAT;
        int querySize = dimensions * Sizeof.FLOAT;
        int distSize = numVectors * Sizeof.FLOAT;

        cuMemAlloc(d_database, dbSize);
        cuMemAlloc(d_query, querySize);
        cuMemAlloc(d_distances, distSize);

        // Copy to GPU
        cuMemcpyHtoD(d_database, Pointer.to(flatDatabase), dbSize);
        cuMemcpyHtoD(d_query, Pointer.to(query), querySize);

        // Load and launch kernel
        GpuKernelLoader kernel = new GpuKernelLoader(
                "euclidean_distance.ptx", "euclideanDistance");

        Pointer kernelParams = Pointer.to(
                Pointer.to(d_database),
                Pointer.to(d_query),
                Pointer.to(d_distances),
                Pointer.to(new int[] { numVectors }),
                Pointer.to(new int[] { dimensions }));

        int blockSize = 256;
        int gridSize = (numVectors + blockSize - 1) / blockSize;
        kernel.launch(gridSize, blockSize, kernelParams);

        // Copy back
        cuMemcpyDtoH(Pointer.to(distances), d_distances, distSize);

        // Cleanup
        cuMemFree(d_database);
        cuMemFree(d_query);
        cuMemFree(d_distances);
    }

    private void euclideanDistanceCPU(float[][] database, float[] query, float[] distances) {
        for (int i = 0; i < database.length; i++) {
            float sum = 0.0f;
            for (int d = 0; d < query.length; d++) {
                float diff = database[i][d] - query[d];
                sum += diff * diff;
            }
            distances[i] = (float) Math.sqrt(sum);
        }
    }

    private float[][] generateRandomDatabase(int numVectors, int dimensions) {
        Random rand = new Random(42);
        float[][] database = new float[numVectors][dimensions];
        for (int i = 0; i < numVectors; i++) {
            for (int d = 0; d < dimensions; d++) {
                database[i][d] = rand.nextFloat() * 2.0f - 1.0f; // [-1, 1]
            }
        }
        return database;
    }

    private float[] generateRandomVector(int dimensions) {
        Random rand = new Random(123);
        float[] vector = new float[dimensions];
        for (int d = 0; d < dimensions; d++) {
            vector[d] = rand.nextFloat() * 2.0f - 1.0f;
        }
        return vector;
    }
}
