package com.vindex.jvectorcuda.benchmarks;

import com.vindex.jvectorcuda.gpu.GpuKernelLoader;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import org.junit.jupiter.api.*;

import java.util.Random;

import static jcuda.driver.JCudaDriver.*;

// Benchmarks to find when GPU becomes faster than CPU for vector search
@DisplayName("GPU Break-Even Point Benchmarks")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class GpuBreakEvenTest {

    private static final int DIMENSIONS = 384; // Common embedding size
    private static final Random random = new Random(42);
    
    private static CUcontext context;
    private static CUdevice device;

    private GpuKernelLoader kernelLoader;

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

    @BeforeEach
    void setUp() {
        kernelLoader = new GpuKernelLoader("euclidean_distance.ptx", "euclideanDistance");
    }

    @AfterEach
    void tearDown() {
        // Kernel loader resources are managed internally
    }

    // Test 1: Find GPU break-even point across dataset sizes and batch sizes
    @Test
    @Order(1)
    @DisplayName("Find GPU break-even point (dataset size x batch size)")
    void testBreakEvenPoints() {
        System.out.println("\n=== GPU Break-Even Point Analysis ===\n");
        System.out.println("Testing when GPU starts outperforming CPU...\n");
        
        int[] vectorCounts = {1_000, 10_000, 50_000, 100_000};
        int[] batchSizes = {1, 10, 100};
        
        System.out.printf("%-15s %-12s %-12s %-12s %-10s %-8s%n",
                "Vectors", "Batch", "CPU (ms)", "GPU (ms)", "Speedup", "Winner");
        System.out.println("-".repeat(75));
        
        for (int numVectors : vectorCounts) {
            for (int batchSize : batchSizes) {
                // Skip very large tests to keep runtime reasonable
                if (numVectors >= 100_000 && batchSize > 10) {
                    continue;
                }
                
                BenchmarkResult result = runBenchmark(numVectors, DIMENSIONS, batchSize);
                String winner = result.speedup >= 1.0f ? "GPU" : "CPU";
                
                System.out.printf("%-15s %-12d %-12.2f %-12.2f %-10.2fx %-8s%n",
                        formatNumber(numVectors),
                        batchSize,
                        result.cpuTimeMs,
                        result.gpuTimeMs,
                        result.speedup,
                        winner);
            }
            System.out.println();
        }
        
        System.out.println("Note: GPU speedup depends on JNI overhead and memory transfer costs.");
    }

    // Test 2: Measure memory transfer overhead percentage
    @Test
    @Order(2)
    @DisplayName("Measure memory transfer overhead percentage")
    void testMemoryTransferOverhead() {
        System.out.println("\n=== Memory Transfer Overhead Analysis ===\n");
        
        int numVectors = 50_000;
        float[][] database = generateRandomDatabase(numVectors, DIMENSIONS);
        float[] query = generateRandomVector(DIMENSIONS);
        
        // Measure total GPU time (includes transfer)
        long totalStart = System.nanoTime();
        @SuppressWarnings("unused")
        float[] gpuResult = euclideanDistanceGPU(database, query, numVectors, DIMENSIONS);
        long totalTime = System.nanoTime() - totalStart;
        
        // Measure transfer time only (upload database + query, download results)
        long transferStart = System.nanoTime();
        CUdeviceptr d_database = uploadToGPU(flattenDatabase(database, numVectors, DIMENSIONS));
        CUdeviceptr d_query = uploadToGPU(query);
        @SuppressWarnings("unused")
        float[] dummy = downloadFromGPU(d_database, numVectors * DIMENSIONS);
        long transferTime = System.nanoTime() - transferStart;
        cuMemFree(d_database);
        cuMemFree(d_query);
        
        // Compute time = total - transfer
        long computeTime = totalTime - transferTime;
        
        double transferPct = 100.0 * transferTime / totalTime;
        double computePct = 100.0 * computeTime / totalTime;
        
        System.out.printf("Total GPU time:     %.2f ms%n", totalTime / 1e6);
        System.out.printf("Transfer time:      %.2f ms (%.1f%%)%n", transferTime / 1e6, transferPct);
        System.out.printf("Compute time:       %.2f ms (%.1f%%)%n", computeTime / 1e6, computePct);
        System.out.println();
        
        if (transferPct > 50) {
            System.out.println("WARNING: Transfer overhead dominates! (>" + 50 + "%)");
            System.out.println("         GPU may not be beneficial for this workload");
        } else {
            System.out.println("GOOD: Compute time dominates - GPU can provide speedup");
        }
        
        System.out.println("\nConclusion: For persistent data scenarios, keep database on GPU.");
    }

    // Test 3: Persistent GPU memory - upload database once, query many times
    @Test
    @Order(3)
    @DisplayName("Test persistent GPU memory (amortize transfer cost)")
    void testPersistentGpuMemory() {
        System.out.println("\n=== Persistent GPU Memory Test ===\n");
        System.out.println("Scenario: Upload database once, run many queries\n");
        
        int numVectors = 50_000;
        int numQueries = 100;
        
        float[][] database = generateRandomDatabase(numVectors, DIMENSIONS);
        float[][] queries = new float[numQueries][];
        for (int i = 0; i < numQueries; i++) {
            queries[i] = generateRandomVector(DIMENSIONS);
        }
        
        // CPU: Run all queries
        long cpuStart = System.nanoTime();
        for (int i = 0; i < numQueries; i++) {
            euclideanDistanceCPU(database, queries[i]);
        }
        long cpuTime = System.nanoTime() - cpuStart;
        
        // GPU: Upload database once, query many times
        long gpuStart = System.nanoTime();
        
        // Upload database once
        CUdeviceptr d_database = uploadToGPU(flattenDatabase(database, numVectors, DIMENSIONS));
        CUdeviceptr d_distances = new CUdeviceptr();
        cuMemAlloc(d_distances, (long) numVectors * Sizeof.FLOAT);
        
        // Query many times (only upload query each time)
        for (int i = 0; i < numQueries; i++) {
            CUdeviceptr d_query = uploadToGPU(queries[i]);
            
            // Launch kernel
            int blockSize = 256;
            int gridSize = (numVectors + blockSize - 1) / blockSize;
            Pointer params = Pointer.to(
                    Pointer.to(d_database),
                    Pointer.to(d_query),
                    Pointer.to(d_distances),
                    Pointer.to(new int[]{numVectors}),
                    Pointer.to(new int[]{DIMENSIONS})
            );
            kernelLoader.launch(gridSize, blockSize, params);
            
            cuMemFree(d_query);
        }
        
        cuMemFree(d_database);
        cuMemFree(d_distances);
        
        long gpuTime = System.nanoTime() - gpuStart;
        float speedup = (float) cpuTime / gpuTime;
        
        System.out.printf("CPU: %d queries x %,d vectors = %.2f ms%n",
                numQueries, numVectors, cpuTime / 1e6);
        System.out.printf("GPU: %d queries x %,d vectors = %.2f ms%n",
                numQueries, numVectors, gpuTime / 1e6);
        System.out.printf("Speedup: %.2fx%n", speedup);
        System.out.println();
        
        if (speedup > 1.0f) {
            System.out.println("SUCCESS: GPU is faster with persistent memory!");
            System.out.printf("Per-query latency: %.2fms%n", (gpuTime / 1e6) / numQueries);
        } else {
            System.out.println("NOTE: Even with persistent memory, GPU is not faster on this hardware");
            System.out.println("      This is expected on GTX 1080 Max-Q due to JNI overhead");
        }
        
        System.out.println("\nThis data helps determine when to route queries to GPU vs CPU.");
    }

    // Test 4: Different dimensions (scaled vector counts to fit GPU memory)
    @Test
    @Order(4)
    @DisplayName("Test different vector dimensions (128, 384, 768, 1536)")
    void testDifferentDimensions() {
        System.out.println("\n=== Dimension Impact on GPU Performance ===\n");
        
        // Scale vector count based on dimensions to stay within memory limits
        // GTX 1080 Max-Q has 8GB but shared with other processes
        int[][] dimAndVectors = {
            {128, 50_000},   // 128D x 50K = ~24 MB
            {384, 50_000},   // 384D x 50K = ~73 MB
            {768, 25_000},   // 768D x 25K = ~73 MB (reduced)
            {1536, 10_000}   // 1536D x 10K = ~58 MB (reduced for safety)
        };
        
        System.out.printf("%-15s %-12s %-12s %-12s %-10s%n",
                "Dimensions", "Vectors", "CPU (ms)", "GPU (ms)", "Speedup");
        System.out.println("-".repeat(65));
        
        for (int[] config : dimAndVectors) {
            int dim = config[0];
            int numVectors = config[1];
            
            try {
                float[][] database = generateRandomDatabase(numVectors, dim);
                float[] query = generateRandomVector(dim);
                
                // CPU
                long cpuStart = System.nanoTime();
                euclideanDistanceCPU(database, query);
                long cpuTime = System.nanoTime() - cpuStart;
                
                // GPU
                long gpuStart = System.nanoTime();
                euclideanDistanceGPU(database, query, numVectors, dim);
                long gpuTime = System.nanoTime() - gpuStart;
                
                float speedup = (float) cpuTime / gpuTime;
                
                System.out.printf("%-15d %-12s %-12.2f %-12.2f %-10.2fx%n",
                        dim, formatNumber(numVectors), cpuTime / 1e6, gpuTime / 1e6, speedup);
            } catch (Exception e) {
                System.out.printf("%-15d %-12s %-12s %-12s (skipped: %s)%n",
                        dim, formatNumber(numVectors), "-", "-", e.getMessage());
            }
        }
        
        System.out.println("\nHigher dimensions = more compute per vector = better GPU utilization");
        System.out.println("Note: Vector counts scaled to stay within GPU memory limits.");
    }

    // ==================== Helper Methods ====================

    private BenchmarkResult runBenchmark(int numVectors, int dimensions, int batchSize) {
        float[][] database = generateRandomDatabase(numVectors, dimensions);
        float[][] queries = new float[batchSize][];
        for (int i = 0; i < batchSize; i++) {
            queries[i] = generateRandomVector(dimensions);
        }
        
        // CPU benchmark
        long cpuStart = System.nanoTime();
        for (float[] query : queries) {
            euclideanDistanceCPU(database, query);
        }
        long cpuTime = System.nanoTime() - cpuStart;
        
        // GPU benchmark
        long gpuStart = System.nanoTime();
        for (float[] query : queries) {
            euclideanDistanceGPU(database, query, numVectors, dimensions);
        }
        long gpuTime = System.nanoTime() - gpuStart;
        
        return new BenchmarkResult(cpuTime / 1e6, gpuTime / 1e6);
    }

    private float[] euclideanDistanceGPU(float[][] database, float[] query, int numVectors, int dimensions) {
        float[] flatDatabase = flattenDatabase(database, numVectors, dimensions);
        float[] distances = new float[numVectors];
        
        CUdeviceptr d_database = new CUdeviceptr();
        CUdeviceptr d_query = new CUdeviceptr();
        CUdeviceptr d_distances = new CUdeviceptr();
        
        long dbSize = (long) numVectors * dimensions * Sizeof.FLOAT;
        int querySize = dimensions * Sizeof.FLOAT;
        long distSize = (long) numVectors * Sizeof.FLOAT;
        
        cuMemAlloc(d_database, dbSize);
        cuMemAlloc(d_query, querySize);
        cuMemAlloc(d_distances, distSize);
        
        cuMemcpyHtoD(d_database, Pointer.to(flatDatabase), dbSize);
        cuMemcpyHtoD(d_query, Pointer.to(query), querySize);
        
        int blockSize = 256;
        int gridSize = (numVectors + blockSize - 1) / blockSize;
        Pointer params = Pointer.to(
                Pointer.to(d_database),
                Pointer.to(d_query),
                Pointer.to(d_distances),
                Pointer.to(new int[]{numVectors}),
                Pointer.to(new int[]{dimensions})
        );
        kernelLoader.launch(gridSize, blockSize, params);
        
        cuMemcpyDtoH(Pointer.to(distances), d_distances, distSize);
        
        cuMemFree(d_database);
        cuMemFree(d_query);
        cuMemFree(d_distances);
        
        return distances;
    }

    private float[] euclideanDistanceCPU(float[][] database, float[] query) {
        float[] distances = new float[database.length];
        for (int i = 0; i < database.length; i++) {
            float sum = 0.0f;
            for (int d = 0; d < query.length; d++) {
                float diff = database[i][d] - query[d];
                sum += diff * diff;
            }
            distances[i] = (float) Math.sqrt(sum);
        }
        return distances;
    }

    private CUdeviceptr uploadToGPU(float[] data) {
        CUdeviceptr devicePtr = new CUdeviceptr();
        int size = data.length * Sizeof.FLOAT;
        cuMemAlloc(devicePtr, size);
        cuMemcpyHtoD(devicePtr, Pointer.to(data), size);
        return devicePtr;
    }

    private float[] downloadFromGPU(CUdeviceptr devicePtr, int length) {
        float[] result = new float[length];
        cuMemcpyDtoH(Pointer.to(result), devicePtr, length * Sizeof.FLOAT);
        return result;
    }

    private float[] flattenDatabase(float[][] database, int numVectors, int dimensions) {
        float[] flat = new float[numVectors * dimensions];
        for (int i = 0; i < numVectors; i++) {
            System.arraycopy(database[i], 0, flat, i * dimensions, dimensions);
        }
        return flat;
    }

    private float[][] generateRandomDatabase(int numVectors, int dimensions) {
        float[][] database = new float[numVectors][dimensions];
        for (int i = 0; i < numVectors; i++) {
            for (int d = 0; d < dimensions; d++) {
                database[i][d] = random.nextFloat();
            }
        }
        return database;
    }

    private float[] generateRandomVector(int dimensions) {
        float[] vector = new float[dimensions];
        for (int d = 0; d < dimensions; d++) {
            vector[d] = random.nextFloat();
        }
        return vector;
    }

    private String formatNumber(int number) {
        if (number >= 1_000_000) {
            return (number / 1_000_000) + "M";
        } else if (number >= 1_000) {
            return (number / 1_000) + "K";
        }
        return String.valueOf(number);
    }

    // Simple result container for benchmark measurements
    private static class BenchmarkResult {
        final double cpuTimeMs;
        final double gpuTimeMs;
        final float speedup;

        BenchmarkResult(double cpuTimeMs, double gpuTimeMs) {
            this.cpuTimeMs = cpuTimeMs;
            this.gpuTimeMs = gpuTimeMs;
            this.speedup = (float) (cpuTimeMs / gpuTimeMs);
        }
    }
}
