package com.vindex.jvectorcuda.benchmarks;

import com.vindex.jvectorcuda.CudaDetector;
import com.vindex.jvectorcuda.gpu.GpuKernelLoader;
import com.vindex.jvectorcuda.gpu.VramUtil;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import org.junit.jupiter.api.*;

import java.util.Random;

import static jcuda.driver.JCudaDriver.*;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Performance benchmarks for JVectorCUDA GPU acceleration.
 * 
 * <p>
 * These benchmarks help determine when GPU acceleration is beneficial
 * compared to CPU-only execution. Results vary by hardware - contributors
 * are encouraged to run these tests and share results.
 * 
 * <p>
 * <b>Key findings (hardware-dependent):</b>
 * <ul>
 * <li>Single queries: CPU usually wins (no transfer overhead)</li>
 * <li>Batch queries with persistent memory: GPU wins (amortized transfer)</li>
 * <li>Break-even point varies by GPU generation and PCIe bandwidth</li>
 * </ul>
 * 
 * <p>
 * Run with: {@code ./gradlew test --tests "*PerformanceBenchmarkTest"}
 */
@DisplayName("Performance Benchmarks")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class PerformanceBenchmarkTest {

    private static final int DIMENSIONS = 384; // Common embedding size (OpenAI ada-002)
    private static final Random random = new Random(42); // Fixed seed for reproducibility

    private static CUcontext context;
    private static CUdevice device;
    private static boolean gpuAvailable;

    private GpuKernelLoader kernelLoader;

    @BeforeAll
    static void setupCuda() {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("JVectorCUDA GPU Benchmark Suite");
        System.out.println("=".repeat(70));

        // Print system specs dynamically
        com.vindex.jvectorcuda.benchmark.BenchmarkRunner.printSystemSpecs();

        // Check if CUDA is available
        gpuAvailable = CudaDetector.isAvailable();
        assumeTrue(gpuAvailable, "Skipping GPU benchmarks - CUDA not available");

        cuInit(0);
        device = new CUdevice();
        cuDeviceGet(device, 0);
        context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // Print VRAM status
        VramUtil.printVramStatus();
        System.out.println();
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

    /**
     * Test 1: Find GPU break-even point across dataset sizes and batch sizes.
     * 
     * <p>
     * This test measures cold-start performance where data is uploaded
     * to GPU for each batch. Results help determine the minimum batch size
     * where GPU becomes beneficial.
     */
    @Test
    @Order(1)
    @DisplayName("Find GPU break-even point (dataset size x batch size)")
    void testBreakEvenPoints() {
        System.out.println("\n=== GPU Break-Even Point Analysis ===\n");
        System.out.println("Testing when GPU starts outperforming CPU (cold-start)...\n");

        // Scale test sizes based on available VRAM
        int maxSafeVectors = VramUtil.getMaxSafeVectorCount(DIMENSIONS);
        int[] baseVectorCounts = { 1_000, 10_000, 50_000, 100_000 };
        int[] vectorCounts = new int[baseVectorCounts.length];

        for (int i = 0; i < baseVectorCounts.length; i++) {
            vectorCounts[i] = Math.min(baseVectorCounts[i], maxSafeVectors);
        }

        System.out.printf("Max safe vectors for your GPU: %,d (at %d dims)%n%n", maxSafeVectors, DIMENSIONS);

        int[] batchSizes = { 1, 10, 100 };

        System.out.printf("%-15s %-12s %-12s %-12s %-10s%n",
                "Vectors", "Batch", "CPU (ms)", "GPU (ms)", "Speedup");
        System.out.println("-".repeat(65));

        for (int numVectors : vectorCounts) {
            for (int batchSize : batchSizes) {
                // Skip very large tests to keep runtime reasonable
                if (numVectors >= 100_000 && batchSize > 10) {
                    continue;
                }

                BenchmarkResult result = runBenchmark(numVectors, DIMENSIONS, batchSize);

                System.out.printf("%-15s %-12d %-12.2f %-12.2f %-10.2fx%n",
                        formatNumber(numVectors),
                        batchSize,
                        result.cpuTimeMs,
                        result.gpuTimeMs,
                        result.speedup);
            }
            System.out.println();
        }

        System.out.println("Note: Cold-start includes memory transfer for each batch.");
        System.out.println("      For persistent memory scenarios, see testPersistentGpuMemory().");
    }

    /**
     * Test 2: Measure memory transfer overhead percentage.
     * 
     * <p>
     * This test isolates the PCIe transfer cost from GPU compute time.
     * High transfer overhead (>50%) indicates that persistent memory
     * patterns should be used for best performance.
     */
    @Test
    @Order(2)
    @DisplayName("Measure memory transfer overhead percentage")
    void testMemoryTransferOverhead() {
        System.out.println("\n=== Memory Transfer Overhead Analysis ===\n");

        // Scale to available VRAM
        int numVectors = VramUtil.scaleToAvailableVram(50_000, DIMENSIONS);
        System.out.printf("Testing with %,d vectors x %d dimensions%n%n", numVectors, DIMENSIONS);

        float[][] database = generateRandomDatabase(numVectors, DIMENSIONS);
        float[] query = generateRandomVector(DIMENSIONS);

        // Measure total GPU time (includes transfer)
        long totalStart = System.nanoTime();
        float[] gpuResult = euclideanDistanceGPU(database, query, numVectors, DIMENSIONS);
        long totalTime = System.nanoTime() - totalStart;
        assertTrue(gpuResult.length > 0, "GPU result should not be empty");

        // Measure transfer time only (upload database + query, download results)
        long transferStart = System.nanoTime();
        CUdeviceptr d_database = uploadToGPU(flattenDatabase(database, numVectors, DIMENSIONS));
        CUdeviceptr d_query = uploadToGPU(query);
        float[] dummy = downloadFromGPU(d_database, numVectors * DIMENSIONS);
        long transferTime = System.nanoTime() - transferStart;
        assertTrue(dummy.length > 0, "Downloaded data should not be empty");
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
            System.out.println("RESULT: Transfer overhead dominates (>" + 50 + "%)");
            System.out.println("        → Use persistent memory pattern for best results");
        } else {
            System.out.println("RESULT: Compute time dominates");
            System.out.println("        → GPU provides good speedup even for single queries");
        }
    }

    /**
     * Test 3: Persistent GPU memory - upload database once, query many times.
     * 
     * <p>
     * This is the recommended usage pattern for JVectorCUDA. By keeping
     * the database in GPU memory and only uploading queries, transfer cost
     * is amortized across many queries.
     * 
     * <p>
     * Expected result: GPU should be significantly faster (2-10x typical).
     */
    @Test
    @Order(3)
    @DisplayName("Test persistent GPU memory (amortize transfer cost)")
    void testPersistentGpuMemory() {
        System.out.println("\n=== Persistent GPU Memory Test ===\n");
        System.out.println("Scenario: Upload database once, run many queries\n");

        // Scale to available VRAM
        int numVectors = VramUtil.scaleToAvailableVram(50_000, DIMENSIONS);
        int numQueries = 100;

        System.out.printf("Dataset: %,d vectors x %d dimensions%n", numVectors, DIMENSIONS);
        System.out.printf("Queries: %d%n%n", numQueries);

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
                    Pointer.to(new int[] { numVectors }),
                    Pointer.to(new int[] { DIMENSIONS }));
            kernelLoader.launch(gridSize, blockSize, params);

            cuMemFree(d_query);
        }

        cuMemFree(d_database);
        cuMemFree(d_distances);

        long gpuTime = System.nanoTime() - gpuStart;
        float speedup = (float) cpuTime / gpuTime;

        System.out.printf("CPU: %d queries x %,d vectors = %.2f ms (%.2f ms/query)%n",
                numQueries, numVectors, cpuTime / 1e6, (cpuTime / 1e6) / numQueries);
        System.out.printf("GPU: %d queries x %,d vectors = %.2f ms (%.2f ms/query)%n",
                numQueries, numVectors, gpuTime / 1e6, (gpuTime / 1e6) / numQueries);
        System.out.printf("Speedup: %.2fx%n", speedup);
        System.out.println();

        if (speedup > 1.0f) {
            System.out.println("RESULT: GPU is faster with persistent memory!");
            System.out.printf("        Per-query latency: %.2f ms%n", (gpuTime / 1e6) / numQueries);
        } else {
            System.out.println("RESULT: GPU is not faster on this hardware configuration");
            System.out.println("        Consider using CPU-only mode for this workload");
        }
    }

    /**
     * Test 4: Different vector dimensions impact on GPU performance.
     * 
     * <p>
     * Higher dimensions mean more compute per vector, which generally
     * improves GPU utilization. This test helps identify the dimension
     * threshold where GPU becomes beneficial.
     */
    @Test
    @Order(4)
    @DisplayName("Test different vector dimensions (128, 384, 768, 1536)")
    void testDifferentDimensions() {
        System.out.println("\n=== Dimension Impact on GPU Performance ===\n");

        // Dynamically scale vector counts based on available VRAM
        int[][] dimAndVectors = {
                { 128, VramUtil.scaleToAvailableVram(50_000, 128) },
                { 384, VramUtil.scaleToAvailableVram(50_000, 384) },
                { 768, VramUtil.scaleToAvailableVram(25_000, 768) },
                { 1536, VramUtil.scaleToAvailableVram(10_000, 1536) }
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

        System.out.println("\nNote: Higher dimensions = more compute per vector = better GPU utilization");
        System.out.println("      Vector counts scaled automatically to fit your GPU memory.");
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
                Pointer.to(new int[] { numVectors }),
                Pointer.to(new int[] { dimensions }));
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
        cuMemcpyDtoH(Pointer.to(result), devicePtr, (long) length * Sizeof.FLOAT);
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
