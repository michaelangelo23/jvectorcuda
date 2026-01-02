package com.vindex.jvectorcuda.benchmarks;

import com.vindex.jvectorcuda.gpu.GpuKernelLoader;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;
import org.junit.jupiter.api.*;

import java.util.Random;

import static jcuda.driver.JCudaDriver.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * GPU Break-Even Point Benchmark Suite
 * 
 * Purpose: Find when GPU acceleration becomes beneficial vs CPU
 * 
 * Tests different scenarios:
 * - Vector counts: 1K, 10K, 100K, 1M
 * - Batch sizes: 1, 10, 100, 1000 queries
 * - Memory transfer overhead analysis
 * - Persistent GPU memory (data stays on GPU)
 * 
 * Results guide adaptive routing strategy for hybrid CPU/GPU approach.
 * 
 * @author JVectorCUDA
 */
@DisplayName("GPU Break-Even Point Benchmarks")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class GpuBreakEvenTest {
    
    private static final int DIMENSIONS = 384;  // Common embedding size
    private static final Random random = new Random(42);
    
    private GpuKernelLoader kernelLoader;
    
    @BeforeEach
    void setUp() {
        kernelLoader = new GpuKernelLoader("euclidean_distance.ptx", "euclideanDistance");
    }
    
    @AfterEach
    void tearDown() {
        if (kernelLoader != null) {
            kernelLoader.cleanup();
        }
    }
    
    /**
     * Test 1: Find break-even point across different dataset sizes and batch sizes
     */
    @Test
    @Order(1)
    @DisplayName("Find GPU break-even point (dataset size × batch size)")
    void testBreakEvenPoints() {
        System.out.println("\n=== GPU Break-Even Point Analysis ===\n");
        System.out.println("Testing when GPU starts outperforming CPU...\n");
        
        int[] vectorCounts = {1_000, 10_000, 100_000, 1_000_000};
        int[] batchSizes = {1, 10, 100, 1000};
        
        System.out.printf("%-15s %-12s %-12s %-12s %-12s %-15s%n",
            "Vectors", "Batch", "CPU (ms)", "GPU (ms)", "Speedup", "Winner");
        System.out.println("-".repeat(80));
        
        for (int numVectors : vectorCounts) {
            for (int batchSize : batchSizes) {
                // Skip very large tests
                if (numVectors == 1_000_000 && batchSize > 100) {
                    continue;
                }
                
                BenchmarkResult result = runBenchmark(numVectors, batchSize);
                
                String winner = result.speedup >= 1.0f ? "GPU" : "CPU";
                System.out.printf("%-15s %-12d %-12.2f %-12.2f %-12.2fx %-15s%n",
                    formatNumber(numVectors),
                    batchSize,
                    result.cpuTimeMs,
                    result.gpuTimeMs,
                    result.speedup,
                    winner);
            }
            System.out.println();
        }
        
        System.out.println("\nConclusion: GPU wins when speedup > 1.0x");
        System.out.println("Use this data to configure adaptive routing thresholds.\n");
    }
    
    /**
     * Test 2: Analyze memory transfer overhead
     */
    @Test
    @Order(2)
    @DisplayName("Measure memory transfer overhead percentage")
    void testMemoryTransferOverhead() {
        System.out.println("\n=== Memory Transfer Overhead Analysis ===\n");
        
        int numVectors = 50_000;
        int batchSize = 10;
        
        float[][] database = generateRandomDatabase(numVectors, DIMENSIONS);
        float[] query = generateRandomVector(DIMENSIONS);
        
        // Measure total GPU time
        long totalStart = System.nanoTime();
        float[] gpuResult = euclideanDistanceGPU(database, query, numVectors, DIMENSIONS);
        long totalTime = System.nanoTime() - totalStart;
        
        // Measure transfer time (simplified - upload + download)
        long transferStart = System.nanoTime();
        CUdeviceptr d_database = uploadToGPU(database);
        CUdeviceptr d_query = uploadToGPU(query);
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
            System.out.println("WARNING: Memory transfer dominates (>50%)");
            System.out.println("         GPU may not be beneficial for this workload");
        } else {
            System.out.println("GOOD: Compute time dominates - GPU can provide speedup");
        }
        
        System.out.println();
    }
    
    /**
     * Test 3: Persistent GPU memory (keep data on GPU, query many times)
     */
    @Test
    @Order(3)
    @DisplayName("Test persistent GPU memory (amortize transfer cost)")
    void testPersistentGpuMemory() {
        System.out.println("\n=== Persistent GPU Memory Test ===\n");
        System.out.println("Scenario: Upload database once, run many queries\n");
        
        int numVectors = 100_000;
        int numQueries = 1000;
        
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
        CUdeviceptr d_database = uploadToGPU(database);
        CUdeviceptr d_distances = new CUdeviceptr();
        cuMemAlloc(d_distances, numVectors * Sizeof.FLOAT);
        
        // Query many times (no upload/download per query)
        for (int i = 0; i < numQueries; i++) {
            CUdeviceptr d_query = uploadToGPU(queries[i]);
            
            // Launch kernel
            Pointer params = Pointer.to(
                Pointer.to(d_database),
                Pointer.to(d_query),
                Pointer.to(d_distances),
                Pointer.to(new int[]{numVectors}),
                Pointer.to(new int[]{DIMENSIONS})
            );
            
            int blockSize = 256;
            int gridSize = (numVectors + blockSize - 1) / blockSize;
            kernelLoader.launch(gridSize, blockSize, params);
            
            cuMemFree(d_query);
        }
        
        cuMemFree(d_database);
        cuMemFree(d_distances);
        
        long gpuTime = System.nanoTime() - gpuStart;
        
        float speedup = (float) cpuTime / gpuTime;
        
        System.out.printf("CPU: %d queries × %,d vectors = %.2f ms%n",
            numQueries, numVectors, cpuTime / 1e6);
        System.out.printf("GPU: %d queries × %,d vectors = %.2f ms%n",
            numQueries, numVectors, gpuTime / 1e6);
        System.out.printf("Speedup: %.2fx%n", speedup);
        System.out.println();
        
        if (speedup > 1.0f) {
            System.out.println("SUCCESS: GPU wins with persistent memory!");
            System.out.printf("Per-query latency: %.2fms%n", (gpuTime / 1e6) / numQueries);
        } else {
            System.out.println("NOTE: Even with persistent memory, CPU is competitive");
        }
        
        System.out.println();
    }
    
    /**
     * Test 4: Different vector dimensions
     */
    @Test
    @Order(4)
    @DisplayName("Test different vector dimensions (128, 384, 768, 1536)")
    void testDifferentDimensions() {
        System.out.println("\n=== Dimension Impact on GPU Performance ===\n");
        
        int[] dimensions = {128, 384, 768, 1536};
        int numVectors = 50_000;
        
        System.out.printf("%-15s %-12s %-12s %-12s%n",
            "Dimensions", "CPU (ms)", "GPU (ms)", "Speedup");
        System.out.println("-".repeat(55));
        
        for (int dim : dimensions) {
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
            
            System.out.printf("%-15d %-12.2f %-12.2f %-12.2fx%n",
                dim, cpuTime / 1e6, gpuTime / 1e6, speedup);
        }
        
        System.out.println("\nHigher dimensions = more compute work per vector");
        System.out.println("May improve GPU speedup if compute > transfer overhead\n");
    }
    
    // ========== Helper Methods ==========
    
    private BenchmarkResult runBenchmark(int numVectors, int batchSize) {
        float[][] database = generateRandomDatabase(numVectors, DIMENSIONS);
        float[][] queries = new float[batchSize][];
        for (int i = 0; i < batchSize; i++) {
            queries[i] = generateRandomVector(DIMENSIONS);
        }
        
        // CPU
        long cpuStart = System.nanoTime();
        for (float[] query : queries) {
            euclideanDistanceCPU(database, query);
        }
        long cpuTime = System.nanoTime() - cpuStart;
        
        // GPU
        long gpuStart = System.nanoTime();
        for (float[] query : queries) {
            euclideanDistanceGPU(database, query, numVectors, DIMENSIONS);
        }
        long gpuTime = System.nanoTime() - gpuStart;
        
        return new BenchmarkResult(cpuTime / 1e6, gpuTime / 1e6);
    }
    
    private float[] euclideanDistanceGPU(float[][] database, float[] query, int numVectors, int dimensions) {
        CUdeviceptr d_database = uploadToGPU(database);
        CUdeviceptr d_query = uploadToGPU(query);
        CUdeviceptr d_distances = new CUdeviceptr();
        cuMemAlloc(d_distances, numVectors * Sizeof.FLOAT);
        
        Pointer params = Pointer.to(
            Pointer.to(d_database),
            Pointer.to(d_query),
            Pointer.to(d_distances),
            Pointer.to(new int[]{numVectors}),
            Pointer.to(new int[]{dimensions})
        );
        
        int blockSize = 256;
        int gridSize = (numVectors + blockSize - 1) / blockSize;
        kernelLoader.launch(gridSize, blockSize, params);
        
        float[] distances = downloadFromGPU(d_distances, numVectors);
        
        cuMemFree(d_database);
        cuMemFree(d_query);
        cuMemFree(d_distances);
        
        return distances;
    }
    
    private float[] euclideanDistanceCPU(float[][] database, float[] query) {
        int numVectors = database.length;
        int dimensions = query.length;
        float[] distances = new float[numVectors];
        
        for (int i = 0; i < numVectors; i++) {
            float sum = 0.0f;
            for (int d = 0; d < dimensions; d++) {
                float diff = database[i][d] - query[d];
                sum += diff * diff;
            }
            distances[i] = (float) Math.sqrt(sum);
        }
        
        return distances;
    }
    
    private CUdeviceptr uploadToGPU(float[][] data) {
        int rows = data.length;
        int cols = data[0].length;
        float[] flat = new float[rows * cols];
        
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, flat, i * cols, cols);
        }
        
        return uploadToGPU(flat);
    }
    
    private CUdeviceptr uploadToGPU(float[] data) {
        CUdeviceptr devicePtr = new CUdeviceptr();
        cuMemAlloc(devicePtr, data.length * Sizeof.FLOAT);
        cuMemcpyHtoD(devicePtr, Pointer.to(data), data.length * Sizeof.FLOAT);
        return devicePtr;
    }
    
    private float[] downloadFromGPU(CUdeviceptr devicePtr, int length) {
        float[] result = new float[length];
        cuMemcpyDtoH(Pointer.to(result), devicePtr, length * Sizeof.FLOAT);
        return result;
    }
    
    private float[][] generateRandomDatabase(int numVectors, int dimensions) {
        float[][] database = new float[numVectors][dimensions];
        for (int i = 0; i < numVectors; i++) {
            for (int d = 0; d < dimensions; d++) {
                database[i][d] = random.nextFloat() * 2.0f - 1.0f;
            }
        }
        return database;
    }
    
    private float[] generateRandomVector(int dimensions) {
        float[] vector = new float[dimensions];
        for (int d = 0; d < dimensions; d++) {
            vector[d] = random.nextFloat() * 2.0f - 1.0f;
        }
        return vector;
    }
    
    private String formatNumber(int number) {
        if (number >= 1_000_000) {
            return String.format("%dM", number / 1_000_000);
        } else if (number >= 1_000) {
            return String.format("%dK", number / 1_000);
        }
        return String.valueOf(number);
    }
    
    static class BenchmarkResult {
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
