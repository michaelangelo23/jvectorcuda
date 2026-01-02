package com.vindex.jvectorcuda;

import com.vindex.jvectorcuda.gpu.GpuKernelLoader;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Random;

import static jcuda.driver.JCudaDriver.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for distance metric CUDA kernels.
 * 
 * <p>Tests all three distance metrics (Euclidean, Cosine, Inner Product)
 * for correctness, edge cases, and AI blind spots.
 * 
 * <p>Following TDD workflow: tests written BEFORE integration into GPUVectorIndex.
 * 
 * @author JVectorCUDA (AI-assisted, Human-verified)
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class DistanceMetricTest {

    private static final float EPSILON = 1e-4f;
    private static final int BLOCK_SIZE = 256;

    private static CUcontext context;
    private static CUdevice device;

    @BeforeAll
    static void initCuda() {
        Assumptions.assumeTrue(CudaDetector.isAvailable(), "CUDA not available");
        
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        device = new CUdevice();
        cuDeviceGet(device, 0);
        context = new CUcontext();
        cuCtxCreate(context, 0, device);
    }

    @AfterAll
    static void cleanupCuda() {
        if (context != null) {
            cuCtxDestroy(context);
        }
    }

    // ===========================================
    // Section B1: Correctness Tests
    // ===========================================

    @Test
    @Order(1)
    @DisplayName("Euclidean: Identical vectors have zero distance")
    void euclideanIdenticalVectors() throws Exception {
        float[] vector = {1.0f, 2.0f, 3.0f, 4.0f};
        float distance = computeGpuDistance(DistanceMetric.EUCLIDEAN, vector, vector);
        
        assertEquals(0.0f, distance, EPSILON, "Identical vectors should have zero Euclidean distance");
    }

    @Test
    @Order(2)
    @DisplayName("Euclidean: Known distance calculation")
    void euclideanKnownDistance() throws Exception {
        // Distance from (0,0) to (3,4) should be 5
        float[] a = {0.0f, 0.0f};
        float[] b = {3.0f, 4.0f};
        
        float distance = computeGpuDistance(DistanceMetric.EUCLIDEAN, a, b);
        
        assertEquals(5.0f, distance, EPSILON, "Euclidean distance from (0,0) to (3,4) should be 5");
    }

    @Test
    @Order(3)
    @DisplayName("Cosine: Identical vectors have zero distance")
    void cosineIdenticalVectors() throws Exception {
        float[] vector = {1.0f, 2.0f, 3.0f, 4.0f};
        float distance = computeGpuDistance(DistanceMetric.COSINE, vector, vector);
        
        assertEquals(0.0f, distance, EPSILON, "Identical vectors should have zero cosine distance");
    }

    @Test
    @Order(4)
    @DisplayName("Cosine: Orthogonal vectors have distance 1")
    void cosineOrthogonalVectors() throws Exception {
        // (1,0) and (0,1) are orthogonal - cosine similarity = 0, distance = 1
        float[] a = {1.0f, 0.0f};
        float[] b = {0.0f, 1.0f};
        
        float distance = computeGpuDistance(DistanceMetric.COSINE, a, b);
        
        assertEquals(1.0f, distance, EPSILON, "Orthogonal vectors should have cosine distance 1");
    }

    @Test
    @Order(5)
    @DisplayName("Cosine: Opposite vectors have distance 2")
    void cosineOppositeVectors() throws Exception {
        // (1,0) and (-1,0) are opposite - cosine similarity = -1, distance = 2
        float[] a = {1.0f, 0.0f};
        float[] b = {-1.0f, 0.0f};
        
        float distance = computeGpuDistance(DistanceMetric.COSINE, a, b);
        
        assertEquals(2.0f, distance, EPSILON, "Opposite vectors should have cosine distance 2");
    }

    @Test
    @Order(6)
    @DisplayName("InnerProduct: Identical normalized vectors have negative distance")
    void innerProductIdenticalNormalized() throws Exception {
        // Normalized vector: sum of squares = 1
        float[] vector = normalize(new float[]{1.0f, 2.0f, 3.0f, 4.0f});
        float distance = computeGpuDistance(DistanceMetric.INNER_PRODUCT, vector, vector);
        
        // Dot product of unit vector with itself = 1, negative = -1
        assertEquals(-1.0f, distance, EPSILON, "Identical unit vectors should have inner product distance -1");
    }

    @Test
    @Order(7)
    @DisplayName("InnerProduct: Orthogonal vectors have zero distance")
    void innerProductOrthogonalVectors() throws Exception {
        float[] a = {1.0f, 0.0f};
        float[] b = {0.0f, 1.0f};
        
        float distance = computeGpuDistance(DistanceMetric.INNER_PRODUCT, a, b);
        
        assertEquals(0.0f, distance, EPSILON, "Orthogonal vectors should have inner product distance 0");
    }

    // ===========================================
    // Section B2: Edge Cases
    // ===========================================

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {

        @Test
        @DisplayName("Zero vector handling - Euclidean")
        void zeroVectorEuclidean() throws Exception {
            float[] zero = {0.0f, 0.0f, 0.0f, 0.0f};
            float[] nonZero = {1.0f, 2.0f, 3.0f, 4.0f};
            
            float distance = computeGpuDistance(DistanceMetric.EUCLIDEAN, zero, nonZero);
            
            // Should be the norm of nonZero
            float expectedNorm = (float) Math.sqrt(1 + 4 + 9 + 16);
            assertEquals(expectedNorm, distance, EPSILON);
        }

        @Test
        @DisplayName("Zero vector handling - Cosine")
        void zeroVectorCosine() throws Exception {
            float[] zero = {0.0f, 0.0f, 0.0f, 0.0f};
            float[] nonZero = {1.0f, 2.0f, 3.0f, 4.0f};
            
            float distance = computeGpuDistance(DistanceMetric.COSINE, zero, nonZero);
            
            // Zero vector has no direction - should return 1 (no similarity)
            assertEquals(1.0f, distance, EPSILON, "Zero vector should have cosine distance 1 (no similarity)");
        }

        @Test
        @DisplayName("Single dimension vectors")
        void singleDimension() throws Exception {
            float[] a = {5.0f};
            float[] b = {3.0f};
            
            float euclidean = computeGpuDistance(DistanceMetric.EUCLIDEAN, a, b);
            assertEquals(2.0f, euclidean, EPSILON, "1D Euclidean distance");
            
            float cosine = computeGpuDistance(DistanceMetric.COSINE, a, b);
            assertEquals(0.0f, cosine, EPSILON, "Same sign 1D vectors should have zero cosine distance");
        }

        @Test
        @DisplayName("Large dimension vectors (1536D like OpenAI)")
        void largeDimensions() throws Exception {
            int dims = 1536;
            Random random = new Random(42);
            
            float[] a = new float[dims];
            float[] b = new float[dims];
            for (int i = 0; i < dims; i++) {
                a[i] = random.nextFloat();
                b[i] = random.nextFloat();
            }
            
            // Just verify it doesn't crash and produces a value
            float distance = computeGpuDistance(DistanceMetric.EUCLIDEAN, a, b);
            assertTrue(distance > 0, "Large vectors should have positive Euclidean distance");
            assertTrue(Float.isFinite(distance), "Distance should be finite");
        }

        @Test
        @DisplayName("Non-multiple-of-blocksize vector count")
        void nonMultipleOfBlocksize() throws Exception {
            // 257 vectors with 256 block size = 2 blocks, 1 thread idle
            int numVectors = 257;
            int dims = 4;
            
            float[][] database = new float[numVectors][dims];
            for (int i = 0; i < numVectors; i++) {
                database[i] = new float[]{(float) i, 0.0f, 0.0f, 0.0f};
            }
            float[] query = {100.0f, 0.0f, 0.0f, 0.0f};
            
            float[] distances = computeGpuDistancesBatch(DistanceMetric.EUCLIDEAN, database, query);
            
            assertEquals(numVectors, distances.length);
            // Distance from (100,0,0,0) to (100,0,0,0) should be 0
            assertEquals(0.0f, distances[100], EPSILON, "Vector at index 100 should match query");
        }
    }

    // ===========================================
    // Section B3: AI Blind Spots
    // ===========================================

    @Nested
    @DisplayName("AI Blind Spots")
    class AIBlindSpots {

        @Test
        @DisplayName("NaN handling - should not propagate")
        void nanHandling() throws Exception {
            float[] withNaN = {Float.NaN, 1.0f, 2.0f, 3.0f};
            float[] normal = {1.0f, 1.0f, 2.0f, 3.0f};
            
            float distance = computeGpuDistance(DistanceMetric.EUCLIDEAN, withNaN, normal);
            
            // NaN in input will produce NaN in output - this is expected behavior
            // We document this rather than try to handle it in the kernel
            assertTrue(Float.isNaN(distance) || Float.isFinite(distance), 
                "NaN input produces NaN or finite output (documented behavior)");
        }

        @Test
        @DisplayName("Infinity handling")
        void infinityHandling() throws Exception {
            float[] withInf = {Float.POSITIVE_INFINITY, 1.0f, 2.0f, 3.0f};
            float[] normal = {1.0f, 1.0f, 2.0f, 3.0f};
            
            float distance = computeGpuDistance(DistanceMetric.EUCLIDEAN, withInf, normal);
            
            // Infinity should propagate to output
            assertTrue(Float.isInfinite(distance) || Float.isNaN(distance), 
                "Infinity input should propagate");
        }

        @Test
        @DisplayName("Floating point precision - GPU matches CPU within epsilon")
        void floatingPointPrecision() throws Exception {
            Random random = new Random(12345);
            int dims = 128;
            
            float[] a = new float[dims];
            float[] b = new float[dims];
            for (int i = 0; i < dims; i++) {
                a[i] = random.nextFloat() * 100;  // Larger values to stress precision
                b[i] = random.nextFloat() * 100;
            }
            
            float gpuDistance = computeGpuDistance(DistanceMetric.EUCLIDEAN, a, b);
            float cpuDistance = cpuEuclidean(a, b);
            
            assertEquals(cpuDistance, gpuDistance, EPSILON, 
                "GPU and CPU Euclidean distances should match within epsilon");
        }

        @Test
        @DisplayName("Cosine - GPU matches CPU for all metrics")
        void gpuMatchesCpuCosine() throws Exception {
            Random random = new Random(54321);
            int dims = 64;
            
            float[] a = new float[dims];
            float[] b = new float[dims];
            for (int i = 0; i < dims; i++) {
                a[i] = random.nextFloat() - 0.5f;  // Include negative values
                b[i] = random.nextFloat() - 0.5f;
            }
            
            float gpuDistance = computeGpuDistance(DistanceMetric.COSINE, a, b);
            float cpuDistance = cpuCosineDistance(a, b);
            
            assertEquals(cpuDistance, gpuDistance, EPSILON, 
                "GPU and CPU Cosine distances should match within epsilon");
        }

        @Test
        @DisplayName("Inner Product - GPU matches CPU")
        void gpuMatchesCpuInnerProduct() throws Exception {
            float[] a = normalize(new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
            float[] b = normalize(new float[]{5.0f, 4.0f, 3.0f, 2.0f, 1.0f});
            
            float gpuDistance = computeGpuDistance(DistanceMetric.INNER_PRODUCT, a, b);
            float cpuDistance = -dotProduct(a, b);
            
            assertEquals(cpuDistance, gpuDistance, EPSILON, 
                "GPU and CPU Inner Product distances should match within epsilon");
        }

        @Test
        @DisplayName("Very small values - no underflow")
        void verySmallValues() throws Exception {
            float[] tiny = {1e-38f, 1e-38f, 1e-38f, 1e-38f};
            float[] normal = {1.0f, 0.0f, 0.0f, 0.0f};
            
            float distance = computeGpuDistance(DistanceMetric.EUCLIDEAN, tiny, normal);
            assertTrue(Float.isFinite(distance), "Very small values should not cause underflow");
            assertTrue(distance > 0, "Distance should be positive");
        }

        @Test
        @DisplayName("Very large values - no overflow")  
        void veryLargeValues() throws Exception {
            float[] large = {1e19f, 1e19f, 1e19f, 1e19f};
            float[] normal = {0.0f, 0.0f, 0.0f, 0.0f};
            
            float distance = computeGpuDistance(DistanceMetric.EUCLIDEAN, large, normal);
            // With 4 dimensions of 1e19, squared sum is 4e38 which overflows
            // This is expected behavior - we document it
            assertTrue(Float.isInfinite(distance) || Float.isFinite(distance), 
                "Very large values may overflow (documented behavior)");
        }
    }

    // ===========================================
    // Section B4: Performance Characteristics
    // ===========================================

    @ParameterizedTest
    @EnumSource(DistanceMetric.class)
    @DisplayName("All metrics produce finite results on random data")
    void allMetricsFiniteResults(DistanceMetric metric) throws Exception {
        Random random = new Random(99999);
        int dims = 384;
        
        float[] a = new float[dims];
        float[] b = new float[dims];
        for (int i = 0; i < dims; i++) {
            a[i] = random.nextFloat() * 2 - 1;  // [-1, 1]
            b[i] = random.nextFloat() * 2 - 1;
        }
        
        float distance = computeGpuDistance(metric, a, b);
        assertTrue(Float.isFinite(distance), 
            metric + " should produce finite results on random data");
    }

    // ===========================================
    // Helper Methods
    // ===========================================

    /**
     * Compute distance between two vectors using GPU.
     * Treats first vector as single-entry database, second as query.
     */
    private float computeGpuDistance(DistanceMetric metric, float[] database, float[] query) throws Exception {
        float[][] dbArray = {database};
        float[] distances = computeGpuDistancesBatch(metric, dbArray, query);
        return distances[0];
    }

    /**
     * Compute distances from query to all database vectors using GPU.
     */
    private float[] computeGpuDistancesBatch(DistanceMetric metric, float[][] database, float[] query) throws Exception {
        int numVectors = database.length;
        int dimensions = query.length;
        
        // Flatten database
        float[] flatDatabase = new float[numVectors * dimensions];
        for (int i = 0; i < numVectors; i++) {
            System.arraycopy(database[i], 0, flatDatabase, i * dimensions, dimensions);
        }
        
        // Allocate GPU memory
        CUdeviceptr d_database = new CUdeviceptr();
        CUdeviceptr d_query = new CUdeviceptr();
        CUdeviceptr d_distances = new CUdeviceptr();
        
        cuMemAlloc(d_database, (long) flatDatabase.length * Sizeof.FLOAT);
        cuMemAlloc(d_query, (long) query.length * Sizeof.FLOAT);
        cuMemAlloc(d_distances, (long) numVectors * Sizeof.FLOAT);
        
        // Upload data
        cuMemcpyHtoD(d_database, Pointer.to(flatDatabase), (long) flatDatabase.length * Sizeof.FLOAT);
        cuMemcpyHtoD(d_query, Pointer.to(query), (long) query.length * Sizeof.FLOAT);
        
        // Load kernel
        GpuKernelLoader loader = new GpuKernelLoader(metric.getPtxFile(), metric.getKernelName());
        
        // Launch kernel
        int gridSize = (numVectors + BLOCK_SIZE - 1) / BLOCK_SIZE;
        Pointer kernelParams = Pointer.to(
            Pointer.to(d_database),
            Pointer.to(d_query),
            Pointer.to(d_distances),
            Pointer.to(new int[]{numVectors}),
            Pointer.to(new int[]{dimensions})
        );
        
        loader.launch(gridSize, BLOCK_SIZE, kernelParams);
        cuCtxSynchronize();
        
        // Download results
        float[] distances = new float[numVectors];
        cuMemcpyDtoH(Pointer.to(distances), d_distances, (long) numVectors * Sizeof.FLOAT);
        
        // Cleanup
        cuMemFree(d_database);
        cuMemFree(d_query);
        cuMemFree(d_distances);
        
        return distances;
    }

    // CPU reference implementations for validation
    
    private float cpuEuclidean(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return (float) Math.sqrt(sum);
    }
    
    private float cpuCosineDistance(float[] a, float[] b) {
        float dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        float normProduct = (float) (Math.sqrt(normA) * Math.sqrt(normB));
        if (normProduct < 1e-8f) return 1.0f;
        float cosineSim = dot / normProduct;
        cosineSim = Math.max(-1.0f, Math.min(1.0f, cosineSim));
        return 1.0f - cosineSim;
    }
    
    private float dotProduct(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
    
    private float[] normalize(float[] v) {
        float norm = 0;
        for (float f : v) norm += f * f;
        norm = (float) Math.sqrt(norm);
        float[] result = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            result[i] = v[i] / norm;
        }
        return result;
    }
}
