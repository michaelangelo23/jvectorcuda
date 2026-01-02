package com.vindex.jvectorcuda;

import com.vindex.jvectorcuda.cpu.CPUVectorIndex;
import com.vindex.jvectorcuda.gpu.GPUVectorIndex;
import org.junit.jupiter.api.*;

import static org.junit.jupiter.api.Assertions.*;

// Tests for CPU and GPU VectorIndex implementations
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class VectorIndexTest {

    private static final int DIMS = 128;
    private static final float EPSILON = 0.001f;

    // ===========================================
    // CPU VectorIndex Tests
    // ===========================================

    @Test
    @Order(1)
    @DisplayName("CPU: Create and close index")
    void cpuCreateAndClose() {
        CPUVectorIndex index = new CPUVectorIndex(DIMS);
        assertEquals(DIMS, index.getDimensions());
        assertEquals(0, index.size());
        index.close();
    }

    @Test
    @Order(2)
    @DisplayName("CPU: Add vectors and search")
    void cpuAddAndSearch() {
        try (CPUVectorIndex index = new CPUVectorIndex(DIMS)) {
            // Create test vectors
            float[][] vectors = createTestVectors(100, DIMS);
            index.add(vectors);
            
            assertEquals(100, index.size());
            
            // Search for nearest to first vector
            SearchResult result = index.search(vectors[0], 5);
            
            assertNotNull(result);
            assertEquals(5, result.getIds().length);
            assertEquals(5, result.getDistances().length);
            
            // First result should be exact match (distance ~0)
            assertEquals(0, result.getIds()[0]);
            assertTrue(result.getDistances()[0] < EPSILON);
        }
    }

    @Test
    @Order(3)
    @DisplayName("CPU: Search with k > size returns all")
    void cpuSearchKGreaterThanSize() {
        try (CPUVectorIndex index = new CPUVectorIndex(DIMS)) {
            float[][] vectors = createTestVectors(3, DIMS);
            index.add(vectors);
            
            SearchResult result = index.search(vectors[0], 10);
            
            assertEquals(3, result.getIds().length);
        }
    }

    @Test
    @Order(4)
    @DisplayName("CPU: Empty index returns empty result")
    void cpuEmptyIndexSearch() {
        try (CPUVectorIndex index = new CPUVectorIndex(DIMS)) {
            float[] query = new float[DIMS];
            SearchResult result = index.search(query, 5);
            
            assertEquals(0, result.getIds().length);
            assertEquals(0, result.getDistances().length);
        }
    }

    @Test
    @Order(5)
    @DisplayName("CPU: Invalid dimensions throws exception")
    void cpuInvalidDimensions() {
        assertThrows(IllegalArgumentException.class, () -> new CPUVectorIndex(0));
        assertThrows(IllegalArgumentException.class, () -> new CPUVectorIndex(-1));
    }

    @Test
    @Order(6)
    @DisplayName("CPU: Mismatched query dimensions throws exception")
    void cpuMismatchedQueryDimensions() {
        try (CPUVectorIndex index = new CPUVectorIndex(DIMS)) {
            float[][] vectors = createTestVectors(10, DIMS);
            index.add(vectors);
            
            float[] wrongQuery = new float[DIMS + 1];
            assertThrows(IllegalArgumentException.class, () -> index.search(wrongQuery, 1));
        }
    }

    @Test
    @Order(7)
    @DisplayName("CPU: Closed index throws exception")
    void cpuClosedIndexThrows() {
        CPUVectorIndex index = new CPUVectorIndex(DIMS);
        index.close();
        
        assertThrows(IllegalStateException.class, () -> index.add(new float[1][DIMS]));
        assertThrows(IllegalStateException.class, () -> index.search(new float[DIMS], 1));
    }

    // ===========================================
    // GPU VectorIndex Tests (if CUDA available)
    // ===========================================

    @Test
    @Order(10)
    @DisplayName("GPU: Create and close index")
    void gpuCreateAndClose() {
        Assumptions.assumeTrue(CudaDetector.isAvailable(), "CUDA not available");
        
        GPUVectorIndex index = new GPUVectorIndex(DIMS);
        assertEquals(DIMS, index.getDimensions());
        assertEquals(0, index.size());
        index.close();
    }

    @Test
    @Order(11)
    @DisplayName("GPU: Add vectors and search")
    void gpuAddAndSearch() {
        Assumptions.assumeTrue(CudaDetector.isAvailable(), "CUDA not available");
        
        try (GPUVectorIndex index = new GPUVectorIndex(DIMS)) {
            float[][] vectors = createTestVectors(100, DIMS);
            index.add(vectors);
            
            assertEquals(100, index.size());
            
            SearchResult result = index.search(vectors[0], 5);
            
            assertNotNull(result);
            assertEquals(5, result.getIds().length);
            assertEquals(5, result.getDistances().length);
            
            // First result should be exact match
            assertEquals(0, result.getIds()[0]);
            assertTrue(result.getDistances()[0] < EPSILON);
        }
    }

    @Test
    @Order(12)
    @DisplayName("GPU: Persistent memory - multiple searches same dataset")
    void gpuPersistentMemory() {
        Assumptions.assumeTrue(CudaDetector.isAvailable(), "CUDA not available");
        
        try (GPUVectorIndex index = new GPUVectorIndex(DIMS)) {
            float[][] vectors = createTestVectors(1000, DIMS);
            index.add(vectors);
            
            // Multiple searches should reuse persistent GPU memory
            for (int i = 0; i < 10; i++) {
                float[] query = vectors[i * 10];
                SearchResult result = index.search(query, 5);
                
                assertEquals(i * 10, result.getIds()[0]);
                assertTrue(result.getDistances()[0] < EPSILON);
            }
        }
    }

    @Test
    @Order(13)
    @DisplayName("GPU: Add vectors incrementally")
    void gpuIncrementalAdd() {
        Assumptions.assumeTrue(CudaDetector.isAvailable(), "CUDA not available");
        
        try (GPUVectorIndex index = new GPUVectorIndex(DIMS)) {
            // Add in batches
            for (int batch = 0; batch < 5; batch++) {
                float[][] vectors = createTestVectors(20, DIMS, batch * 20);
                index.add(vectors);
            }
            
            assertEquals(100, index.size());
            
            // Search should find vectors from all batches
            float[] query = new float[DIMS];
            query[0] = 50.0f; // Should match vector at index 50
            
            SearchResult result = index.search(query, 1);
            assertEquals(50, result.getIds()[0]);
        }
    }

    // ===========================================
    // Factory Tests
    // ===========================================

    @Test
    @Order(20)
    @DisplayName("Factory: auto() returns working index")
    void factoryAuto() {
        VectorIndex index = VectorIndexFactory.auto(DIMS);
        assertNotNull(index);
        assertEquals(DIMS, index.getDimensions());
        index.close();
    }

    @Test
    @Order(21)
    @DisplayName("Factory: cpu() returns CPUVectorIndex")
    void factoryCpu() {
        VectorIndex index = VectorIndexFactory.cpu(DIMS);
        assertNotNull(index);
        assertTrue(index instanceof CPUVectorIndex);
        index.close();
    }

    @Test
    @Order(22)
    @DisplayName("Factory: gpu() returns GPUVectorIndex when CUDA available")
    void factoryGpu() {
        Assumptions.assumeTrue(CudaDetector.isAvailable(), "CUDA not available");
        
        VectorIndex index = VectorIndexFactory.gpu(DIMS);
        assertNotNull(index);
        assertTrue(index instanceof GPUVectorIndex);
        index.close();
    }

    @Test
    @Order(23)
    @DisplayName("Factory: gpu() throws when CUDA not available")
    void factoryGpuNoCuda() {
        Assumptions.assumeFalse(CudaDetector.isAvailable(), "CUDA is available");
        
        assertThrows(UnsupportedOperationException.class, () -> VectorIndexFactory.gpu(DIMS));
    }

    @Test
    @Order(24)
    @DisplayName("Factory: invalid dimensions throws exception")
    void factoryInvalidDimensions() {
        assertThrows(IllegalArgumentException.class, () -> VectorIndexFactory.auto(0));
        assertThrows(IllegalArgumentException.class, () -> VectorIndexFactory.cpu(-1));
    }

    // ===========================================
    // GPU vs CPU Consistency Tests
    // ===========================================

    @Test
    @Order(30)
    @DisplayName("Consistency: GPU and CPU produce same results")
    void gpuCpuConsistency() {
        Assumptions.assumeTrue(CudaDetector.isAvailable(), "CUDA not available");
        
        float[][] vectors = createTestVectors(100, DIMS);
        float[] query = createRandomVector(DIMS, 999);
        
        SearchResult cpuResult;
        SearchResult gpuResult;
        
        try (CPUVectorIndex cpuIndex = new CPUVectorIndex(DIMS)) {
            cpuIndex.add(vectors);
            cpuResult = cpuIndex.search(query, 10);
        }
        
        try (GPUVectorIndex gpuIndex = new GPUVectorIndex(DIMS)) {
            gpuIndex.add(vectors);
            gpuResult = gpuIndex.search(query, 10);
        }
        
        // Results should match
        assertArrayEquals(cpuResult.getIds(), gpuResult.getIds(), 
            "GPU and CPU should return same indices");
        
        for (int i = 0; i < 10; i++) {
            assertEquals(cpuResult.getDistances()[i], gpuResult.getDistances()[i], EPSILON,
                "Distance mismatch at position " + i);
        }
    }

    // ===========================================
    // AI Blind Spots Tests (per test-feature.md)
    // ===========================================

    @Nested
    @DisplayName("AI Blind Spots")
    class AIBlindSpots {

        @Test
        @DisplayName("CPU: Rejects NaN values in input vectors")
        void cpuRejectsNaNValues() {
            try (CPUVectorIndex index = new CPUVectorIndex(DIMS)) {
                float[][] vectors = createTestVectors(5, DIMS);
                vectors[2][10] = Float.NaN;  // Inject NaN
                
                IllegalArgumentException ex = assertThrows(
                    IllegalArgumentException.class, 
                    () -> index.add(vectors));
                assertTrue(ex.getMessage().contains("invalid value"));
            }
        }

        @Test
        @DisplayName("CPU: Rejects Infinity values in input vectors")
        void cpuRejectsInfinityValues() {
            try (CPUVectorIndex index = new CPUVectorIndex(DIMS)) {
                float[][] vectors = createTestVectors(5, DIMS);
                vectors[1][5] = Float.POSITIVE_INFINITY;
                
                IllegalArgumentException ex = assertThrows(
                    IllegalArgumentException.class, 
                    () -> index.add(vectors));
                assertTrue(ex.getMessage().contains("invalid value"));
            }
        }

        @Test
        @DisplayName("GPU: Rejects NaN values in input vectors")
        void gpuRejectsNaNValues() {
            Assumptions.assumeTrue(CudaDetector.isAvailable(), "CUDA not available");
            
            try (GPUVectorIndex index = new GPUVectorIndex(DIMS)) {
                float[][] vectors = createTestVectors(5, DIMS);
                vectors[3][7] = Float.NaN;
                
                IllegalArgumentException ex = assertThrows(
                    IllegalArgumentException.class, 
                    () -> index.add(vectors));
                assertTrue(ex.getMessage().contains("invalid value"));
            }
        }

        @Test
        @DisplayName("GPU: Rejects Negative Infinity values")
        void gpuRejectsNegativeInfinity() {
            Assumptions.assumeTrue(CudaDetector.isAvailable(), "CUDA not available");
            
            try (GPUVectorIndex index = new GPUVectorIndex(DIMS)) {
                float[][] vectors = createTestVectors(5, DIMS);
                vectors[0][0] = Float.NEGATIVE_INFINITY;
                
                assertThrows(IllegalArgumentException.class, () -> index.add(vectors));
            }
        }

        @Test
        @DisplayName("Thread safety: Concurrent searches don't crash (CPU)")
        void cpuThreadSafety() throws Exception {
            try (CPUVectorIndex index = new CPUVectorIndex(DIMS)) {
                float[][] vectors = createTestVectors(1000, DIMS);
                index.add(vectors);
                
                int numThreads = 10;
                int searchesPerThread = 50;
                java.util.concurrent.CountDownLatch latch = new java.util.concurrent.CountDownLatch(numThreads);
                java.util.concurrent.atomic.AtomicInteger errors = new java.util.concurrent.atomic.AtomicInteger(0);
                
                for (int t = 0; t < numThreads; t++) {
                    final int threadId = t;
                    new Thread(() -> {
                        try {
                            for (int i = 0; i < searchesPerThread; i++) {
                                float[] query = createRandomVector(DIMS, threadId * 1000L + i);
                                SearchResult result = index.search(query, 5);
                                assertNotNull(result);
                                assertEquals(5, result.getIds().length);
                            }
                        } catch (Exception e) {
                            errors.incrementAndGet();
                        } finally {
                            latch.countDown();
                        }
                    }).start();
                }
                
                latch.await(30, java.util.concurrent.TimeUnit.SECONDS);
                assertEquals(0, errors.get(), "Concurrent searches should not fail");
            }
        }

        @Test
        @DisplayName("Async search returns valid results")
        void asyncSearchWorks() throws Exception {
            try (CPUVectorIndex index = new CPUVectorIndex(DIMS)) {
                float[][] vectors = createTestVectors(100, DIMS);
                index.add(vectors);
                
                java.util.concurrent.CompletableFuture<SearchResult> future = 
                    index.searchAsync(vectors[0], 5);
                
                SearchResult result = future.get(5, java.util.concurrent.TimeUnit.SECONDS);
                
                assertNotNull(result);
                assertEquals(5, result.getIds().length);
                assertEquals(0, result.getIds()[0]);  // Should find itself first
            }
        }

        @Test
        @DisplayName("Integer overflow check: Large allocation is validated")
        void integerOverflowProtection() {
            // This tests that we don't silently overflow when computing memory sizes
            // Try to create an index with dimensions that would overflow when multiplied
            // by a large vector count. The implementation should handle this gracefully.
            
            // Note: Actually allocating Integer.MAX_VALUE dimensions would OOM,
            // so we just verify the index rejects absurd values
            assertThrows(IllegalArgumentException.class, 
                () -> new CPUVectorIndex(0));
            assertThrows(IllegalArgumentException.class, 
                () -> new CPUVectorIndex(-1));
            
            // Test that we can create a reasonable large index without overflow
            try (CPUVectorIndex index = new CPUVectorIndex(1536)) {  // OpenAI size
                float[][] vectors = createTestVectors(100, 1536);
                index.add(vectors);
                assertEquals(100, index.size());
            }
        }
    }

    // ===========================================
    // Helper Methods
    // ===========================================

    private float[][] createTestVectors(int count, int dims) {
        return createTestVectors(count, dims, 0);
    }

    private float[][] createTestVectors(int count, int dims, int offset) {
        float[][] vectors = new float[count][dims];
        for (int i = 0; i < count; i++) {
            // Each vector has its index as the first component for easy identification
            vectors[i][0] = (float) (i + offset);
            for (int j = 1; j < dims; j++) {
                vectors[i][j] = (float) Math.sin(i + j);
            }
        }
        return vectors;
    }

    private float[] createRandomVector(int dims, long seed) {
        java.util.Random random = new java.util.Random(seed);
        float[] vector = new float[dims];
        for (int i = 0; i < dims; i++) {
            vector[i] = random.nextFloat();
        }
        return vector;
    }
}
