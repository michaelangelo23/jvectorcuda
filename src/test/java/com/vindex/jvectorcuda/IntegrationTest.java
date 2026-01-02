package com.vindex.jvectorcuda;

import com.vindex.jvectorcuda.cpu.CPUVectorIndex;
import com.vindex.jvectorcuda.gpu.GPUVectorIndex;
import org.junit.jupiter.api.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for end-to-end workflows, CPU-GPU switching, and large-scale operations.
 */
class IntegrationTest {

    private static final int DIMENSIONS = 128;
    private static final float EPSILON = 1e-3f;

    @Nested
    @DisplayName("End-to-End Workflow Tests")
    class EndToEndWorkflows {

        @Test
        @DisplayName("Complete workflow: factory → add → search → close (CPU)")
        void completeWorkflowCpu() {
            try (VectorIndex index = VectorIndexFactory.cpu(DIMENSIONS)) {
                // Add vectors
                float[][] vectors = createRandomVectors(1000, DIMENSIONS);
                index.add(vectors);
                assertEquals(1000, index.size());

                // Search
                float[] query = createRandomVector(DIMENSIONS);
                SearchResult result = index.search(query, 10);
                
                assertNotNull(result);
                assertEquals(10, result.getIds().length);
                assertEquals(10, result.getDistances().length);
                assertTrue(result.getSearchTimeMs() >= 0);
                
                // Verify distances are sorted ascending
                for (int i = 1; i < result.getDistances().length; i++) {
                    assertTrue(result.getDistances()[i] >= result.getDistances()[i-1],
                        "Distances should be sorted ascending");
                }
            }
        }

        @Test
        @DisplayName("Complete workflow: factory → add → search → close (GPU)")
        void completeWorkflowGpu() {
            Assumptions.assumeTrue(CudaDetector.isAvailable(), "CUDA not available");
            
            try (VectorIndex index = VectorIndexFactory.gpu(DIMENSIONS)) {
                // Add vectors
                float[][] vectors = createRandomVectors(1000, DIMENSIONS);
                index.add(vectors);
                assertEquals(1000, index.size());

                // Search
                float[] query = createRandomVector(DIMENSIONS);
                SearchResult result = index.search(query, 10);
                
                assertNotNull(result);
                assertEquals(10, result.getIds().length);
                assertEquals(10, result.getDistances().length);
                assertTrue(result.getSearchTimeMs() >= 0);
            }
        }

        @Test
        @DisplayName("Complete workflow: auto-detection → operations")
        void completeWorkflowAuto() {
            try (VectorIndex index = VectorIndexFactory.auto(DIMENSIONS)) {
                float[][] vectors = createRandomVectors(500, DIMENSIONS);
                index.add(vectors);
                
                SearchResult result = index.search(vectors[0], 5);
                
                // First result should be the query itself (distance ~0)
                assertEquals(0, result.getIds()[0]);
                assertTrue(result.getDistances()[0] < EPSILON);
            }
        }

        @Test
        @DisplayName("Batch search workflow")
        void batchSearchWorkflow() {
            try (VectorIndex index = VectorIndexFactory.auto(DIMENSIONS)) {
                float[][] database = createRandomVectors(1000, DIMENSIONS);
                index.add(database);
                
                float[][] queries = createRandomVectors(50, DIMENSIONS);
                List<SearchResult> results = index.searchBatch(queries, 10);
                
                assertEquals(50, results.size());
                for (SearchResult result : results) {
                    assertEquals(10, result.getIds().length);
                    assertEquals(10, result.getDistances().length);
                }
            }
        }

        @Test
        @DisplayName("Async search workflow")
        void asyncSearchWorkflow() throws Exception {
            // Use CPU for async - GPU async requires additional synchronization
            try (VectorIndex index = VectorIndexFactory.cpu(DIMENSIONS)) {
                float[][] vectors = createRandomVectors(1000, DIMENSIONS);
                index.add(vectors);
                
                // Test that async search completes successfully
                float[] query = createRandomVector(DIMENSIONS);
                CompletableFuture<SearchResult> future = index.searchAsync(query, 10);
                SearchResult result = future.get(5, TimeUnit.SECONDS);
                assertNotNull(result);
                assertEquals(10, result.getIds().length);
            }
        }

        @Test
        @DisplayName("Different distance metrics workflow")
        void differentMetricsWorkflow() {
            for (DistanceMetric metric : DistanceMetric.values()) {
                try (VectorIndex index = VectorIndexFactory.auto(DIMENSIONS, metric)) {
                    float[][] vectors = createRandomVectors(100, DIMENSIONS);
                    index.add(vectors);
                    
                    SearchResult result = index.search(vectors[0], 5);
                    assertNotNull(result);
                    assertEquals(5, result.getIds().length);
                }
            }
        }
    }

    @Nested
    @DisplayName("CPU ↔ GPU Switching Tests")
    class CpuGpuSwitching {

        @Test
        @DisplayName("CPU and GPU produce similar results on same data")
        void cpuGpuSimilarResults() {
            Assumptions.assumeTrue(CudaDetector.isAvailable(), "CUDA not available");
            
            float[][] database = createRandomVectors(500, DIMENSIONS);
            float[] query = createRandomVector(DIMENSIONS);
            
            SearchResult cpuResult;
            try (VectorIndex cpuIndex = new CPUVectorIndex(DIMENSIONS)) {
                cpuIndex.add(database);
                cpuResult = cpuIndex.search(query, 10);
            }
            
            SearchResult gpuResult;
            try (VectorIndex gpuIndex = new GPUVectorIndex(DIMENSIONS, DistanceMetric.EUCLIDEAN)) {
                gpuIndex.add(database);
                gpuResult = gpuIndex.search(query, 10);
            }
            
            // GPU uses brute-force (exact), CPU uses HNSW (approximate)
            // Results may differ but top results should overlap significantly
            assertNotNull(cpuResult);
            assertNotNull(gpuResult);
            
            // Check that at least 50% of top-10 results overlap
            int overlap = countOverlap(cpuResult.getIds(), gpuResult.getIds());
            assertTrue(overlap >= 5, "At least 50% overlap expected between CPU and GPU results, got " + overlap);
        }

        @Test
        @DisplayName("Auto-detection fallback works when GPU unavailable")
        void autoDetectionFallback() {
            // Auto should always work (falls back to CPU if no GPU)
            try (VectorIndex index = VectorIndexFactory.auto(DIMENSIONS)) {
                float[][] vectors = createRandomVectors(100, DIMENSIONS);
                index.add(vectors);
                
                SearchResult result = index.search(vectors[0], 5);
                assertNotNull(result);
            }
        }

        @Test
        @DisplayName("Switching metrics between indices")
        void switchingMetrics() {
            // Create a database with extreme values to ensure different rankings
            float[][] database = new float[200][DIMENSIONS];
            for (int i = 0; i < database.length; i++) {
                for (int j = 0; j < DIMENSIONS; j++) {
                    // Mix of large positive, negative, and small values
                    int mod = i % 3;
                    if (mod == 0) {
                        database[i][j] = 100.0f;
                    } else if (mod == 1) {
                        database[i][j] = -100.0f;
                    } else {
                        database[i][j] = 0.1f;
                    }
                }
            }
            // Query with all positive values
            float[] query = new float[DIMENSIONS];
            for (int j = 0; j < DIMENSIONS; j++) {
                query[j] = 1.0f;
            }
            
            // Search with Euclidean
            SearchResult euclideanResult;
            try (VectorIndex index = VectorIndexFactory.auto(DIMENSIONS, DistanceMetric.EUCLIDEAN)) {
                index.add(database);
                euclideanResult = index.search(query, 5);
            }
            
            // Search with Cosine
            SearchResult cosineResult;
            try (VectorIndex index = VectorIndexFactory.auto(DIMENSIONS, DistanceMetric.COSINE)) {
                index.add(database);
                cosineResult = index.search(query, 5);
            }
            
            // Both searches should succeed
            assertEquals(5, euclideanResult.getIds().length);
            assertEquals(5, cosineResult.getIds().length);
        }
    }

    @Nested
    @DisplayName("Large-Scale Data Tests")
    class LargeScaleTests {

        @Test
        @DisplayName("100K vectors search")
        @Timeout(value = 30, unit = TimeUnit.SECONDS)
        void hundredThousandVectors() {
            Assumptions.assumeTrue(CudaDetector.isAvailable(), "CUDA not available for large-scale test");
            
            try (VectorIndex index = VectorIndexFactory.auto(DIMENSIONS)) {
                float[][] vectors = createRandomVectors(100_000, DIMENSIONS);
                index.add(vectors);
                assertEquals(100_000, index.size());
                
                SearchResult result = index.search(vectors[0], 10);
                assertEquals(10, result.getIds().length);
                
                // Verify search is reasonably fast
                assertTrue(result.getSearchTimeMs() < 1000, 
                    "100K vector search should complete in under 1 second");
            }
        }

        @Test
        @DisplayName("Large batch queries (1000 queries)")
        @Timeout(value = 60, unit = TimeUnit.SECONDS)
        void largeBatchQueries() {
            try (VectorIndex index = VectorIndexFactory.auto(DIMENSIONS)) {
                float[][] database = createRandomVectors(10_000, DIMENSIONS);
                index.add(database);
                
                float[][] queries = createRandomVectors(1000, DIMENSIONS);
                List<SearchResult> results = index.searchBatch(queries, 10);
                
                assertEquals(1000, results.size());
                for (SearchResult result : results) {
                    assertEquals(10, result.getIds().length);
                }
            }
        }

        @Test
        @DisplayName("High-dimensional vectors (1536D like OpenAI)")
        void highDimensionalVectors() {
            int highDims = 1536;
            
            try (VectorIndex index = VectorIndexFactory.auto(highDims)) {
                float[][] vectors = createRandomVectors(1000, highDims);
                index.add(vectors);
                
                SearchResult result = index.search(vectors[0], 5);
                assertEquals(5, result.getIds().length);
                
                // First result should be query itself
                assertEquals(0, result.getIds()[0]);
            }
        }
    }

    @Nested
    @DisplayName("Stress Tests")
    class StressTests {

        @Test
        @DisplayName("Rapid open/close cycles")
        void rapidOpenCloseCycles() {
            for (int i = 0; i < 10; i++) {
                try (VectorIndex index = VectorIndexFactory.auto(DIMENSIONS)) {
                    float[][] vectors = createRandomVectors(100, DIMENSIONS);
                    index.add(vectors);
                    index.search(vectors[0], 5);
                }
            }
            // Should not leak resources or crash
        }

        @Test
        @DisplayName("Concurrent searches (multiple threads)")
        void concurrentSearches() throws Exception {
            try (VectorIndex index = VectorIndexFactory.autoThreadSafe(DIMENSIONS)) {
                float[][] database = createRandomVectors(5000, DIMENSIONS);
                index.add(database);
                
                ExecutorService executor = Executors.newFixedThreadPool(10);
                CountDownLatch latch = new CountDownLatch(100);
                
                for (int i = 0; i < 100; i++) {
                    executor.submit(() -> {
                        try {
                            float[] query = createRandomVector(DIMENSIONS);
                            SearchResult result = index.search(query, 10);
                            assertEquals(10, result.getIds().length);
                        } finally {
                            latch.countDown();
                        }
                    });
                }
                
                assertTrue(latch.await(30, TimeUnit.SECONDS));
                executor.shutdown();
            }
        }

        @Test
        @DisplayName("Memory stress test")
        @Timeout(value = 60, unit = TimeUnit.SECONDS)
        void memoryStressTest() {
            Assumptions.assumeTrue(CudaDetector.isAvailable(), "CUDA not available for memory stress test");
            
            // Create and destroy multiple indices to stress memory management
            for (int i = 0; i < 5; i++) {
                try (VectorIndex index = VectorIndexFactory.gpu(DIMENSIONS)) {
                    float[][] vectors = createRandomVectors(50_000, DIMENSIONS);
                    index.add(vectors);
                    
                    for (int j = 0; j < 10; j++) {
                        float[] query = createRandomVector(DIMENSIONS);
                        index.search(query, 10);
                    }
                }
                // Force GC to clean up
                System.gc();
            }
        }

        @Test
        @DisplayName("Edge case: k larger than database size")
        void kLargerThanDatabase() {
            try (VectorIndex index = VectorIndexFactory.auto(DIMENSIONS)) {
                float[][] vectors = createRandomVectors(5, DIMENSIONS);
                index.add(vectors);
                
                // Request 10 neighbors but only 5 vectors exist
                SearchResult result = index.search(vectors[0], 10);
                
                // Should return only available vectors
                assertTrue(result.getIds().length <= 5);
            }
        }
    }

    @Nested
    @DisplayName("Error Handling Tests")
    class ErrorHandlingTests {

        @Test
        @DisplayName("Invalid dimensions throw exception")
        void invalidDimensions() {
            assertThrows(IllegalArgumentException.class, 
                () -> VectorIndexFactory.auto(0));
            assertThrows(IllegalArgumentException.class, 
                () -> VectorIndexFactory.auto(-10));
        }

        @Test
        @DisplayName("Null vectors throw exception")
        void nullVectorsThrow() {
            try (VectorIndex index = VectorIndexFactory.auto(DIMENSIONS)) {
                assertThrows(IllegalArgumentException.class, 
                    () -> index.add(null));
            }
        }

        @Test
        @DisplayName("Wrong dimensions throw exception")
        void wrongDimensionsThrow() {
            try (VectorIndex index = VectorIndexFactory.auto(DIMENSIONS)) {
                float[][] wrongDims = {{1.0f, 2.0f}}; // Only 2 dimensions
                assertThrows(IllegalArgumentException.class, 
                    () -> index.add(wrongDims));
            }
        }

        @Test
        @DisplayName("Search on empty index")
        void searchOnEmptyIndex() {
            try (VectorIndex index = VectorIndexFactory.auto(DIMENSIONS)) {
                float[] query = createRandomVector(DIMENSIONS);
                SearchResult result = index.search(query, 10);
                
                assertEquals(0, result.getIds().length);
                assertEquals(0, result.getDistances().length);
            }
        }

        @Test
        @DisplayName("Operations after close throw exception")
        void operationsAfterClose() {
            VectorIndex index = VectorIndexFactory.auto(DIMENSIONS);
            index.close();
            
            assertThrows(Exception.class, 
                () -> index.add(createRandomVectors(10, DIMENSIONS)));
        }
    }

    // Helper methods

    private static float[][] createRandomVectors(int count, int dimensions) {
        Random random = new Random(42);
        float[][] vectors = new float[count][dimensions];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < dimensions; j++) {
                vectors[i][j] = random.nextFloat();
            }
        }
        return vectors;
    }

    private static float[] createRandomVector(int dimensions) {
        Random random = new Random();
        float[] vector = new float[dimensions];
        for (int i = 0; i < dimensions; i++) {
            vector[i] = random.nextFloat();
        }
        return vector;
    }

    private int countOverlap(int[] array1, int[] array2) {
        int count = 0;
        for (int id1 : array1) {
            for (int id2 : array2) {
                if (id1 == id2) {
                    count++;
                    break;
                }
            }
        }
        return count;
    }
}
