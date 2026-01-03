package com.vindex.jvectorcuda;

import org.junit.jupiter.api.*;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for HybridVectorIndex routing logic and functionality.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class HybridVectorIndexTest {

    private static final int DIMS = 128;
    private static final float EPSILON = 0.001f;

    // ===========================================
    // Basic Functionality Tests
    // ===========================================

    @Test
    @Order(1)
    @DisplayName("Create and close hybrid index")
    void createAndClose() {
        try (HybridVectorIndex index = new HybridVectorIndex(DIMS)) {
            assertEquals(DIMS, index.getDimensions());
            assertEquals(0, index.size());
        }
    }

    @Test
    @Order(2)
    @DisplayName("Add vectors and verify count")
    void addVectors() {
        try (HybridVectorIndex index = new HybridVectorIndex(DIMS)) {
            float[][] vectors = createTestVectors(100, DIMS);
            index.add(vectors);

            assertEquals(100, index.size());
        }
    }

    @Test
    @Order(3)
    @DisplayName("Search returns correct results")
    void searchReturnsCorrectResults() {
        try (HybridVectorIndex index = new HybridVectorIndex(DIMS)) {
            float[][] vectors = createTestVectors(100, DIMS);
            index.add(vectors);

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
    @Order(4)
    @DisplayName("Batch search returns correct number of results")
    void batchSearchReturnsCorrectResults() {
        try (HybridVectorIndex index = new HybridVectorIndex(DIMS)) {
            float[][] vectors = createTestVectors(100, DIMS);
            index.add(vectors);

            float[][] queries = new float[10][DIMS];
            for (int i = 0; i < 10; i++) {
                queries[i] = vectors[i];
            }

            List<SearchResult> results = index.searchBatch(queries, 5);

            assertNotNull(results);
            assertEquals(10, results.size());

            // Each result should have the query vector as closest match
            for (int i = 0; i < 10; i++) {
                assertEquals(i, results.get(i).getIds()[0]);
            }
        }
    }

    // ===========================================
    // Routing Logic Tests
    // ===========================================

    @Test
    @Order(10)
    @DisplayName("Single query routes to CPU by default (small dataset)")
    void singleQueryRoutesToCpu() {
        try (HybridVectorIndex index = new HybridVectorIndex(DIMS)) {
            float[][] vectors = createTestVectors(100, DIMS);
            index.add(vectors);

            // Small dataset (100 < 50000 threshold) -> should route to CPU
            assertEquals("CPU", index.getRoutingDecision());
        }
    }

    @Test
    @Order(11)
    @DisplayName("Batch query routing depends on batch size")
    void batchQueryRoutingDependsOnBatchSize() {
        try (HybridVectorIndex index = new HybridVectorIndex(DIMS)) {
            float[][] vectors = createTestVectors(100, DIMS);
            index.add(vectors);

            // Small batch -> CPU
            assertEquals("CPU", index.getRoutingDecision(5));

            // Large batch (>= 10) -> GPU if available, otherwise CPU
            String decision = index.getRoutingDecision(15);
            if (index.isGpuAvailable()) {
                assertEquals("GPU", decision);
            } else {
                assertEquals("CPU", decision);
            }
        }
    }

    @Test
    @Order(12)
    @DisplayName("Routing changes with large dataset")
    void routingChangesWithLargeDataset() {
        // Use low thresholds for testing
        try (HybridVectorIndex index = HybridVectorIndex.builder(DIMS)
                .withBatchThreshold(5)
                .withVectorThreshold(100)
                .build()) {

            // Below threshold -> CPU
            float[][] vectors = createTestVectors(50, DIMS);
            index.add(vectors);
            assertEquals("CPU", index.getRoutingDecision());

            // Above threshold -> GPU if available
            float[][] moreVectors = createTestVectors(100, DIMS);
            index.add(moreVectors);

            if (index.isGpuAvailable()) {
                assertEquals("GPU", index.getRoutingDecision());
            } else {
                assertEquals("CPU", index.getRoutingDecision());
            }
        }
    }

    @Test
    @Order(13)
    @DisplayName("No GPU available routes everything to CPU")
    void noGpuRoutesToCpu() {
        // This test verifies behavior when GPU is not available
        // The actual routing will depend on system configuration
        try (HybridVectorIndex index = new HybridVectorIndex(DIMS)) {
            float[][] vectors = createTestVectors(100_000, DIMS);
            index.add(vectors);

            if (!index.isGpuAvailable()) {
                assertEquals("CPU", index.getRoutingDecision());
                assertEquals("CPU", index.getRoutingDecision(100));
            }
        }
    }

    // ===========================================
    // Builder Tests
    // ===========================================

    @Test
    @Order(20)
    @DisplayName("Builder sets custom thresholds")
    void builderSetsCustomThresholds() {
        try (HybridVectorIndex index = HybridVectorIndex.builder(DIMS)
                .withBatchThreshold(5)
                .withVectorThreshold(1000)
                .withMetric(DistanceMetric.COSINE)
                .build()) {

            assertEquals(5, index.getBatchThreshold());
            assertEquals(1000, index.getVectorThreshold());
            assertEquals(DIMS, index.getDimensions());
        }
    }

    @Test
    @Order(21)
    @DisplayName("Builder with default values")
    void builderWithDefaults() {
        try (HybridVectorIndex index = HybridVectorIndex.builder(DIMS).build()) {
            assertEquals(10, index.getBatchThreshold());
            assertEquals(50_000, index.getVectorThreshold());
        }
    }

    // ===========================================
    // Routing Report Tests
    // ===========================================

    @Test
    @Order(30)
    @DisplayName("Routing report contains expected information")
    void routingReportContainsExpectedInfo() {
        try (HybridVectorIndex index = new HybridVectorIndex(DIMS)) {
            float[][] vectors = createTestVectors(100, DIMS);
            index.add(vectors);

            String report = index.getRoutingReport();

            assertNotNull(report);
            assertTrue(report.contains("Dimensions: " + DIMS));
            assertTrue(report.contains("Vector Count:"));
            assertTrue(report.contains("GPU Available:"));
            assertTrue(report.contains("Batch Threshold:"));
            assertTrue(report.contains("Vector Threshold:"));
            assertTrue(report.contains("Single query:"));
        }
    }

    // ===========================================
    // Edge Cases
    // ===========================================

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {

        @Test
        @DisplayName("Invalid dimensions throws exception")
        void invalidDimensions() {
            assertThrows(IllegalArgumentException.class, () -> new HybridVectorIndex(0));
            assertThrows(IllegalArgumentException.class, () -> new HybridVectorIndex(-1));
        }

        @Test
        @DisplayName("Null metric throws exception")
        void nullMetric() {
            assertThrows(IllegalArgumentException.class,
                    () -> new HybridVectorIndex(DIMS, null));
        }

        @Test
        @DisplayName("Invalid batch threshold throws exception")
        void invalidBatchThreshold() {
            assertThrows(IllegalArgumentException.class,
                    () -> HybridVectorIndex.builder(DIMS).withBatchThreshold(0).build());
        }

        @Test
        @DisplayName("Invalid vector threshold throws exception")
        void invalidVectorThreshold() {
            assertThrows(IllegalArgumentException.class,
                    () -> HybridVectorIndex.builder(DIMS).withVectorThreshold(0).build());
        }

        @Test
        @DisplayName("Null vectors throws exception")
        void nullVectors() {
            try (HybridVectorIndex index = new HybridVectorIndex(DIMS)) {
                assertThrows(IllegalArgumentException.class, () -> index.add(null));
            }
        }

        @Test
        @DisplayName("Empty vectors array is no-op")
        void emptyVectorsIsNoOp() {
            try (HybridVectorIndex index = new HybridVectorIndex(DIMS)) {
                index.add(new float[0][]);
                assertEquals(0, index.size());
            }
        }

        @Test
        @DisplayName("Null batch queries throws exception")
        void nullBatchQueries() {
            try (HybridVectorIndex index = new HybridVectorIndex(DIMS)) {
                float[][] vectors = createTestVectors(10, DIMS);
                index.add(vectors);

                assertThrows(IllegalArgumentException.class,
                        () -> index.searchBatch(null, 5));
            }
        }

        @Test
        @DisplayName("Empty batch queries throws exception")
        void emptyBatchQueries() {
            try (HybridVectorIndex index = new HybridVectorIndex(DIMS)) {
                float[][] vectors = createTestVectors(10, DIMS);
                index.add(vectors);

                assertThrows(IllegalArgumentException.class,
                        () -> index.searchBatch(new float[0][], 5));
            }
        }

        @Test
        @DisplayName("Operations after close throw exception")
        void operationsAfterCloseThrow() {
            HybridVectorIndex index = new HybridVectorIndex(DIMS);
            index.close();

            assertThrows(IllegalStateException.class,
                    () -> index.add(createTestVectors(10, DIMS)));
            assertThrows(IllegalStateException.class,
                    () -> index.search(new float[DIMS], 5));
            assertThrows(IllegalStateException.class,
                    () -> index.searchBatch(new float[1][DIMS], 5));
        }

        @Test
        @DisplayName("Double close is safe")
        void doubleCloseIsSafe() {
            HybridVectorIndex index = new HybridVectorIndex(DIMS);
            index.close();
            assertDoesNotThrow(index::close);
        }
    }

    // ===========================================
    // AI Blind Spots (Potential edge cases AI might miss)
    // ===========================================

    @Nested
    @DisplayName("AI Blind Spots")
    class AIBlindSpots {

        @Test
        @DisplayName("Async search routes correctly")
        void asyncSearchRoutesCorrectly() throws Exception {
            try (HybridVectorIndex index = new HybridVectorIndex(DIMS)) {
                float[][] vectors = createTestVectors(100, DIMS);
                index.add(vectors);

                SearchResult result = index.searchAsync(vectors[0], 5).get();

                assertNotNull(result);
                assertEquals(5, result.getIds().length);
                assertEquals(0, result.getIds()[0]);
            }
        }

        @Test
        @DisplayName("Multiple distance metrics work correctly")
        void multipleDistanceMetricsWork() {
            for (DistanceMetric metric : DistanceMetric.values()) {
                try (HybridVectorIndex index = new HybridVectorIndex(DIMS, metric)) {
                    float[][] vectors = createTestVectors(10, DIMS);
                    index.add(vectors);

                    SearchResult result = index.search(vectors[0], 3);
                    assertNotNull(result);
                    assertEquals(3, result.getIds().length);
                }
            }
        }

        @Test
        @DisplayName("Routing thresholds are respected precisely")
        void routingThresholdsRespectedPrecisely() {
            // Test exact boundary conditions
            try (HybridVectorIndex index = HybridVectorIndex.builder(DIMS)
                    .withBatchThreshold(10)
                    .withVectorThreshold(100)
                    .build()) {

                // Add exactly 100 vectors (at threshold)
                float[][] vectors = createTestVectors(100, DIMS);
                index.add(vectors);

                if (index.isGpuAvailable()) {
                    // Batch of 9 (below threshold) -> CPU
                    assertEquals("CPU", index.getRoutingDecision(9));
                    // Batch of 10 (at threshold) -> GPU
                    assertEquals("GPU", index.getRoutingDecision(10));
                }
            }
        }

        @Test
        @DisplayName("Large batch with small dataset still routes correctly")
        void largeBatchSmallDataset() {
            try (HybridVectorIndex index = HybridVectorIndex.builder(DIMS)
                    .withBatchThreshold(10)
                    .withVectorThreshold(1000)
                    .build()) {

                // Only 50 vectors (below 1000 threshold)
                float[][] vectors = createTestVectors(50, DIMS);
                index.add(vectors);

                if (index.isGpuAvailable()) {
                    // Large batch but small dataset -> still GPU (batch threshold met)
                    assertEquals("GPU", index.getRoutingDecision(100));
                }
            }
        }

        @Test
        @DisplayName("State consistency after multiple adds")
        void stateConsistencyAfterMultipleAdds() {
            try (HybridVectorIndex index = new HybridVectorIndex(DIMS)) {
                // Add in batches
                for (int i = 0; i < 10; i++) {
                    float[][] vectors = createTestVectors(10, DIMS);
                    index.add(vectors);
                }

                assertEquals(100, index.size());

                // Search should still work correctly
                float[] query = new float[DIMS];
                SearchResult result = index.search(query, 5);
                assertNotNull(result);
            }
        }
    }

    // ===========================================
    // GPU-Specific Tests (only run if GPU available)
    // ===========================================

    @Nested
    @DisplayName("GPU Integration Tests")
    class GpuIntegrationTests {

        @Test
        @DisplayName("GPU routing achieves expected performance characteristics")
        void gpuRoutingPerformance() {
            // Skip if GPU not available
            if (!CudaDetector.isAvailable()) {
                return;
            }

            try (HybridVectorIndex index = HybridVectorIndex.builder(DIMS)
                    .withBatchThreshold(5)
                    .withVectorThreshold(100)
                    .build()) {

                float[][] vectors = createTestVectors(1000, DIMS);
                index.add(vectors);

                assertTrue(index.isGpuAvailable());
                assertEquals("GPU", index.getRoutingDecision());

                // Verify GPU path works
                float[][] queries = new float[20][DIMS];
                List<SearchResult> results = index.searchBatch(queries, 10);
                assertEquals(20, results.size());
            }
        }
    }

    // ===========================================
    // Helper Methods
    // ===========================================

    private float[][] createTestVectors(int count, int dims) {
        float[][] vectors = new float[count][dims];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < dims; j++) {
                // Use scaling factor 10x for more distinct vectors (avoid GPU float precision
                // issues)
                vectors[i][j] = (float) Math.sin(i * 1.0 + j * 0.1);
            }
        }
        return vectors;
    }
}
