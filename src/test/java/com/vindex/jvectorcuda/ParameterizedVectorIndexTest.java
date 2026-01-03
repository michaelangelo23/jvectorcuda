package com.vindex.jvectorcuda;

import com.vindex.jvectorcuda.cpu.CPUVectorIndex;
import com.vindex.jvectorcuda.gpu.GPUVectorIndex;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Random;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Parameterized tests for vector index implementations.
 * Tests all combinations of index types, metrics, dimensions, and dataset sizes.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class ParameterizedVectorIndexTest {

    private static final float EPSILON = 0.001f;

    /**
     * Test configuration holder.
     */
    static class TestConfig {
        final IndexType indexType;
        final DistanceMetric metric;
        final int dimensions;
        final int vectorCount;
        final int k;

        TestConfig(IndexType indexType, DistanceMetric metric, int dimensions, int vectorCount, int k) {
            this.indexType = indexType;
            this.metric = metric;
            this.dimensions = dimensions;
            this.vectorCount = vectorCount;
            this.k = k;
        }

        @Override
        public String toString() {
            return String.format("%s[%s, %dd, %d vectors, k=%d]",
                indexType, metric, dimensions, vectorCount, k);
        }
    }

    enum IndexType {
        CPU, GPU, HYBRID
    }

    /**
     * Provides test configurations for parameterized tests.
     */
    Stream<Arguments> provideTestConfigurations() {
        return Stream.of(
            // CPU tests - all metrics and dimensions
            Arguments.of(new TestConfig(IndexType.CPU, DistanceMetric.EUCLIDEAN, 128, 100, 10)),
            Arguments.of(new TestConfig(IndexType.CPU, DistanceMetric.EUCLIDEAN, 384, 1000, 10)),
            Arguments.of(new TestConfig(IndexType.CPU, DistanceMetric.COSINE, 128, 100, 10)),
            Arguments.of(new TestConfig(IndexType.CPU, DistanceMetric.COSINE, 768, 1000, 5)),
            Arguments.of(new TestConfig(IndexType.CPU, DistanceMetric.INNER_PRODUCT, 384, 100, 10)),
            
            // GPU tests - if available
            Arguments.of(new TestConfig(IndexType.GPU, DistanceMetric.EUCLIDEAN, 384, 1000, 10)),
            Arguments.of(new TestConfig(IndexType.GPU, DistanceMetric.COSINE, 768, 1000, 10)),
            
            // Hybrid tests
            Arguments.of(new TestConfig(IndexType.HYBRID, DistanceMetric.EUCLIDEAN, 384, 1000, 10))
        );
    }

    /**
     * Provides edge case configurations.
     */
    Stream<Arguments> provideEdgeCaseConfigurations() {
        return Stream.of(
            // Empty index
            Arguments.of(new TestConfig(IndexType.CPU, DistanceMetric.EUCLIDEAN, 128, 0, 10)),
            
            // Single vector
            Arguments.of(new TestConfig(IndexType.CPU, DistanceMetric.EUCLIDEAN, 128, 1, 10)),
            
            // k > vector count
            Arguments.of(new TestConfig(IndexType.CPU, DistanceMetric.EUCLIDEAN, 128, 5, 10)),
            
            // Large k
            Arguments.of(new TestConfig(IndexType.CPU, DistanceMetric.EUCLIDEAN, 128, 100, 50))
        );
    }

    @ParameterizedTest
    @MethodSource("provideTestConfigurations")
    @DisplayName("Test search correctness across configurations")
    void testSearchCorrectness(TestConfig config) {
        VectorIndex index = createIndex(config);
        if (index == null) {
            // Skip if GPU not available
            return;
        }

        try {
            // Generate test data
            float[][] vectors = generateRandomVectors(config.vectorCount, config.dimensions, 42L);
            float[] query = generateRandomVector(config.dimensions, 100L);

            // Add vectors to index
            index.add(vectors);

            // Search
            SearchResult result = index.search(query, config.k);

            // Verify result
            assertNotNull(result, "Search result should not be null");
            assertNotNull(result.getIds(), "Result IDs should not be null");
            assertNotNull(result.getDistances(), "Result distances should not be null");

            int expectedResultCount = Math.min(config.k, config.vectorCount);
            assertEquals(expectedResultCount, result.getIds().length, "Should return correct number of results");
            assertEquals(expectedResultCount, result.getDistances().length, "Distances array should match IDs array length");

            // Verify distances are sorted (closest first)
            for (int i = 1; i < result.getDistances().length; i++) {
                assertTrue(result.getDistances()[i] >= result.getDistances()[i - 1],
                    "Distances should be sorted in ascending order");
            }

            // Verify all IDs are valid
            for (int id : result.getIds()) {
                assertTrue(id >= 0 && id < config.vectorCount, "ID should be within valid range");
            }
        } finally {
            index.close();
        }
    }

    @ParameterizedTest
    @MethodSource("provideEdgeCaseConfigurations")
    @DisplayName("Test edge cases")
    void testEdgeCases(TestConfig config) {
        VectorIndex index = createIndex(config);
        if (index == null) {
            return;
        }

        try {
            float[][] vectors = generateRandomVectors(config.vectorCount, config.dimensions, 42L);
            float[] query = generateRandomVector(config.dimensions, 100L);

            if (config.vectorCount > 0) {
                index.add(vectors);
            }

            SearchResult result = index.search(query, config.k);

            assertNotNull(result);
            int expectedCount = Math.min(config.k, config.vectorCount);
            assertEquals(expectedCount, result.getIds().length);
            assertEquals(expectedCount, result.getDistances().length);
        } finally {
            index.close();
        }
    }

    @ParameterizedTest
    @MethodSource("provideTestConfigurations")
    @DisplayName("Test CPU vs GPU correctness (when GPU available)")
    void testCpuGpuCorrectness(TestConfig config) {
        if (config.indexType != IndexType.CPU) {
            return; // Only run for CPU configs
        }

        if (!CudaDetector.isAvailable()) {
            return; // Skip if GPU not available
        }

        // Create both CPU and GPU indices
        try (CPUVectorIndex cpuIndex = new CPUVectorIndex(config.dimensions);
             GPUVectorIndex gpuIndex = new GPUVectorIndex(config.dimensions, config.metric)) {

            float[][] vectors = generateRandomVectors(config.vectorCount, config.dimensions, 42L);
            float[] query = generateRandomVector(config.dimensions, 100L);

            cpuIndex.add(vectors);
            gpuIndex.add(vectors);

            SearchResult cpuResult = cpuIndex.search(query, config.k);
            SearchResult gpuResult = gpuIndex.search(query, config.k);

            // Results should be similar (allowing for floating point differences)
            assertEquals(cpuResult.getIds().length, gpuResult.getIds().length,
                "CPU and GPU should return same number of results");

            // Check that top results are similar
            // Note: Due to floating point precision, exact matches may differ slightly
            for (int i = 0; i < Math.min(3, cpuResult.getIds().length); i++) {
                float cpuDist = cpuResult.getDistances()[i];
                float gpuDist = gpuResult.getDistances()[i];
                
                // Allow 1% difference for floating point errors
                float tolerance = Math.max(EPSILON, cpuDist * 0.01f);
                assertTrue(Math.abs(cpuDist - gpuDist) < tolerance,
                    String.format("CPU and GPU distances should be similar at rank %d: CPU=%.6f, GPU=%.6f", 
                        i, cpuDist, gpuDist));
            }
        }
    }

    /**
     * Creates an index based on the test configuration.
     */
    private VectorIndex createIndex(TestConfig config) {
        return switch (config.indexType) {
            case CPU -> new CPUVectorIndex(config.dimensions);
            case GPU -> {
                if (!CudaDetector.isAvailable()) {
                    yield null; // Skip GPU tests if not available
                }
                yield new GPUVectorIndex(config.dimensions, config.metric);
            }
            case HYBRID -> new HybridVectorIndex(config.dimensions, config.metric);
        };
    }

    /**
     * Generates random vectors for testing.
     */
    private float[][] generateRandomVectors(int count, int dimensions, long seed) {
        Random random = new Random(seed);
        float[][] vectors = new float[count][dimensions];

        for (int i = 0; i < count; i++) {
            for (int j = 0; j < dimensions; j++) {
                vectors[i][j] = random.nextFloat() * 2.0f - 1.0f;
            }
        }

        return vectors;
    }

    /**
     * Generates a single random vector for testing.
     */
    private float[] generateRandomVector(int dimensions, long seed) {
        Random random = new Random(seed);
        float[] vector = new float[dimensions];

        for (int i = 0; i < dimensions; i++) {
            vector[i] = random.nextFloat() * 2.0f - 1.0f;
        }

        return vector;
    }
}
