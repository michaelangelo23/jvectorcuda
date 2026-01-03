package com.vindex.jvectorcuda.benchmark;

import com.vindex.jvectorcuda.DistanceMetric;
import com.vindex.jvectorcuda.VectorIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Standard benchmark suite with comprehensive metrics.
 * Provides reproducible benchmarks with percentile latency,
 * throughput, and memory measurements.
 */
public class StandardBenchmarkSuite {

    private static final Logger logger = LoggerFactory.getLogger(StandardBenchmarkSuite.class);

    /**
     * Dataset definition for benchmarking.
     */
    public static class Dataset {
        private final String name;
        private final int vectorCount;
        private final int dimensions;
        private final float[][] vectors;
        private final float[][] queries;

        public Dataset(String name, int vectorCount, int dimensions, float[][] vectors, float[][] queries) {
            this.name = name;
            this.vectorCount = vectorCount;
            this.dimensions = dimensions;
            this.vectors = vectors;
            this.queries = queries;
        }

        public String getName() {
            return name;
        }

        public int getVectorCount() {
            return vectorCount;
        }

        public int getDimensions() {
            return dimensions;
        }

        public float[][] getVectors() {
            return vectors;
        }

        public float[][] getQueries() {
            return queries;
        }
    }

    /**
     * Creates synthetic dataset for testing.
     *
     * @param vectorCount Number of vectors
     * @param dimensions  Vector dimensions
     * @param queryCount  Number of query vectors
     * @param seed        Random seed for reproducibility
     * @return Synthetic dataset
     */
    public static Dataset createSyntheticDataset(int vectorCount, int dimensions, int queryCount, long seed) {
        Random random = new Random(seed);

        float[][] vectors = new float[vectorCount][dimensions];
        for (int i = 0; i < vectorCount; i++) {
            for (int j = 0; j < dimensions; j++) {
                vectors[i][j] = random.nextFloat() * 2.0f - 1.0f;
            }
        }

        float[][] queries = new float[queryCount][dimensions];
        for (int i = 0; i < queryCount; i++) {
            for (int j = 0; j < dimensions; j++) {
                queries[i][j] = random.nextFloat() * 2.0f - 1.0f;
            }
        }

        String name = String.format("Synthetic[%,d×%d]", vectorCount, dimensions);
        return new Dataset(name, vectorCount, dimensions, vectors, queries);
    }

    /**
     * Standard datasets for reproducible benchmarking.
     */
    public static Dataset[] STANDARD_DATASETS = {
            createSyntheticDataset(1_000, 384, 100, 42L), // Small: Quick iteration
            createSyntheticDataset(10_000, 384, 100, 42L), // Medium: Development
            createSyntheticDataset(100_000, 384, 100, 42L), // Large: Pre-production
    };

    /**
     * Comprehensive benchmark result with all metrics.
     */
    public static class ComprehensiveBenchmarkResult {
        private final String datasetName;
        private final int vectorCount;
        private final int dimensions;
        private final int k;
        private final DistanceMetric metric;

        private final double throughputQPS;
        private final PercentileMetrics latency;
        private final MemoryMetrics memory;
        private final double buildTimeMs;
        private final double recallAt10;
        private final long timestamp;

        private ComprehensiveBenchmarkResult(Builder builder) {
            this.datasetName = builder.datasetName;
            this.vectorCount = builder.vectorCount;
            this.dimensions = builder.dimensions;
            this.k = builder.k;
            this.metric = builder.metric;
            this.throughputQPS = builder.throughputQPS;
            this.latency = builder.latency;
            this.memory = builder.memory;
            this.buildTimeMs = builder.buildTimeMs;
            this.recallAt10 = builder.recallAt10;
            this.timestamp = builder.timestamp;
        }

        public static Builder builder() {
            return new Builder();
        }

        public String getDatasetName() {
            return datasetName;
        }

        public int getVectorCount() {
            return vectorCount;
        }

        public int getDimensions() {
            return dimensions;
        }

        public int getK() {
            return k;
        }

        public DistanceMetric getMetric() {
            return metric;
        }

        public double getThroughputQPS() {
            return throughputQPS;
        }

        public PercentileMetrics getLatency() {
            return latency;
        }

        public MemoryMetrics getMemory() {
            return memory;
        }

        public double getBuildTimeMs() {
            return buildTimeMs;
        }

        public double getRecallAt10() {
            return recallAt10;
        }

        public long getTimestamp() {
            return timestamp;
        }

        @Override
        public String toString() {
            return String.format(
                    "BenchmarkResult[dataset=%s, vectors=%,d, dims=%d, QPS=%.1f, p50=%.2fms, p95=%.2fms, p99=%.2fms]",
                    datasetName, vectorCount, dimensions, throughputQPS,
                    latency.getP50(), latency.getP95(), latency.getP99());
        }

        public static class Builder {
            private String datasetName = "Unknown";
            private int vectorCount;
            private int dimensions;
            private int k = 10;
            private DistanceMetric metric = DistanceMetric.EUCLIDEAN;
            private double throughputQPS;
            private PercentileMetrics latency;
            private MemoryMetrics memory;
            private double buildTimeMs;
            private double recallAt10 = 1.0;
            private long timestamp = System.currentTimeMillis();

            public Builder datasetName(String datasetName) {
                this.datasetName = datasetName;
                return this;
            }

            public Builder vectorCount(int vectorCount) {
                this.vectorCount = vectorCount;
                return this;
            }

            public Builder dimensions(int dimensions) {
                this.dimensions = dimensions;
                return this;
            }

            public Builder k(int k) {
                this.k = k;
                return this;
            }

            public Builder metric(DistanceMetric metric) {
                this.metric = metric;
                return this;
            }

            public Builder throughputQPS(double throughputQPS) {
                this.throughputQPS = throughputQPS;
                return this;
            }

            public Builder latency(PercentileMetrics latency) {
                this.latency = latency;
                return this;
            }

            public Builder memory(MemoryMetrics memory) {
                this.memory = memory;
                return this;
            }

            public Builder buildTimeMs(double buildTimeMs) {
                this.buildTimeMs = buildTimeMs;
                return this;
            }

            public Builder recallAt10(double recallAt10) {
                this.recallAt10 = recallAt10;
                return this;
            }

            public Builder timestamp(long timestamp) {
                this.timestamp = timestamp;
                return this;
            }

            public ComprehensiveBenchmarkResult build() {
                Objects.requireNonNull(latency, "latency must be set");
                Objects.requireNonNull(memory, "memory must be set");
                return new ComprehensiveBenchmarkResult(this);
            }
        }
    }

    /**
     * Runs comprehensive benchmark on a vector index with full metrics.
     *
     * @param index   Vector index to benchmark
     * @param dataset Dataset to use for benchmarking
     * @param k       Number of nearest neighbors to search
     * @return Comprehensive benchmark result
     */
    public ComprehensiveBenchmarkResult run(VectorIndex index, Dataset dataset, int k) {
        logger.info("Starting benchmark: dataset={}, k={}", dataset.getName(), k);

        // Measure build time
        long buildStart = System.nanoTime();
        index.add(dataset.getVectors());
        long buildEnd = System.nanoTime();
        double buildTimeMs = (buildEnd - buildStart) / 1_000_000.0;

        // Warmup phase (5 iterations)
        logger.debug("Warmup phase...");
        for (int i = 0; i < 5; i++) {
            for (float[] query : dataset.getQueries()) {
                index.search(query, k);
            }
        }

        // Measure latency for each query
        int queryCount = dataset.getQueries().length;
        double[] latencies = new double[queryCount];

        logger.debug("Measurement phase...");
        long totalStart = System.nanoTime();
        for (int i = 0; i < queryCount; i++) {
            long queryStart = System.nanoTime();
            index.search(dataset.getQueries()[i], k);
            long queryEnd = System.nanoTime();
            latencies[i] = (queryEnd - queryStart) / 1_000_000.0;
        }
        long totalEnd = System.nanoTime();

        // Calculate metrics
        double totalTimeMs = (totalEnd - totalStart) / 1_000_000.0;
        double throughputQPS = (queryCount * 1000.0) / totalTimeMs;
        PercentileMetrics latencyMetrics = new PercentileMetrics(latencies);

        // Capture memory
        long gpuMemoryUsed = 0;
        try {
            long total = com.vindex.jvectorcuda.gpu.VramUtil.getTotalVramBytes();
            long available = com.vindex.jvectorcuda.gpu.VramUtil.getAvailableVramBytes();
            if (total > 0 && available >= 0) {
                gpuMemoryUsed = total - available;
            }
        } catch (NoClassDefFoundError | Exception e) {
            // GPU not available or VramUtil not in classpath
            logger.debug("Could not capture GPU memory: {}", e.getMessage());
        }

        MemoryMetrics memoryMetrics = MemoryMetrics.capture(gpuMemoryUsed);

        logger.info("Benchmark complete: QPS={}, p50={}ms, p95={}ms, p99={}ms",
                String.format("%.1f", throughputQPS),
                String.format("%.2f", latencyMetrics.getP50()),
                String.format("%.2f", latencyMetrics.getP95()),
                String.format("%.2f", latencyMetrics.getP99()));

        return ComprehensiveBenchmarkResult.builder()
                .datasetName(dataset.getName())
                .vectorCount(dataset.getVectorCount())
                .dimensions(dataset.getDimensions())
                .k(k)
                .metric(DistanceMetric.EUCLIDEAN) // Note: VectorIndex interface doesn't expose metric yet
                .throughputQPS(throughputQPS)
                .latency(latencyMetrics)
                .memory(memoryMetrics)
                .buildTimeMs(buildTimeMs)
                .recallAt10(1.0) // Note: Recall requires ground truth which is not available here
                .build();
    }

    /**
     * Runs benchmark suite across all standard datasets.
     *
     * @param indexFactory Function to create index instances
     * @param k            Number of nearest neighbors
     * @return List of benchmark results
     */
    public List<ComprehensiveBenchmarkResult> runSuite(
            java.util.function.Function<Integer, VectorIndex> indexFactory, int k) {
        List<ComprehensiveBenchmarkResult> results = new ArrayList<>();

        for (Dataset dataset : STANDARD_DATASETS) {
            try {
                VectorIndex index = indexFactory.apply(dataset.getDimensions());
                ComprehensiveBenchmarkResult result = run(index, dataset, k);
                results.add(result);
                index.close();
            } catch (Exception e) {
                logger.error("Failed to benchmark dataset {}: {}", dataset.getName(), e.getMessage(), e);
            }
        }

        return results;
    }
}
