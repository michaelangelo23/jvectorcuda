package com.vindex.jvectorcuda.benchmarks;

import com.vindex.jvectorcuda.benchmark.*;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for enhanced benchmarking classes.
 */
class EnhancedBenchmarkTest {

    @Test
    @DisplayName("PercentileMetrics calculates percentiles correctly")
    void testPercentileMetrics() {
        double[] measurements = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
        PercentileMetrics metrics = new PercentileMetrics(measurements);

        assertEquals(5.5, metrics.getP50(), 0.1); // Median
        assertEquals(9.5, metrics.getP95(), 0.1); // 95th percentile
        assertEquals(9.9, metrics.getP99(), 0.1); // 99th percentile
        assertEquals(1.0, metrics.getMin());
        assertEquals(10.0, metrics.getMax());
        assertEquals(5.5, metrics.getMean(), 0.1);
        assertEquals(10, metrics.getSampleCount());
    }

    @Test
    @DisplayName("PercentileMetrics handles single measurement")
    void testPercentileMetricsSingleValue() {
        double[] measurements = { 42.0 };
        PercentileMetrics metrics = new PercentileMetrics(measurements);

        assertEquals(42.0, metrics.getP50());
        assertEquals(42.0, metrics.getP95());
        assertEquals(42.0, metrics.getP99());
        assertEquals(42.0, metrics.getMin());
        assertEquals(42.0, metrics.getMax());
        assertEquals(42.0, metrics.getMean());
    }

    @Test
    @DisplayName("PercentileMetrics rejects invalid input")
    void testPercentileMetricsInvalidInput() {
        assertThrows(IllegalArgumentException.class, () -> new PercentileMetrics(null));
        assertThrows(IllegalArgumentException.class, () -> new PercentileMetrics(new double[0]));
    }

    @Test
    @DisplayName("MemoryMetrics captures memory correctly")
    void testMemoryMetrics() {
        MemoryMetrics metrics = MemoryMetrics.capture(1024 * 1024 * 100); // 100 MB off-heap

        assertTrue(metrics.getHeapUsedBytes() > 0);
        assertTrue(metrics.getHeapMaxBytes() > 0);
        assertEquals(100.0, metrics.getOffHeapUsedMB(), 0.1);
        assertTrue(metrics.getHeapUsagePercent() >= 0 && metrics.getHeapUsagePercent() <= 100);
    }

    @Test
    @DisplayName("StandardBenchmarkSuite creates synthetic datasets")
    void testSyntheticDatasetCreation() {
        StandardBenchmarkSuite.Dataset dataset = StandardBenchmarkSuite.createSyntheticDataset(100, 128, 10, 42L);

        assertNotNull(dataset);
        assertEquals(100, dataset.getVectorCount());
        assertEquals(128, dataset.getDimensions());
        assertEquals(100, dataset.getVectors().length);
        assertEquals(10, dataset.getQueries().length);
        assertEquals(128, dataset.getVectors()[0].length);
        assertEquals(128, dataset.getQueries()[0].length);
    }

    @Test
    @DisplayName("StandardBenchmarkSuite has standard datasets")
    void testStandardDatasets() {
        StandardBenchmarkSuite.Dataset[] datasets = StandardBenchmarkSuite.STANDARD_DATASETS;

        assertNotNull(datasets);
        assertTrue(datasets.length >= 3);

        // Verify datasets are ordered by size
        assertTrue(datasets[0].getVectorCount() < datasets[1].getVectorCount());
        assertTrue(datasets[1].getVectorCount() < datasets[2].getVectorCount());
    }

    @Test
    @DisplayName("BenchmarkResultExporter exports to CSV")
    void testCSVExport(@TempDir Path tempDir) throws IOException {
        Path csvPath = tempDir.resolve("test.csv");

        // Create mock result
        PercentileMetrics latency = new PercentileMetrics(new double[] { 1.0, 2.0, 3.0 });
        MemoryMetrics memory = MemoryMetrics.capture(0);

        StandardBenchmarkSuite.ComprehensiveBenchmarkResult result = StandardBenchmarkSuite.ComprehensiveBenchmarkResult
                .builder()
                .datasetName("Test Dataset")
                .vectorCount(1000)
                .dimensions(384)
                .k(10)
                .throughputQPS(1000.0)
                .latency(latency)
                .memory(memory)
                .buildTimeMs(10.0)
                .recallAt10(1.0)
                .build();

        List<StandardBenchmarkSuite.ComprehensiveBenchmarkResult> results = Arrays.asList(result);

        // Export to CSV
        BenchmarkResultExporter.exportToCSV(results, csvPath);

        // Verify file exists and has content
        assertTrue(Files.exists(csvPath));
        List<String> lines = Files.readAllLines(csvPath);
        assertTrue(lines.size() >= 2); // Header + 1 data row

        // Check header
        String header = lines.get(0);
        assertTrue(header.contains("dataset"));
        assertTrue(header.contains("throughput_qps"));
        assertTrue(header.contains("latency_p50_ms"));

        // Check data row
        String dataRow = lines.get(1);
        assertTrue(dataRow.contains("Test Dataset"));
        assertTrue(dataRow.contains("1000"));
        assertTrue(dataRow.contains("384"));
    }

    @Test
    @DisplayName("BenchmarkResultExporter exports to JSON")
    void testJSONExport(@TempDir Path tempDir) throws IOException {
        Path jsonPath = tempDir.resolve("test.json");

        // Create mock result
        PercentileMetrics latency = new PercentileMetrics(new double[] { 1.0, 2.0, 3.0 });
        MemoryMetrics memory = MemoryMetrics.capture(0);

        StandardBenchmarkSuite.ComprehensiveBenchmarkResult result = StandardBenchmarkSuite.ComprehensiveBenchmarkResult
                .builder()
                .datasetName("Test Dataset")
                .vectorCount(1000)
                .dimensions(384)
                .k(10)
                .throughputQPS(1000.0)
                .latency(latency)
                .memory(memory)
                .buildTimeMs(10.0)
                .recallAt10(1.0)
                .build();

        List<StandardBenchmarkSuite.ComprehensiveBenchmarkResult> results = Arrays.asList(result);

        // Export to JSON
        BenchmarkResultExporter.exportToJSON(results, jsonPath);

        // Verify file exists and has content
        assertTrue(Files.exists(jsonPath));
        String content = Files.readString(jsonPath);

        assertTrue(content.contains("\"benchmark_results\""));
        assertTrue(content.contains("\"dataset\": \"Test Dataset\""));
        assertTrue(content.contains("\"vector_count\": 1000"));
        assertTrue(content.contains("\"throughput_qps\""));
        assertTrue(content.contains("\"latency\""));
        assertTrue(content.contains("\"p50_ms\""));
        assertTrue(content.contains("\"p95_ms\""));
        assertTrue(content.contains("\"p99_ms\""));
    }

    @Test
    @DisplayName("BenchmarkResultExporter appends to CSV")
    void testCSVAppend(@TempDir Path tempDir) throws IOException {
        Path csvPath = tempDir.resolve("test.csv");

        PercentileMetrics latency = new PercentileMetrics(new double[] { 1.0, 2.0, 3.0 });
        MemoryMetrics memory = MemoryMetrics.capture(0);

        StandardBenchmarkSuite.ComprehensiveBenchmarkResult result1 = StandardBenchmarkSuite.ComprehensiveBenchmarkResult
                .builder()
                .datasetName("Dataset 1")
                .vectorCount(1000)
                .dimensions(384)
                .k(10)
                .throughputQPS(1000.0)
                .latency(latency)
                .memory(memory)
                .buildTimeMs(10.0)
                .build();

        StandardBenchmarkSuite.ComprehensiveBenchmarkResult result2 = StandardBenchmarkSuite.ComprehensiveBenchmarkResult
                .builder()
                .datasetName("Dataset 2")
                .vectorCount(2000)
                .dimensions(768)
                .k(10)
                .throughputQPS(2000.0)
                .latency(latency)
                .memory(memory)
                .buildTimeMs(20.0)
                .build();

        // Append first result (creates file with header)
        BenchmarkResultExporter.appendToCSV(result1, csvPath);

        // Append second result (appends without header)
        BenchmarkResultExporter.appendToCSV(result2, csvPath);

        // Verify file has 3 lines (header + 2 data rows)
        List<String> lines = Files.readAllLines(csvPath);
        assertEquals(3, lines.size());

        assertTrue(lines.get(1).contains("Dataset 1"));
        assertTrue(lines.get(2).contains("Dataset 2"));
    }

    @Test
    @DisplayName("ComprehensiveBenchmarkResult builder validates required fields")
    void testComprehensiveBenchmarkResultValidation() {
        // Missing latency should throw
        assertThrows(NullPointerException.class, () -> {
            StandardBenchmarkSuite.ComprehensiveBenchmarkResult.builder()
                    .datasetName("Test")
                    .vectorCount(1000)
                    .dimensions(384)
                    // .latency(latency) // Missing!
                    .memory(MemoryMetrics.capture(0))
                    .build();
        });

        // Missing memory should throw
        assertThrows(NullPointerException.class, () -> {
            StandardBenchmarkSuite.ComprehensiveBenchmarkResult.builder()
                    .datasetName("Test")
                    .vectorCount(1000)
                    .dimensions(384)
                    .latency(new PercentileMetrics(new double[] { 1.0 }))
                    // .memory(memory) // Missing!
                    .build();
        });
    }
}
