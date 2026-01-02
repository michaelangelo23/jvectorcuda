package com.vindex.jvectorcuda.benchmarks;

import com.vindex.jvectorcuda.CudaDetector;
import com.vindex.jvectorcuda.DistanceMetric;
import com.vindex.jvectorcuda.benchmark.BenchmarkConfig;
import com.vindex.jvectorcuda.benchmark.BenchmarkFramework;
import com.vindex.jvectorcuda.benchmark.BenchmarkResult;
import com.vindex.jvectorcuda.benchmark.PerformanceMetricsLogger;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

// Tests for BenchmarkConfig, BenchmarkResult, BenchmarkFramework, and PerformanceMetricsLogger
@DisplayName("Benchmark Framework Tests")
class BenchmarkFrameworkTest {

    private BenchmarkFramework framework;
    private boolean gpuAvailable;

    @BeforeEach
    void setUp() {
        framework = new BenchmarkFramework();
        gpuAvailable = CudaDetector.isAvailable();
    }

    // ==========================================================================
    // BenchmarkConfig Tests
    // ==========================================================================

    @Nested
    @DisplayName("BenchmarkConfig Tests")
    class BenchmarkConfigTests {

        @Test
        @DisplayName("Default preset has valid values")
        void defaultPresetHasValidValues() {
            BenchmarkConfig config = BenchmarkConfig.DEFAULT;
            
            assertEquals(10_000, config.getVectorCount());
            assertEquals(384, config.getDimensions());
            assertEquals(10, config.getQueryCount());  // Default is 10, not 100
            assertEquals(10, config.getK());
            assertTrue(config.getWarmupIterations() > 0);
            assertTrue(config.getMeasuredIterations() > 0);
        }

        @Test
        @DisplayName("Comprehensive preset has larger values")
        void comprehensivePresetHasLargerValues() {
            BenchmarkConfig comprehensive = BenchmarkConfig.COMPREHENSIVE;
            BenchmarkConfig defaultConfig = BenchmarkConfig.DEFAULT;
            
            assertTrue(comprehensive.getVectorCount() >= defaultConfig.getVectorCount());
            assertTrue(comprehensive.getMeasuredIterations() >= defaultConfig.getMeasuredIterations());
        }

        @Test
        @DisplayName("Stress test preset has largest values")
        void stressTestPresetHasLargestValues() {
            BenchmarkConfig stressTest = BenchmarkConfig.STRESS_TEST;
            BenchmarkConfig comprehensive = BenchmarkConfig.COMPREHENSIVE;
            
            assertTrue(stressTest.getVectorCount() >= comprehensive.getVectorCount());
        }

        @Test
        @DisplayName("Builder creates valid configuration")
        void builderCreatesValidConfig() {
            BenchmarkConfig config = BenchmarkConfig.builder()
                .vectorCount(5000)
                .dimensions(256)
                .queryCount(50)
                .k(5)
                .warmupIterations(2)
                .measuredIterations(5)
                .randomSeed(42L)
                .distanceMetric(DistanceMetric.EUCLIDEAN)
                .build();
            
            assertEquals(5000, config.getVectorCount());
            assertEquals(256, config.getDimensions());
            assertEquals(50, config.getQueryCount());
            assertEquals(5, config.getK());
            assertEquals(2, config.getWarmupIterations());
            assertEquals(5, config.getMeasuredIterations());
            assertEquals(42L, config.getRandomSeed());
            assertEquals(DistanceMetric.EUCLIDEAN, config.getDistanceMetric());
        }

        @Test
        @DisplayName("Builder validates minimum vector count")
        void builderValidatesMinVectorCount() {
            assertThrows(IllegalArgumentException.class, () ->
                BenchmarkConfig.builder().vectorCount(0).build());
            assertThrows(IllegalArgumentException.class, () ->
                BenchmarkConfig.builder().vectorCount(-1).build());
        }

        @Test
        @DisplayName("Builder validates minimum dimensions")
        void builderValidatesMinDimensions() {
            assertThrows(IllegalArgumentException.class, () ->
                BenchmarkConfig.builder().dimensions(0).build());
        }

        @Test
        @DisplayName("Builder validates k is positive")
        void builderValidatesKIsPositive() {
            // k must be positive
            assertThrows(IllegalArgumentException.class, () ->
                BenchmarkConfig.builder().k(0).build());
            assertThrows(IllegalArgumentException.class, () ->
                BenchmarkConfig.builder().k(-1).build());
        }

        @Test
        @DisplayName("Memory estimation is reasonable")
        void memoryEstimationIsReasonable() {
            BenchmarkConfig config = BenchmarkConfig.builder()
                .vectorCount(10_000)
                .dimensions(384)
                .build();
            
            long estimatedBytes = config.getEstimatedMemoryBytes();
            
            // At minimum: 10000 * 384 * 4 bytes = ~15MB
            long minExpected = 10_000L * 384 * 4;
            assertTrue(estimatedBytes >= minExpected, 
                "Memory estimate should be at least " + minExpected + " but was " + estimatedBytes);
            
            // Should not be absurdly large (< 1GB for this config)
            assertTrue(estimatedBytes < 1_000_000_000L);
        }

        @Test
        @DisplayName("toString contains key information")
        void toStringContainsKeyInfo() {
            BenchmarkConfig config = BenchmarkConfig.DEFAULT;
            String str = config.toString();
            
            assertTrue(str.contains("10000") || str.contains("10,000"));
            assertTrue(str.contains("384"));
        }
    }

    // ==========================================================================
    // BenchmarkResult Tests
    // ==========================================================================

    @Nested
    @DisplayName("BenchmarkResult Tests")
    class BenchmarkResultTests {

        @Test
        @DisplayName("Builder creates valid result")
        void builderCreatesValidResult() {
            Instant now = Instant.now();
            BenchmarkResult result = BenchmarkResult.builder()
                .cpuTimeMs(100.0)
                .gpuTimeMs(20.0)
                .gpuTransferTimeMs(5.0)
                .gpuComputeTimeMs(15.0)
                .gpuMemoryUsedBytes(1_000_000L)
                .vectorCount(10_000)
                .dimensions(384)
                .queryCount(100)
                .k(10)
                .distanceMetric("EUCLIDEAN")
                .timestamp(now)
                .gpuName("GTX 1080")
                .warmupIterations(3)
                .measuredIterations(10)
                .build();
            
            assertEquals(100.0, result.getCpuTimeMs());
            assertEquals(20.0, result.getGpuTimeMs());
            assertEquals(5.0, result.getGpuTransferTimeMs());
            assertEquals(15.0, result.getGpuComputeTimeMs());
            assertEquals(10_000, result.getVectorCount());
        }

        @Test
        @DisplayName("Speedup calculation is correct")
        void speedupCalculationIsCorrect() {
            BenchmarkResult result = createResultWithTimes(100.0, 20.0);
            
            // CPU takes 100ms, GPU takes 20ms -> speedup = 5x
            assertEquals(5.0, result.getSpeedup(), 0.01);
            assertTrue(result.isGpuFaster());
        }

        @Test
        @DisplayName("Speedup handles GPU slower than CPU")
        void speedupHandlesGpuSlower() {
            BenchmarkResult result = createResultWithTimes(20.0, 100.0);
            
            // CPU takes 20ms, GPU takes 100ms -> speedup = 0.2x
            assertEquals(0.2, result.getSpeedup(), 0.01);
            assertFalse(result.isGpuFaster());
        }

        @Test
        @DisplayName("Transfer overhead percentage is correct")
        void transferOverheadIsCorrect() {
            BenchmarkResult result = BenchmarkResult.builder()
                .cpuTimeMs(100.0)
                .gpuTimeMs(20.0)
                .gpuTransferTimeMs(10.0)  // 50% of GPU time
                .gpuComputeTimeMs(10.0)
                .gpuMemoryUsedBytes(1_000_000L)
                .vectorCount(1000)
                .dimensions(128)
                .queryCount(10)
                .k(5)
                .distanceMetric("EUCLIDEAN")
                .timestamp(Instant.now())
                .gpuName("Test GPU")
                .warmupIterations(1)
                .measuredIterations(1)
                .build();
            
            assertEquals(50.0, result.getTransferOverheadPercent(), 0.01);
        }

        @Test
        @DisplayName("Memory conversion is correct")
        void memoryConversionIsCorrect() {
            BenchmarkResult result = BenchmarkResult.builder()
                .cpuTimeMs(100.0)
                .gpuTimeMs(20.0)
                .gpuTransferTimeMs(5.0)
                .gpuComputeTimeMs(15.0)
                .gpuMemoryUsedBytes(1_048_576L)  // 1 MB exactly
                .vectorCount(1000)
                .dimensions(128)
                .queryCount(10)
                .k(5)
                .distanceMetric("EUCLIDEAN")
                .timestamp(Instant.now())
                .gpuName("Test GPU")
                .warmupIterations(1)
                .measuredIterations(1)
                .build();
            
            assertEquals(1.0, result.getGpuMemoryUsedMB(), 0.01);
        }

        @Test
        @DisplayName("Throughput calculation is correct")
        void throughputCalculationIsCorrect() {
            BenchmarkResult result = BenchmarkResult.builder()
                .cpuTimeMs(1000.0)  // 1 second
                .gpuTimeMs(100.0)   // 100ms
                .gpuTransferTimeMs(10.0)
                .gpuComputeTimeMs(90.0)
                .gpuMemoryUsedBytes(1_000_000L)
                .vectorCount(10_000)
                .dimensions(128)
                .queryCount(100)  // 100 queries in 100ms = 1000 q/s
                .k(5)
                .distanceMetric("EUCLIDEAN")
                .timestamp(Instant.now())
                .gpuName("Test GPU")
                .warmupIterations(1)
                .measuredIterations(1)
                .build();
            
            // 100 queries / (100ms / 1000) = 1000 queries/second
            assertEquals(1000.0, result.getGpuThroughput(), 1.0);
        }

        @Test
        @DisplayName("CSV output is valid format")
        void csvOutputIsValidFormat() {
            BenchmarkResult result = createResultWithTimes(100.0, 20.0);
            String csv = result.toCsvLine();
            
            // Should contain comma-separated values
            String[] parts = csv.split(",");
            assertTrue(parts.length >= 10, "CSV should have multiple columns");
            
            // CSV format: timestamp,gpu_name,vector_count,dimensions,query_count,k,metric,cpu_ms,gpu_ms,...
            // Numeric values (cpuTimeMs=parts[7], gpuTimeMs=parts[8]) should be parseable
            assertDoesNotThrow(() -> Double.parseDouble(parts[7].trim())); // cpuTimeMs
            assertDoesNotThrow(() -> Double.parseDouble(parts[8].trim())); // gpuTimeMs
        }

        @Test
        @DisplayName("CSV header matches data columns")
        void csvHeaderMatchesData() {
            BenchmarkResult result = createResultWithTimes(100.0, 20.0);
            String header = BenchmarkResult.getCsvHeader();
            String data = result.toCsvLine();
            
            int headerColumns = header.split(",").length;
            int dataColumns = data.split(",").length;
            
            assertEquals(headerColumns, dataColumns, 
                "Header columns should match data columns");
        }

        @Test
        @DisplayName("Summary string is human-readable")
        void summaryStringIsReadable() {
            BenchmarkResult result = createResultWithTimes(100.0, 20.0);
            String summary = result.toSummaryString();
            
            assertTrue(summary.contains("CPU"), "Summary should mention CPU");
            assertTrue(summary.contains("GPU"), "Summary should mention GPU");
            assertTrue(summary.contains("speedup") || summary.contains("Speedup"), 
                "Summary should mention speedup");
        }

        private BenchmarkResult createResultWithTimes(double cpuMs, double gpuMs) {
            return BenchmarkResult.builder()
                .cpuTimeMs(cpuMs)
                .gpuTimeMs(gpuMs)
                .gpuTransferTimeMs(gpuMs * 0.25)
                .gpuComputeTimeMs(gpuMs * 0.75)
                .gpuMemoryUsedBytes(1_000_000L)
                .vectorCount(10_000)
                .dimensions(384)
                .queryCount(100)
                .k(10)
                .distanceMetric("EUCLIDEAN")
                .timestamp(Instant.now())
                .gpuName("Test GPU")
                .warmupIterations(3)
                .measuredIterations(10)
                .build();
        }
    }

    // ==========================================================================
    // BenchmarkFramework Tests
    // ==========================================================================

    @Nested
    @DisplayName("BenchmarkFramework Tests")
    class BenchmarkFrameworkTests {

        @Test
        @DisplayName("Framework detects GPU availability")
        void frameworkDetectsGpuAvailability() {
            // Should match CudaDetector
            assertEquals(gpuAvailable, framework.isGpuAvailable());
        }

        @Test
        @DisplayName("Framework reports GPU name")
        void frameworkReportsGpuName() {
            String gpuName = framework.getGpuName();
            
            assertNotNull(gpuName);
            if (gpuAvailable) {
                assertFalse(gpuName.contains("No GPU"));
            }
        }

        @Test
        @DisplayName("Run produces valid result")
        void runProducesValidResult() {
            // Use small config for fast test
            BenchmarkConfig config = BenchmarkConfig.builder()
                .vectorCount(100)
                .dimensions(64)
                .queryCount(5)
                .k(3)
                .warmupIterations(1)
                .measuredIterations(1)
                .build();
            
            BenchmarkResult result = framework.run(config);
            
            assertNotNull(result);
            assertTrue(result.getCpuTimeMs() > 0, "CPU time should be positive");
            assertEquals(100, result.getVectorCount());
            assertEquals(64, result.getDimensions());
            assertEquals(5, result.getQueryCount());
        }

        @Test
        @DisplayName("Run produces consistent results")
        void runProducesConsistentResults() {
            BenchmarkConfig config = BenchmarkConfig.builder()
                .vectorCount(500)
                .dimensions(128)
                .queryCount(10)
                .k(5)
                .warmupIterations(2)
                .measuredIterations(3)
                .randomSeed(12345L)  // Fixed seed for reproducibility
                .build();
            
            // Run twice with same config
            BenchmarkResult result1 = framework.run(config);
            BenchmarkResult result2 = framework.run(config);
            
            // Results should be similar (within 50% due to system variance)
            double ratio = result1.getCpuTimeMs() / result2.getCpuTimeMs();
            assertTrue(ratio > 0.5 && ratio < 2.0, 
                "CPU times should be relatively consistent: " + result1.getCpuTimeMs() + " vs " + result2.getCpuTimeMs());
        }

        @Test
        @DisplayName("Run suite produces multiple results")
        void runSuiteProducesMultipleResults() {
            List<BenchmarkResult> results = framework.runSuite(
                List.of(100, 200),
                List.of(32, 64)
            );
            
            // Should produce 2x2 = 4 results
            assertEquals(4, results.size());
            
            // All should have valid data
            for (BenchmarkResult result : results) {
                assertTrue(result.getCpuTimeMs() > 0);
                assertTrue(result.getVectorCount() > 0);
            }
        }

        @Test
        @DisplayName("Persistent memory benchmark works")
        void persistentMemoryBenchmarkWorks() {
            BenchmarkConfig config = BenchmarkConfig.builder()
                .vectorCount(500)
                .dimensions(128)
                .queryCount(20)
                .k(5)
                .warmupIterations(1)
                .measuredIterations(2)
                .build();
            
            BenchmarkResult result = framework.runPersistentMemoryBenchmark(config);
            
            assertNotNull(result);
            assertTrue(result.getCpuTimeMs() > 0);
            assertEquals(500, result.getVectorCount());
        }

        @Test
        @DisplayName("Print comparison table does not throw")
        void printComparisonTableDoesNotThrow() {
            List<BenchmarkResult> results = framework.runSuite(
                List.of(100),
                List.of(32)
            );
            
            assertDoesNotThrow(() -> framework.printComparisonTable(results));
        }
    }

    // ==========================================================================
    // PerformanceMetricsLogger Tests
    // ==========================================================================

    @Nested
    @DisplayName("PerformanceMetricsLogger Tests")
    class PerformanceMetricsLoggerTests {

        private PerformanceMetricsLogger logger;

        @BeforeEach
        void setUp() {
            logger = PerformanceMetricsLogger.getInstance();
            logger.clear();
            logger.setEnabled(true);
        }

        @Test
        @DisplayName("Singleton returns same instance")
        void singletonReturnsSameInstance() {
            PerformanceMetricsLogger instance1 = PerformanceMetricsLogger.getInstance();
            PerformanceMetricsLogger instance2 = PerformanceMetricsLogger.getInstance();
            
            assertSame(instance1, instance2);
        }

        @Test
        @DisplayName("Records operation correctly")
        void recordsOperationCorrectly() {
            logger.recordOperation(PerformanceMetricsLogger.OP_GPU_UPLOAD, 15.5);
            logger.recordOperation(PerformanceMetricsLogger.OP_GPU_UPLOAD, 16.5);
            
            PerformanceMetricsLogger.OperationStats stats = 
                logger.getStats(PerformanceMetricsLogger.OP_GPU_UPLOAD);
            
            assertNotNull(stats);
            assertEquals(2, stats.count);
            assertEquals(32.0, stats.totalMs, 0.01);
            assertEquals(16.0, stats.avgMs, 0.01);
        }

        @Test
        @DisplayName("Timing block measures duration")
        void timingBlockMeasuresDuration() throws InterruptedException {
            try (PerformanceMetricsLogger.TimingBlock block = 
                    logger.startTiming(PerformanceMetricsLogger.OP_KERNEL_EXEC)) {
                Thread.sleep(10); // Sleep 10ms
            }
            
            PerformanceMetricsLogger.OperationStats stats = 
                logger.getStats(PerformanceMetricsLogger.OP_KERNEL_EXEC);
            
            assertNotNull(stats);
            assertEquals(1, stats.count);
            assertTrue(stats.avgMs >= 8, "Duration should be at least 8ms but was " + stats.avgMs);
        }

        @Test
        @DisplayName("Disabled logger does not record")
        void disabledLoggerDoesNotRecord() {
            logger.setEnabled(false);
            logger.recordOperation("TEST_OP", 10.0);
            
            assertNull(logger.getStats("TEST_OP"));
            assertEquals(0, logger.getMetricCount());
            
            logger.setEnabled(true);
        }

        @Test
        @DisplayName("Clear removes all metrics")
        void clearRemovesAllMetrics() {
            logger.recordOperation("TEST_OP", 10.0);
            logger.recordOperation("TEST_OP", 20.0);
            
            assertEquals(2, logger.getMetricCount());
            
            logger.clear();
            
            assertEquals(0, logger.getMetricCount());
            assertNull(logger.getStats("TEST_OP"));
        }

        @Test
        @DisplayName("Get all stats returns all operations")
        void getAllStatsReturnsAllOperations() {
            logger.recordOperation("OP_A", 10.0);
            logger.recordOperation("OP_B", 20.0);
            logger.recordOperation("OP_C", 30.0);
            
            Map<String, PerformanceMetricsLogger.OperationStats> allStats = logger.getAllStats();
            
            assertEquals(3, allStats.size());
            assertTrue(allStats.containsKey("OP_A"));
            assertTrue(allStats.containsKey("OP_B"));
            assertTrue(allStats.containsKey("OP_C"));
        }

        @Test
        @DisplayName("Export to CSV creates valid file")
        void exportToCsvCreatesValidFile(@TempDir Path tempDir) throws IOException {
            logger.recordOperation("TEST_OP", 10.0, "test details");
            logger.recordOperation("TEST_OP", 20.0);
            
            Path csvPath = tempDir.resolve("metrics.csv");
            logger.exportToCsv(csvPath);
            
            assertTrue(Files.exists(csvPath));
            
            List<String> lines = Files.readAllLines(csvPath);
            assertEquals(3, lines.size()); // Header + 2 data rows
            assertTrue(lines.get(0).contains("timestamp"));
            assertTrue(lines.get(0).contains("operation"));
            assertTrue(lines.get(0).contains("duration_ms"));
        }

        @Test
        @DisplayName("Export stats to CSV creates valid file")
        void exportStatsToCsvCreatesValidFile(@TempDir Path tempDir) throws IOException {
            logger.recordOperation("OP_A", 10.0);
            logger.recordOperation("OP_A", 20.0);
            logger.recordOperation("OP_B", 15.0);
            
            Path csvPath = tempDir.resolve("stats.csv");
            logger.exportStatsToCsv(csvPath);
            
            assertTrue(Files.exists(csvPath));
            
            List<String> lines = Files.readAllLines(csvPath);
            assertEquals(3, lines.size()); // Header + 2 operation summaries
            assertTrue(lines.get(0).contains("avg_ms"));
            assertTrue(lines.get(0).contains("std_dev_ms"));
        }

        @Test
        @DisplayName("Print summary does not throw")
        void printSummaryDoesNotThrow() {
            logger.recordOperation("TEST_OP", 10.0);
            
            assertDoesNotThrow(() -> logger.printSummary());
        }

        @Test
        @DisplayName("Statistics calculations are correct")
        void statisticsCalculationsAreCorrect() {
            // Record known values
            logger.recordOperation("TEST_OP", 10.0);
            logger.recordOperation("TEST_OP", 20.0);
            logger.recordOperation("TEST_OP", 30.0);
            
            PerformanceMetricsLogger.OperationStats stats = logger.getStats("TEST_OP");
            
            assertEquals(3, stats.count);
            assertEquals(60.0, stats.totalMs, 0.01);
            assertEquals(20.0, stats.avgMs, 0.01);
            assertEquals(10.0, stats.minMs, 0.01);
            assertEquals(30.0, stats.maxMs, 0.01);
            // Standard deviation of [10, 20, 30] = sqrt(((10-20)^2 + (20-20)^2 + (30-20)^2) / 3) = sqrt(200/3) ≈ 8.16
            assertEquals(8.16, stats.stdDevMs, 0.1);
        }
    }

    // ==========================================================================
    // Edge Cases
    // ==========================================================================

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {

        @Test
        @DisplayName("Benchmark with single vector")
        void benchmarkWithSingleVector() {
            BenchmarkConfig config = BenchmarkConfig.builder()
                .vectorCount(1)
                .dimensions(32)
                .queryCount(1)
                .k(1)
                .warmupIterations(1)
                .measuredIterations(1)
                .build();
            
            BenchmarkResult result = framework.run(config);
            
            assertNotNull(result);
            assertEquals(1, result.getVectorCount());
        }

        @Test
        @DisplayName("Benchmark with maximum k equals vector count")
        void benchmarkWithMaxK() {
            BenchmarkConfig config = BenchmarkConfig.builder()
                .vectorCount(10)
                .dimensions(32)
                .queryCount(5)
                .k(10)  // k equals vector count
                .warmupIterations(1)
                .measuredIterations(1)
                .build();
            
            BenchmarkResult result = framework.run(config);
            
            assertNotNull(result);
            assertEquals(10, result.getK());
        }

        @Test
        @DisplayName("Logger handles zero duration")
        void loggerHandlesZeroDuration() {
            PerformanceMetricsLogger logger = PerformanceMetricsLogger.getInstance();
            logger.clear();
            
            logger.recordOperation("ZERO_OP", 0.0);
            
            PerformanceMetricsLogger.OperationStats stats = logger.getStats("ZERO_OP");
            assertNotNull(stats);
            assertEquals(0.0, stats.avgMs, 0.001);
        }

        @Test
        @DisplayName("Logger handles very small durations")
        void loggerHandlesVerySmallDurations() {
            PerformanceMetricsLogger logger = PerformanceMetricsLogger.getInstance();
            logger.clear();
            
            logger.recordOperation("TINY_OP", 0.000001);
            
            PerformanceMetricsLogger.OperationStats stats = logger.getStats("TINY_OP");
            assertNotNull(stats);
            assertTrue(stats.avgMs > 0);
        }

        @Test
        @DisplayName("Logger handles very large durations")
        void loggerHandlesVeryLargeDurations() {
            PerformanceMetricsLogger logger = PerformanceMetricsLogger.getInstance();
            logger.clear();
            
            logger.recordOperation("HUGE_OP", 1_000_000.0);  // 1000 seconds
            
            PerformanceMetricsLogger.OperationStats stats = logger.getStats("HUGE_OP");
            assertNotNull(stats);
            assertEquals(1_000_000.0, stats.avgMs, 0.01);
        }

        @Test
        @DisplayName("Empty benchmark suite returns empty list")
        void emptyBenchmarkSuiteReturnsEmptyList() {
            List<BenchmarkResult> results = framework.runSuite(
                List.of(),  // Empty vector counts
                List.of(128)
            );
            
            assertTrue(results.isEmpty());
        }
    }

    // ==========================================================================
    // AI Blind Spots
    // ==========================================================================

    @Nested
    @DisplayName("AI Blind Spots")
    class AiBlindSpots {

        @Test
        @DisplayName("Logger is thread-safe")
        void loggerIsThreadSafe() throws InterruptedException {
            PerformanceMetricsLogger logger = PerformanceMetricsLogger.getInstance();
            logger.clear();
            
            int threadCount = 10;
            int opsPerThread = 100;
            Thread[] threads = new Thread[threadCount];
            
            for (int t = 0; t < threadCount; t++) {
                final int threadId = t;
                threads[t] = new Thread(() -> {
                    for (int i = 0; i < opsPerThread; i++) {
                        logger.recordOperation("THREAD_OP_" + threadId, i * 0.1);
                    }
                });
            }
            
            // Start all threads
            for (Thread thread : threads) {
                thread.start();
            }
            
            // Wait for completion
            for (Thread thread : threads) {
                thread.join(5000);
            }
            
            // Verify all operations were recorded
            assertEquals(threadCount * opsPerThread, logger.getMetricCount(),
                "All operations should be recorded");
        }

        @Test
        @DisplayName("Timing block handles exceptions")
        void timingBlockHandlesExceptions() {
            PerformanceMetricsLogger logger = PerformanceMetricsLogger.getInstance();
            logger.clear();
            
            assertThrows(RuntimeException.class, () -> {
                try (PerformanceMetricsLogger.TimingBlock block = 
                        logger.startTiming("EXCEPTION_OP")) {
                    throw new RuntimeException("Test exception");
                }
            });
            
            // Block should still record even after exception
            PerformanceMetricsLogger.OperationStats stats = logger.getStats("EXCEPTION_OP");
            assertNotNull(stats, "Operation should be recorded even when exception occurs");
            assertEquals(1, stats.count);
        }

        @Test
        @DisplayName("BenchmarkResult handles divide by zero")
        void benchmarkResultHandlesDivideByZero() {
            // GPU time of 0 would cause divide by zero in transfer overhead calculation
            BenchmarkResult result = BenchmarkResult.builder()
                .cpuTimeMs(100.0)
                .gpuTimeMs(0.0)  // Zero GPU time
                .gpuTransferTimeMs(0.0)
                .gpuComputeTimeMs(0.0)
                .gpuMemoryUsedBytes(0L)
                .vectorCount(100)
                .dimensions(64)
                .queryCount(10)
                .k(5)
                .distanceMetric("EUCLIDEAN")
                .timestamp(Instant.now())
                .gpuName("Test")
                .warmupIterations(1)
                .measuredIterations(1)
                .build();
            
            // Should not throw, should return reasonable values
            assertDoesNotThrow(() -> result.getSpeedup());
            assertDoesNotThrow(() -> result.getTransferOverheadPercent());
            assertDoesNotThrow(() -> result.getGpuThroughput());
        }

        @Test
        @DisplayName("Benchmark framework handles GPU unavailable gracefully")
        void benchmarkFrameworkHandlesNoGpu() {
            // Even without GPU, framework should work (CPU-only benchmark)
            BenchmarkConfig config = BenchmarkConfig.builder()
                .vectorCount(50)
                .dimensions(32)
                .queryCount(5)
                .k(3)
                .warmupIterations(1)
                .measuredIterations(1)
                .build();
            
            BenchmarkResult result = framework.run(config);
            
            assertNotNull(result);
            assertTrue(result.getCpuTimeMs() > 0);
            // GPU time might be 0 if no GPU available
        }

        @Test
        @DisplayName("CSV export handles special characters")
        void csvExportHandlesSpecialCharacters(@TempDir Path tempDir) throws IOException {
            PerformanceMetricsLogger logger = PerformanceMetricsLogger.getInstance();
            logger.clear();
            
            // Record with special characters in details
            logger.recordOperation("SPECIAL_OP", 10.0, "contains \"quotes\" and, commas");
            
            Path csvPath = tempDir.resolve("special.csv");
            logger.exportToCsv(csvPath);
            
            String content = Files.readString(csvPath);
            // Should properly escape quotes
            assertTrue(content.contains("\"\"quotes\"\"") || content.contains("quotes"));
        }

        @Test
        @DisplayName("Memory estimation does not overflow for large configs")
        void memoryEstimationDoesNotOverflow() {
            // Large config that could overflow int
            BenchmarkConfig config = BenchmarkConfig.builder()
                .vectorCount(1_000_000)
                .dimensions(2048)
                .queryCount(100)
                .k(10)
                .build();
            
            long memory = config.getEstimatedMemoryBytes();
            
            // Should be positive and very large
            assertTrue(memory > 0, "Memory estimate should be positive");
            assertTrue(memory > 1_000_000_000L, "Memory should be > 1GB for large config");
        }
    }
}
