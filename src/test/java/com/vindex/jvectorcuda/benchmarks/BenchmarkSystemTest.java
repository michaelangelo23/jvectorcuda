package com.vindex.jvectorcuda.benchmarks;

import com.vindex.jvectorcuda.CudaDetector;
import com.vindex.jvectorcuda.benchmark.*;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for the JVectorCUDA benchmark system.
 * Tests BenchmarkConfig, BenchmarkResult, BenchmarkFramework,
 * PerformanceMetricsLogger,
 * PercentileMetrics, MemoryMetrics, StandardBenchmarkSuite, and
 * BenchmarkResultExporter.
 */
@DisplayName("Benchmark System Tests")
class BenchmarkSystemTest {

    private BenchmarkFramework framework;
    private boolean gpuAvailable;

    @BeforeAll
    static void printSpecs() {
        com.vindex.jvectorcuda.benchmark.BenchmarkRunner.printSystemSpecs();
    }

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
            assertTrue(config.getWarmupIterations() > 0);
        }

        @Test
        @DisplayName("Builder creates valid configuration")
        void builderCreatesValidConfig() {
            BenchmarkConfig config = BenchmarkConfig.builder()
                    .vectorCount(5000)
                    .dimensions(256)
                    .build();
            assertEquals(5000, config.getVectorCount());
        }

        @Test
        @DisplayName("Builder validates minimum vector count")
        void builderValidatesMinVectorCount() {
            assertThrows(IllegalArgumentException.class, () -> BenchmarkConfig.builder().vectorCount(0).build());
        }
    }

    // ==========================================================================
    // BenchmarkResult Tests
    // ==========================================================================

    @Nested
    @DisplayName("BenchmarkResult Tests")
    class BenchmarkResultTests {

        @Test
        @DisplayName("Speedup calculation is correct")
        void speedupCalculationIsCorrect() {
            BenchmarkResult result = createTestResult(100.0, 20.0);
            assertEquals(5.0, result.getSpeedup(), 0.01);
            assertTrue(result.isGpuFaster());
        }

        @Test
        @DisplayName("CSV output is valid format")
        void csvOutputIsValidFormat() {
            BenchmarkResult result = createTestResult(100.0, 20.0);
            String csv = result.toCsvLine();
            assertTrue(csv.split(",").length >= 10);
        }

        private BenchmarkResult createTestResult(double cpuMs, double gpuMs) {
            return BenchmarkResult.builder()
                    .cpuTimeMs(cpuMs)
                    .gpuTimeMs(gpuMs)
                    .gpuTransferTimeMs(5.0)
                    .gpuComputeTimeMs(15.0)
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
            assertEquals(gpuAvailable, framework.isGpuAvailable());
        }

        @Test
        @DisplayName("Run produces valid result")
        void runProducesValidResult() {
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
            assertTrue(result.getCpuTimeMs() > 0);
        }
    }

    // ==========================================================================
    // PerformanceMetricsLogger Tests
    // ==========================================================================

    @Nested
    @DisplayName("PerformanceMetricsLogger Tests")
    class PerformanceMetricsLoggerTests {

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
            PerformanceMetricsLogger logger = PerformanceMetricsLogger.getInstance();
            logger.clear();
            logger.recordOperation("TEST_OP", 100.0);
            assertTrue(logger.getAllStats().containsKey("TEST_OP"));
        }
    }

    // ==========================================================================
    // Enhanced Metrics Tests
    // ==========================================================================

    @Nested
    @DisplayName("PercentileMetrics Tests")
    class PercentileMetricsTests {

        @Test
        @DisplayName("Calculates percentiles correctly")
        void calculatesPercentilesCorrectly() {
            double[] measurements = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
            PercentileMetrics metrics = new PercentileMetrics(measurements);
            assertEquals(5.5, metrics.getP50(), 0.1);
            assertEquals(9.5, metrics.getP95(), 0.1);
            assertEquals(1.0, metrics.getMin());
            assertEquals(10.0, metrics.getMax());
        }

        @Test
        @DisplayName("Rejects invalid input")
        void rejectsInvalidInput() {
            assertThrows(IllegalArgumentException.class, () -> new PercentileMetrics(null));
            assertThrows(IllegalArgumentException.class, () -> new PercentileMetrics(new double[0]));
        }
    }

    @Nested
    @DisplayName("MemoryMetrics Tests")
    class MemoryMetricsTests {

        @Test
        @DisplayName("Captures memory correctly")
        void capturesMemoryCorrectly() {
            MemoryMetrics metrics = MemoryMetrics.capture(1024 * 1024 * 100);
            assertTrue(metrics.getHeapUsedBytes() > 0);
            assertEquals(100.0, metrics.getOffHeapUsedMB(), 0.1);
        }
    }

    @Nested
    @DisplayName("StandardBenchmarkSuite Tests")
    class StandardBenchmarkSuiteTests {

        @Test
        @DisplayName("Creates synthetic datasets")
        void createsSyntheticDatasets() {
            StandardBenchmarkSuite.Dataset dataset = StandardBenchmarkSuite.createSyntheticDataset(100, 128, 10, 42L);
            assertNotNull(dataset);
            assertEquals(100, dataset.getVectorCount());
            assertEquals(128, dataset.getDimensions());
        }
    }

    // ==========================================================================
    // BenchmarkResultExporter Tests
    // ==========================================================================

    @Nested
    @DisplayName("BenchmarkResultExporter Tests")
    class BenchmarkResultExporterTests {

        @Test
        @DisplayName("Exports to CSV")
        void exportsToCsv(@TempDir Path tempDir) throws IOException {
            Path csvPath = tempDir.resolve("test.csv");
            StandardBenchmarkSuite.ComprehensiveBenchmarkResult result = createResult("Test");
            BenchmarkResultExporter.exportToCSV(Arrays.asList(result), csvPath);
            assertTrue(Files.exists(csvPath));
            List<String> lines = Files.readAllLines(csvPath);
            assertTrue(lines.size() >= 2);
        }

        @Test
        @DisplayName("Exports to JSON")
        void exportsToJson(@TempDir Path tempDir) throws IOException {
            Path jsonPath = tempDir.resolve("test.json");
            StandardBenchmarkSuite.ComprehensiveBenchmarkResult result = createResult("Test");
            BenchmarkResultExporter.exportToJSON(Arrays.asList(result), jsonPath);
            assertTrue(Files.exists(jsonPath));
            assertTrue(Files.readString(jsonPath).contains("\"benchmark_results\""));
        }

        @Test
        @DisplayName("Validates required fields")
        void validatesRequiredFields() {
            assertThrows(NullPointerException.class, () -> {
                StandardBenchmarkSuite.ComprehensiveBenchmarkResult.builder()
                        .datasetName("Test")
                        .vectorCount(1000)
                        .dimensions(384)
                        .memory(MemoryMetrics.capture(0))
                        .build();
            });
        }

        private StandardBenchmarkSuite.ComprehensiveBenchmarkResult createResult(String name) {
            return StandardBenchmarkSuite.ComprehensiveBenchmarkResult.builder()
                    .datasetName(name)
                    .vectorCount(1000)
                    .dimensions(384)
                    .k(10)
                    .throughputQPS(1000.0)
                    .latency(new PercentileMetrics(new double[] { 1.0, 2.0, 3.0 }))
                    .memory(MemoryMetrics.capture(0))
                    .buildTimeMs(10.0)
                    .recallAt10(1.0)
                    .build();
        }
    }
}
