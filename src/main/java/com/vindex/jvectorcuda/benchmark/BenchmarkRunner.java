package com.vindex.jvectorcuda.benchmark;

import com.vindex.jvectorcuda.CudaDetector;
import com.vindex.jvectorcuda.DistanceMetric;
import com.vindex.jvectorcuda.VectorIndex;
import com.vindex.jvectorcuda.VectorIndexFactory;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.lang.management.ManagementFactory;
import java.lang.management.OperatingSystemMXBean;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Random;

/**
 * Standalone benchmark runner for community GPU testing.
 * 
 * Run with: ./gradlew runBenchmark
 * Or: java -cp build/libs/jvectorcuda-1.0.0.jar com.vindex.jvectorcuda.benchmark.BenchmarkRunner
 * 
 * Outputs a markdown-formatted report ready to paste into a GitHub Issue.
 */
public class BenchmarkRunner {

    private static final int WARMUP_ITERATIONS = 3;
    private static final int MEASURED_ITERATIONS = 5;
    private static final int DIMENSIONS = 384;
    private static final int[] VECTOR_COUNTS = {1_000, 10_000, 50_000};
    private static final int[] QUERY_COUNTS = {1, 10, 100};

    public static void main(String[] args) {
        System.out.println("JVectorCUDA Benchmark Runner");
        System.out.println("============================\n");

        BenchmarkRunner runner = new BenchmarkRunner();
        String report = runner.runFullBenchmark();

        System.out.println(report);

        // Also save to file
        try (PrintWriter writer = new PrintWriter(new FileWriter("benchmark-report.md"))) {
            writer.print(report);
            System.out.println("\nReport saved to: benchmark-report.md");
        } catch (Exception e) {
            System.err.println("Could not save report file: " + e.getMessage());
        }
    }

    public String runFullBenchmark() {
        StringBuilder report = new StringBuilder();

        // Header
        report.append("# GPU Benchmark Results\n\n");
        report.append("Generated: ").append(LocalDateTime.now().format(
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"))).append("\n\n");

        // System info
        report.append("## System Information\n\n");
        report.append(getSystemInfo());
        report.append("\n");

        // Check GPU availability
        boolean gpuAvailable = CudaDetector.isAvailable();
        if (!gpuAvailable) {
            report.append("**WARNING: No CUDA-capable GPU detected. CPU-only results below.**\n\n");
        }

        // Run benchmarks
        report.append("## Benchmark Results\n\n");

        // Single query benchmarks (tests JNI overhead)
        report.append("### Single Query Performance\n\n");
        report.append("| Vectors | CPU (ms) | GPU (ms) | Speedup | Winner |\n");
        report.append("|---------|----------|----------|---------|--------|\n");

        for (int vectorCount : VECTOR_COUNTS) {
            BenchmarkResult result = runSingleQueryBenchmark(vectorCount);
            report.append(formatResultRow(result));
        }
        report.append("\n");

        // Persistent memory benchmarks (tests amortized transfer)
        report.append("### Persistent Memory (Many Queries, Same Dataset)\n\n");
        report.append("| Vectors | Queries | CPU (ms) | GPU (ms) | Speedup | Winner |\n");
        report.append("|---------|---------|----------|----------|---------|--------|\n");

        for (int vectorCount : VECTOR_COUNTS) {
            for (int queryCount : QUERY_COUNTS) {
                if (queryCount == 1) continue; // Skip single query (covered above)
                BenchmarkResult result = runPersistentMemoryBenchmark(vectorCount, queryCount);
                report.append(formatResultRowWithQueries(result));
            }
        }
        report.append("\n");

        // Transfer overhead
        report.append("### Memory Transfer Analysis\n\n");
        TransferAnalysis transfer = measureTransferOverhead(50_000);
        report.append(String.format("- Upload time (50K vectors × %dD): %.1f ms\n", DIMENSIONS, transfer.uploadMs));
        report.append(String.format("- Search time (single query): %.1f ms\n", transfer.searchMs));
        report.append(String.format("- Transfer overhead: %.0f%%\n", transfer.overheadPercent));
        report.append("\n");

        // Summary
        report.append("## Summary\n\n");
        report.append(generateSummary());

        return report.toString();
    }

    private String getSystemInfo() {
        StringBuilder info = new StringBuilder();

        // GPU
        if (CudaDetector.isAvailable()) {
            info.append("- **GPU:** ").append(CudaDetector.getGpuInfo()).append("\n");
        } else {
            info.append("- **GPU:** Not detected\n");
        }

        // CPU
        OperatingSystemMXBean os = ManagementFactory.getOperatingSystemMXBean();
        info.append("- **CPU:** ").append(System.getProperty("os.arch"))
            .append(" (").append(os.getAvailableProcessors()).append(" cores)\n");

        // Memory
        long maxMemory = Runtime.getRuntime().maxMemory() / (1024 * 1024);
        info.append("- **JVM Memory:** ").append(maxMemory).append(" MB\n");

        // OS
        info.append("- **OS:** ").append(System.getProperty("os.name"))
            .append(" ").append(System.getProperty("os.version")).append("\n");

        // Java
        info.append("- **Java:** ").append(System.getProperty("java.version")).append("\n");

        return info.toString();
    }

    private BenchmarkResult runSingleQueryBenchmark(int vectorCount) {
        float[][] database = generateRandomVectors(vectorCount, DIMENSIONS);
        float[] query = generateRandomVectors(1, DIMENSIONS)[0];

        // CPU benchmark
        double cpuTime = benchmarkCpu(database, query, 1);

        // GPU benchmark
        double gpuTime = benchmarkGpu(database, query, 1);

        return new BenchmarkResult(vectorCount, 1, cpuTime, gpuTime);
    }

    private BenchmarkResult runPersistentMemoryBenchmark(int vectorCount, int queryCount) {
        float[][] database = generateRandomVectors(vectorCount, DIMENSIONS);
        float[][] queries = generateRandomVectors(queryCount, DIMENSIONS);

        // CPU benchmark - create index once, run all queries
        double cpuTime = benchmarkCpuPersistent(database, queries);

        // GPU benchmark - upload once, run all queries
        double gpuTime = benchmarkGpuPersistent(database, queries);

        return new BenchmarkResult(vectorCount, queryCount, cpuTime, gpuTime);
    }

    private double benchmarkCpu(float[][] database, float[] query, int iterations) {
        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            try (VectorIndex index = VectorIndexFactory.cpu(DIMENSIONS)) {
                index.add(database);
                index.search(query, 10);
            }
        }

        // Measure
        long start = System.nanoTime();
        for (int i = 0; i < MEASURED_ITERATIONS; i++) {
            try (VectorIndex index = VectorIndexFactory.cpu(DIMENSIONS)) {
                index.add(database);
                index.search(query, 10);
            }
        }
        return (System.nanoTime() - start) / 1_000_000.0 / MEASURED_ITERATIONS;
    }

    private double benchmarkGpu(float[][] database, float[] query, int iterations) {
        if (!CudaDetector.isAvailable()) return 0;

        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            try (VectorIndex index = VectorIndexFactory.gpu(DIMENSIONS, DistanceMetric.EUCLIDEAN)) {
                index.add(database);
                index.search(query, 10);
            }
        }

        // Measure
        long start = System.nanoTime();
        for (int i = 0; i < MEASURED_ITERATIONS; i++) {
            try (VectorIndex index = VectorIndexFactory.gpu(DIMENSIONS, DistanceMetric.EUCLIDEAN)) {
                index.add(database);
                index.search(query, 10);
            }
        }
        return (System.nanoTime() - start) / 1_000_000.0 / MEASURED_ITERATIONS;
    }

    private double benchmarkCpuPersistent(float[][] database, float[][] queries) {
        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            try (VectorIndex index = VectorIndexFactory.cpu(DIMENSIONS)) {
                index.add(database);
                for (float[] q : queries) index.search(q, 10);
            }
        }

        // Measure
        long start = System.nanoTime();
        for (int i = 0; i < MEASURED_ITERATIONS; i++) {
            try (VectorIndex index = VectorIndexFactory.cpu(DIMENSIONS)) {
                index.add(database);
                for (float[] q : queries) index.search(q, 10);
            }
        }
        return (System.nanoTime() - start) / 1_000_000.0 / MEASURED_ITERATIONS;
    }

    private double benchmarkGpuPersistent(float[][] database, float[][] queries) {
        if (!CudaDetector.isAvailable()) return 0;

        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            try (VectorIndex index = VectorIndexFactory.gpu(DIMENSIONS, DistanceMetric.EUCLIDEAN)) {
                index.add(database);
                for (float[] q : queries) index.search(q, 10);
            }
        }

        // Measure
        long start = System.nanoTime();
        for (int i = 0; i < MEASURED_ITERATIONS; i++) {
            try (VectorIndex index = VectorIndexFactory.gpu(DIMENSIONS, DistanceMetric.EUCLIDEAN)) {
                index.add(database);
                for (float[] q : queries) index.search(q, 10);
            }
        }
        return (System.nanoTime() - start) / 1_000_000.0 / MEASURED_ITERATIONS;
    }

    private TransferAnalysis measureTransferOverhead(int vectorCount) {
        if (!CudaDetector.isAvailable()) {
            return new TransferAnalysis(0, 0, 0);
        }

        float[][] database = generateRandomVectors(vectorCount, DIMENSIONS);
        float[] query = generateRandomVectors(1, DIMENSIONS)[0];

        // Measure upload time
        long uploadStart = System.nanoTime();
        VectorIndex index = VectorIndexFactory.gpu(DIMENSIONS, DistanceMetric.EUCLIDEAN);
        index.add(database);
        double uploadMs = (System.nanoTime() - uploadStart) / 1_000_000.0;

        // Measure search time (data already on GPU)
        // Warmup
        for (int i = 0; i < 5; i++) index.search(query, 10);

        long searchStart = System.nanoTime();
        for (int i = 0; i < 10; i++) index.search(query, 10);
        double searchMs = (System.nanoTime() - searchStart) / 1_000_000.0 / 10;

        index.close();

        double overheadPercent = (uploadMs / (uploadMs + searchMs)) * 100;
        return new TransferAnalysis(uploadMs, searchMs, overheadPercent);
    }

    private String formatResultRow(BenchmarkResult r) {
        String winner = r.gpuTimeMs == 0 ? "N/A" : (r.getSpeedup() > 1 ? "GPU" : "CPU");
        return String.format("| %,d | %.1f | %.1f | %.2fx | %s |\n",
            r.vectorCount, r.cpuTimeMs, r.gpuTimeMs, r.getSpeedup(), winner);
    }

    private String formatResultRowWithQueries(BenchmarkResult r) {
        String winner = r.gpuTimeMs == 0 ? "N/A" : (r.getSpeedup() > 1 ? "GPU" : "CPU");
        return String.format("| %,d | %d | %.1f | %.1f | %.2fx | %s |\n",
            r.vectorCount, r.queryCount, r.cpuTimeMs, r.gpuTimeMs, r.getSpeedup(), winner);
    }

    private String generateSummary() {
        if (!CudaDetector.isAvailable()) {
            return "No GPU detected. Install CUDA drivers and try again.\n";
        }

        return """
            **To contribute:** Copy this entire report and paste it into a GitHub Issue at:
            https://github.com/michaelangelo23/jvectorcuda/issues/new
            """;
    }

    private float[][] generateRandomVectors(int count, int dimensions) {
        Random random = new Random(42); // Fixed seed for reproducibility
        float[][] vectors = new float[count][dimensions];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < dimensions; j++) {
                vectors[i][j] = random.nextFloat() * 2 - 1;
            }
        }
        return vectors;
    }

    // Inner classes for results
    private static class BenchmarkResult {
        final int vectorCount;
        final int queryCount;
        final double cpuTimeMs;
        final double gpuTimeMs;

        BenchmarkResult(int vectorCount, int queryCount, double cpuTimeMs, double gpuTimeMs) {
            this.vectorCount = vectorCount;
            this.queryCount = queryCount;
            this.cpuTimeMs = cpuTimeMs;
            this.gpuTimeMs = gpuTimeMs;
        }

        double getSpeedup() {
            if (gpuTimeMs == 0) return 0;
            return cpuTimeMs / gpuTimeMs;
        }
    }

    private static class TransferAnalysis {
        final double uploadMs;
        final double searchMs;
        final double overheadPercent;

        TransferAnalysis(double uploadMs, double searchMs, double overheadPercent) {
            this.uploadMs = uploadMs;
            this.searchMs = searchMs;
            this.overheadPercent = overheadPercent;
        }
    }
}
