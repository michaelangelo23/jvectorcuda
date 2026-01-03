package com.vindex.jvectorcuda.benchmark;

import com.vindex.jvectorcuda.CudaDetector;
import com.vindex.jvectorcuda.DistanceMetric;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

// Standalone benchmark runner. Run: ./gradlew benchmark
public class BenchmarkRunner {

    private static final int WARMUP_ITERATIONS = 3;
    private static final int MEASURED_ITERATIONS = 5;
    private static final int DIMENSIONS = 384;
    private static final int[] VECTOR_COUNTS = { 1_000, 10_000, 50_000 };
    private static final int[] QUERY_COUNTS = { 1, 10, 100 };

    // Collect results for CSV/JSON export
    private final List<BenchmarkResult> allResults = new ArrayList<>();
    private final BenchmarkFramework framework;

    public BenchmarkRunner() {
        this.framework = new BenchmarkFramework();
    }

    /**
     * Print system specs to console. Useful for test setup.
     */
    public static void printSystemSpecs() {

        System.out.println("\n" + "=".repeat(70));
        System.out.println("SYSTEM SPECIFICATIONS");
        System.out.println("=".repeat(70));
        // Using a temporary BenchmarkFramework to get system info would be cleaner,
        // but for now we'll stick to what we have or reuse the one in runner if
        // instance available.
        // Since this is static, we'll just create a fresh runner/framework.
        System.out.println(new BenchmarkFramework().getSystemInfo());
        System.out.println("=".repeat(70) + "\n");
    }

    public static void main(String[] args) {
        System.out.println("JVectorCUDA Benchmark Runner");
        System.out.println("============================\n");

        BenchmarkRunner runner = new BenchmarkRunner();

        // Print system specs
        System.out.println(runner.framework.getSystemInfo());
        System.out.println("=".repeat(70) + "\n");

        String report = runner.runFullBenchmark();

        System.out.println(report);

        // Generate timestamp for file names
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));

        // Output directory for benchmark results
        String outputDir = "benchmarkTests/";
        java.io.File dir = new java.io.File(outputDir);
        if (!dir.exists()) {
            dir.mkdirs();
        }

        // Save Markdown report
        try (PrintWriter writer = new PrintWriter(new FileWriter(outputDir + "benchmark-report.md"))) {
            writer.print(report);
            System.out.println("\nReport saved to: " + outputDir + "benchmark-report.md");
        } catch (Exception e) {
            System.err.println("Could not save report file: " + e.getMessage());
        }

        // Save CSV for regression tracking
        try {
            String csvPath = outputDir + "benchmark-results-" + timestamp + ".csv";
            runner.exportToCsv(csvPath);
            System.out.println("CSV saved to: " + csvPath);
        } catch (Exception e) {
            System.err.println("Could not save CSV file: " + e.getMessage());
        }

        // Save JSON for programmatic analysis
        try {
            String jsonPath = outputDir + "benchmark-results-" + timestamp + ".json";
            runner.exportToJson(jsonPath);
            System.out.println("JSON saved to: " + jsonPath);
        } catch (Exception e) {
            System.err.println("Could not save JSON file: " + e.getMessage());
        }
    }

    public String runFullBenchmark() {
        StringBuilder report = new StringBuilder();

        report.append("# GPU Benchmark Results\n\n");
        report.append("Generated: ").append(LocalDateTime.now().format(
                DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"))).append("\n\n");

        report.append("## System Information\n\n");
        report.append(framework.getSystemInfo());
        report.append("\n");

        if (!framework.isGpuAvailable()) {
            report.append("**WARNING: No CUDA-capable GPU detected. CPU-only results below.**\n\n");
        }

        report.append("## Benchmark Results\n\n");

        report.append("### Single Query Performance\n\n");
        report.append("| Vectors | CPU (ms) | GPU (ms) | CPU (QPS) | GPU (QPS) | Speedup |\n");
        report.append("|---------|----------|----------|-----------|-----------|---------|\n");

        for (int vectorCount : VECTOR_COUNTS) {
            BenchmarkConfig config = BenchmarkConfig.builder()
                    .vectorCount(vectorCount)
                    .dimensions(DIMENSIONS)
                    .queryCount(1)
                    .k(10)
                    .warmupIterations(WARMUP_ITERATIONS)
                    .measuredIterations(MEASURED_ITERATIONS)
                    .distanceMetric(DistanceMetric.EUCLIDEAN)
                    .build();

            BenchmarkResult result = framework.run(config);
            allResults.add(result);
            report.append(formatResultRow(result));
        }
        report.append("\n");

        report.append("### Persistent Memory (Many Queries, Same Dataset)\n\n");
        report.append("| Vectors | Queries | CPU (ms) | GPU (ms) | CPU (QPS) | GPU (QPS) | Speedup |\n");
        report.append("|---------|---------|----------|----------|-----------|-----------|---------|\n");

        for (int vectorCount : VECTOR_COUNTS) {
            for (int queryCount : QUERY_COUNTS) {
                if (queryCount == 1)
                    continue; // Skip single query (covered above)

                BenchmarkConfig config = BenchmarkConfig.builder()
                        .vectorCount(vectorCount)
                        .dimensions(DIMENSIONS)
                        .queryCount(queryCount)
                        .k(10)
                        .warmupIterations(WARMUP_ITERATIONS)
                        .measuredIterations(MEASURED_ITERATIONS)
                        .distanceMetric(DistanceMetric.EUCLIDEAN)
                        .build();

                BenchmarkResult result = framework.runPersistentMemoryBenchmark(config);
                allResults.add(result);
                report.append(formatResultRowWithQueries(result));
            }
        }
        report.append("\n");

        report.append("## Summary\n\n");
        report.append(generateSummary());

        return report.toString();
    }

    private String formatResultRow(BenchmarkResult r) {
        return String.format("| %,d | %.2f | %.2f | %.1f | %.1f | %.2fx |\n",
                r.getVectorCount(), r.getCpuTimeMs(), r.getGpuTimeMs(),
                r.getCpuThroughput(), r.getGpuThroughput(), r.getSpeedup());
    }

    private String formatResultRowWithQueries(BenchmarkResult r) {
        return String.format("| %,d | %d | %.2f | %.2f | %.1f | %.1f | %.2fx |\n",
                r.getVectorCount(), r.getQueryCount(), r.getCpuTimeMs(), r.getGpuTimeMs(),
                r.getCpuThroughput(), r.getGpuThroughput(), r.getSpeedup());
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

    public void exportToCsv(String filename) throws java.io.IOException {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println(BenchmarkResult.getCsvHeader());

            for (BenchmarkResult r : allResults) {
                writer.println(r.toCsvLine());
            }
        }
    }

    public void exportToJson(String filename) throws java.io.IOException {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("{");
            writer.println("  \"benchmark_metadata\": {");
            writer.printf("    \"timestamp\": \"%s\",%n",
                    LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
            writer.printf("    \"gpu_name\": \"%s\",%n",
                    framework.getGpuName());
            writer.printf("    \"dimensions\": %d,%n", DIMENSIONS);
            writer.printf("    \"warmup_iterations\": %d,%n", WARMUP_ITERATIONS);
            writer.printf("    \"measured_iterations\": %d%n", MEASURED_ITERATIONS);
            writer.println("  },");

            writer.println("  \"results\": [");
            for (int i = 0; i < allResults.size(); i++) {
                BenchmarkResult r = allResults.get(i);
                writer.println("    {");
                writer.printf("      \"vector_count\": %d,%n", r.getVectorCount());
                writer.printf("      \"query_count\": %d,%n", r.getQueryCount());
                writer.printf("      \"cpu_time_ms\": %.2f,%n", r.getCpuTimeMs());
                writer.printf("      \"gpu_time_ms\": %.2f,%n", r.getGpuTimeMs());
                writer.printf("      \"cpu_qps\": %.2f,%n", r.getCpuThroughput());
                writer.printf("      \"gpu_qps\": %.2f,%n", r.getGpuThroughput());
                writer.printf("      \"speedup\": %.2f%n", r.getSpeedup());
                writer.print("    }");
                if (i < allResults.size() - 1)
                    writer.println(",");
                else
                    writer.println();
            }
            writer.println("  ]");
            writer.println("}");
        }
    }
}
