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
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

// Standalone benchmark runner. Run: ./gradlew benchmark
public class BenchmarkRunner {

    private static final int WARMUP_ITERATIONS = 3;
    private static final int MEASURED_ITERATIONS = 5;
    private static final int DIMENSIONS = 384;
    private static final int[] VECTOR_COUNTS = { 1_000, 10_000, 50_000 };
    private static final int[] QUERY_COUNTS = { 1, 10, 100 };

    // Collect results for CSV/JSON export
    private final List<BenchmarkResult> allResults = new ArrayList<>();

    /**
     * Print system specs to console. Useful for test setup.
     */
    public static void printSystemSpecs() {
        BenchmarkRunner runner = new BenchmarkRunner();
        System.out.println("\n" + "=".repeat(70));
        System.out.println("SYSTEM SPECIFICATIONS");
        System.out.println("=".repeat(70));
        System.out.println(runner.getSystemInfo());
        System.out.println("=".repeat(70) + "\n");
    }

    public static void main(String[] args) {
        System.out.println("JVectorCUDA Benchmark Runner");
        System.out.println("============================\n");

        // Print system specs first
        printSystemSpecs();

        BenchmarkRunner runner = new BenchmarkRunner();
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
        report.append(getSystemInfo());
        report.append("\n");

        boolean gpuAvailable = CudaDetector.isAvailable();
        if (!gpuAvailable) {
            report.append("**WARNING: No CUDA-capable GPU detected. CPU-only results below.**\n\n");
        }

        report.append("## Benchmark Results\n\n");

        report.append("### Single Query Performance\n\n");
        report.append("| Vectors | CPU (ms) | GPU (ms) | Speedup | Winner |\n");
        report.append("|---------|----------|----------|---------|--------|\n");

        for (int vectorCount : VECTOR_COUNTS) {
            BenchmarkResult result = runSingleQueryBenchmark(vectorCount);
            allResults.add(result); // Collect for CSV/JSON export
            report.append(formatResultRow(result));
        }
        report.append("\n");

        report.append("### Persistent Memory (Many Queries, Same Dataset)\n\n");
        report.append("| Vectors | Queries | CPU (ms) | GPU (ms) | Speedup | Winner |\n");
        report.append("|---------|---------|----------|----------|---------|--------|\n");

        for (int vectorCount : VECTOR_COUNTS) {
            for (int queryCount : QUERY_COUNTS) {
                if (queryCount == 1)
                    continue; // Skip single query (covered above)
                BenchmarkResult result = runPersistentMemoryBenchmark(vectorCount, queryCount);
                allResults.add(result); // Collect for CSV/JSON export
                report.append(formatResultRowWithQueries(result));
            }
        }
        report.append("\n");

        report.append("### Memory Transfer Analysis\n\n");
        TransferAnalysis transfer = measureTransferOverhead(50_000);
        report.append(String.format("- Upload time (50K vectors x %dD): %.1f ms\n", DIMENSIONS, transfer.uploadMs));
        report.append(String.format("- Search time (single query): %.1f ms\n", transfer.searchMs));
        report.append(String.format("- Transfer overhead: %.0f%%\n", transfer.overheadPercent));
        report.append("\n");

        report.append("## Summary\n\n");
        report.append(generateSummary());

        return report.toString();
    }

    public String getSystemInfo() {
        StringBuilder info = new StringBuilder();

        // GPU Information
        if (CudaDetector.isAvailable()) {
            info.append("- **GPU:** ").append(CudaDetector.getGpuInfo()).append("\n");
        } else {
            info.append("- **GPU:** Not detected\n");
        }

        // CPU Information (attempt to get actual model)
        String cpuModel = getCpuModel();
        OperatingSystemMXBean os = ManagementFactory.getOperatingSystemMXBean();
        int processors = os.getAvailableProcessors();
        info.append("- **CPU:** ").append(cpuModel)
                .append(" (").append(processors).append(" threads)\n");

        // Memory Information
        long maxMemory = Runtime.getRuntime().maxMemory() / (1024 * 1024);
        long totalMemory = Runtime.getRuntime().totalMemory() / (1024 * 1024);
        info.append("- **JVM Memory:** ").append(maxMemory).append(" MB max, ")
                .append(totalMemory).append(" MB allocated\n");

        // OS Information
        info.append("- **OS:** ").append(System.getProperty("os.name"))
                .append(" ").append(System.getProperty("os.version"))
                .append(" (").append(System.getProperty("os.arch")).append(")\n");

        // Java Information
        info.append("- **Java:** ").append(System.getProperty("java.version"))
                .append(" (").append(System.getProperty("java.vm.name")).append(")\n");

        return info.toString();
    }

    private String getCpuModel() {
        try {
            String os = System.getProperty("os.name").toLowerCase();

            if (os.contains("win")) {
                // Windows: Query WMI for CPU name
                String wmicPath = System.getenv("SystemRoot") + "\\System32\\wbem\\wmic.exe";
                Process process = Runtime.getRuntime().exec(
                        new String[] { wmicPath, "cpu", "get", "name" });
                try (java.io.BufferedReader reader = new java.io.BufferedReader(
                        new java.io.InputStreamReader(process.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        line = line.trim();
                        if (!line.isEmpty() && !line.equals("Name")) {
                            return line;
                        }
                    }
                }
            } else if (os.contains("linux")) {
                // Linux: Read from /proc/cpuinfo
                Process process = Runtime.getRuntime().exec(
                        new String[] { "/bin/cat", "/proc/cpuinfo" });
                try (java.io.BufferedReader reader = new java.io.BufferedReader(
                        new java.io.InputStreamReader(process.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        if (line.startsWith("model name")) {
                            return line.split(":")[1].trim();
                        }
                    }
                }
            } else if (os.contains("mac")) {
                // macOS: Use sysctl
                Process process = Runtime.getRuntime().exec(
                        new String[] { "/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string" });
                try (java.io.BufferedReader reader = new java.io.BufferedReader(
                        new java.io.InputStreamReader(process.getInputStream()))) {
                    String line = reader.readLine();
                    if (line != null) {
                        return line.trim();
                    }
                }
            }
        } catch (Exception e) {
            // Fallback silently
        }

        // Fallback to architecture if detection fails
        return System.getProperty("os.arch") + " CPU";
    }

    private BenchmarkResult runSingleQueryBenchmark(int vectorCount) {
        float[][] database = generateRandomVectors(vectorCount, DIMENSIONS);
        float[] query = generateRandomVectors(1, DIMENSIONS)[0];

        double cpuTime = benchmarkCpu(database, query);

        double gpuTime = benchmarkGpu(database, query);

        return new BenchmarkResult(vectorCount, 1, cpuTime, gpuTime);
    }

    private BenchmarkResult runPersistentMemoryBenchmark(int vectorCount, int queryCount) {
        float[][] database = generateRandomVectors(vectorCount, DIMENSIONS);
        float[][] queries = generateRandomVectors(queryCount, DIMENSIONS);

        double cpuTime = benchmarkCpuPersistent(database, queries);
        double gpuTime = benchmarkGpuPersistent(database, queries);

        return new BenchmarkResult(vectorCount, queryCount, cpuTime, gpuTime);
    }

    private double benchmarkCpu(float[][] database, float[] query) {
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            try (VectorIndex index = VectorIndexFactory.cpu(DIMENSIONS)) {
                index.add(database);
                index.search(query, 10);
            }
        }

        long start = System.nanoTime();
        for (int i = 0; i < MEASURED_ITERATIONS; i++) {
            try (VectorIndex index = VectorIndexFactory.cpu(DIMENSIONS)) {
                index.add(database);
                index.search(query, 10);
            }
        }
        return (System.nanoTime() - start) / 1_000_000.0 / MEASURED_ITERATIONS;
    }

    private double benchmarkGpu(float[][] database, float[] query) {
        if (!CudaDetector.isAvailable())
            return 0;

        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            try (VectorIndex index = VectorIndexFactory.gpu(DIMENSIONS, DistanceMetric.EUCLIDEAN)) {
                index.add(database);
                index.search(query, 10);
            }
        }

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
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            try (VectorIndex index = VectorIndexFactory.cpu(DIMENSIONS)) {
                index.add(database);
                for (float[] q : queries)
                    index.search(q, 10);
            }
        }

        long start = System.nanoTime();
        for (int i = 0; i < MEASURED_ITERATIONS; i++) {
            try (VectorIndex index = VectorIndexFactory.cpu(DIMENSIONS)) {
                index.add(database);
                for (float[] q : queries)
                    index.search(q, 10);
            }
        }
        return (System.nanoTime() - start) / 1_000_000.0 / MEASURED_ITERATIONS;
    }

    private double benchmarkGpuPersistent(float[][] database, float[][] queries) {
        if (!CudaDetector.isAvailable())
            return 0;

        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            try (VectorIndex index = VectorIndexFactory.gpu(DIMENSIONS, DistanceMetric.EUCLIDEAN)) {
                index.add(database);
                for (float[] q : queries)
                    index.search(q, 10);
            }
        }

        long start = System.nanoTime();
        for (int i = 0; i < MEASURED_ITERATIONS; i++) {
            try (VectorIndex index = VectorIndexFactory.gpu(DIMENSIONS, DistanceMetric.EUCLIDEAN)) {
                index.add(database);
                for (float[] q : queries)
                    index.search(q, 10);
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

        long uploadStart = System.nanoTime();
        VectorIndex index = VectorIndexFactory.gpu(DIMENSIONS, DistanceMetric.EUCLIDEAN);
        index.add(database);
        double uploadMs = (System.nanoTime() - uploadStart) / 1_000_000.0;

        for (int i = 0; i < 5; i++)
            index.search(query, 10);

        long searchStart = System.nanoTime();
        for (int i = 0; i < 10; i++)
            index.search(query, 10);
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

    /**
     * Export benchmark results to CSV format for regression tracking.
     * Each run appends to your data, allowing trend analysis over time.
     * 
     * @param filename Path to the CSV file
     */
    public void exportToCsv(String filename) throws java.io.IOException {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            // Header row
            writer.println(
                    "timestamp,gpu_name,vector_count,query_count,dimensions,cpu_time_ms,gpu_time_ms,speedup,winner");

            String timestamp = LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME);
            String gpuName = CudaDetector.isAvailable() ? CudaDetector.getGpuInfo().replace(",", ";") : "N/A";

            for (BenchmarkResult r : allResults) {
                String winner = r.gpuTimeMs == 0 ? "N/A" : (r.getSpeedup() > 1 ? "GPU" : "CPU");
                writer.printf("%s,%s,%d,%d,%d,%.2f,%.2f,%.2f,%s%n",
                        timestamp, gpuName, r.vectorCount, r.queryCount, DIMENSIONS,
                        r.cpuTimeMs, r.gpuTimeMs, r.getSpeedup(), winner);
            }
        }
    }

    /**
     * Export benchmark results to JSON format for programmatic analysis.
     * Includes system info and all benchmark results.
     * 
     * @param filename Path to the JSON file
     */
    public void exportToJson(String filename) throws java.io.IOException {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("{");
            writer.println("  \"benchmark_metadata\": {");
            writer.printf("    \"timestamp\": \"%s\",%n",
                    LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
            writer.printf("    \"gpu_name\": \"%s\",%n",
                    CudaDetector.isAvailable() ? CudaDetector.getGpuInfo() : "N/A");
            writer.printf("    \"dimensions\": %d,%n", DIMENSIONS);
            writer.printf("    \"warmup_iterations\": %d,%n", WARMUP_ITERATIONS);
            writer.printf("    \"measured_iterations\": %d%n", MEASURED_ITERATIONS);
            writer.println("  },");

            writer.println("  \"results\": [");
            for (int i = 0; i < allResults.size(); i++) {
                BenchmarkResult r = allResults.get(i);
                String winner = r.gpuTimeMs == 0 ? "N/A" : (r.getSpeedup() > 1 ? "GPU" : "CPU");
                writer.println("    {");
                writer.printf("      \"vector_count\": %d,%n", r.vectorCount);
                writer.printf("      \"query_count\": %d,%n", r.queryCount);
                writer.printf("      \"cpu_time_ms\": %.2f,%n", r.cpuTimeMs);
                writer.printf("      \"gpu_time_ms\": %.2f,%n", r.gpuTimeMs);
                writer.printf("      \"speedup\": %.2f,%n", r.getSpeedup());
                writer.printf("      \"winner\": \"%s\"%n", winner);
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

    private float[][] generateRandomVectors(int count, int dimensions) {
        Random random = new Random(42);
        float[][] vectors = new float[count][dimensions];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < dimensions; j++) {
                vectors[i][j] = random.nextFloat() * 2 - 1;
            }
        }
        return vectors;
    }

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
            if (gpuTimeMs == 0)
                return 0;
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
