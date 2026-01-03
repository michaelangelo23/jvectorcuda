package com.vindex.jvectorcuda.benchmark;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;

/**
 * Exports benchmark results to CSV and JSON formats.
 */
public class BenchmarkResultExporter {

    /**
     * Exports comprehensive benchmark results to CSV file.
     *
     * @param results List of benchmark results
     * @param outputPath Path to output CSV file
     * @throws IOException If file write fails
     */
    public static void exportToCSV(List<StandardBenchmarkSuite.ComprehensiveBenchmarkResult> results, Path outputPath) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(outputPath, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            // Write header
            writer.write("dataset,vectors,dimensions,k,metric,throughput_qps,latency_p50_ms,latency_p95_ms,latency_p99_ms,latency_min_ms,latency_max_ms,latency_mean_ms,build_time_ms,recall_at_10,heap_used_mb,heap_max_mb,offheap_mb,timestamp\n");

            // Write data rows
            for (StandardBenchmarkSuite.ComprehensiveBenchmarkResult result : results) {
                writer.write(String.format("%s,%d,%d,%d,%s,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%.4f,%.2f,%.2f,%.2f,%d\n",
                    escapeCsv(result.getDatasetName()),
                    result.getVectorCount(),
                    result.getDimensions(),
                    result.getK(),
                    result.getMetric(),
                    result.getThroughputQPS(),
                    result.getLatency().getP50(),
                    result.getLatency().getP95(),
                    result.getLatency().getP99(),
                    result.getLatency().getMin(),
                    result.getLatency().getMax(),
                    result.getLatency().getMean(),
                    result.getBuildTimeMs(),
                    result.getRecallAt10(),
                    result.getMemory().getHeapUsedMB(),
                    result.getMemory().getHeapMaxMB(),
                    result.getMemory().getOffHeapUsedMB(),
                    result.getTimestamp()
                ));
            }
        }
    }

    /**
     * Exports comprehensive benchmark results to JSON file.
     *
     * @param results List of benchmark results
     * @param outputPath Path to output JSON file
     * @throws IOException If file write fails
     */
    public static void exportToJSON(List<StandardBenchmarkSuite.ComprehensiveBenchmarkResult> results, Path outputPath) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(outputPath, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            writer.write("{\n");
            writer.write("  \"benchmark_results\": [\n");

            for (int i = 0; i < results.size(); i++) {
                StandardBenchmarkSuite.ComprehensiveBenchmarkResult result = results.get(i);
                writer.write("    {\n");
                writer.write(String.format("      \"dataset\": \"%s\",\n", escapeJson(result.getDatasetName())));
                writer.write(String.format("      \"vector_count\": %d,\n", result.getVectorCount()));
                writer.write(String.format("      \"dimensions\": %d,\n", result.getDimensions()));
                writer.write(String.format("      \"k\": %d,\n", result.getK()));
                writer.write(String.format("      \"metric\": \"%s\",\n", result.getMetric()));
                writer.write("      \"performance\": {\n");
                writer.write(String.format("        \"throughput_qps\": %.2f,\n", result.getThroughputQPS()));
                writer.write(String.format("        \"build_time_ms\": %.2f\n", result.getBuildTimeMs()));
                writer.write("      },\n");
                writer.write("      \"latency\": {\n");
                writer.write(String.format("        \"p50_ms\": %.4f,\n", result.getLatency().getP50()));
                writer.write(String.format("        \"p95_ms\": %.4f,\n", result.getLatency().getP95()));
                writer.write(String.format("        \"p99_ms\": %.4f,\n", result.getLatency().getP99()));
                writer.write(String.format("        \"min_ms\": %.4f,\n", result.getLatency().getMin()));
                writer.write(String.format("        \"max_ms\": %.4f,\n", result.getLatency().getMax()));
                writer.write(String.format("        \"mean_ms\": %.4f\n", result.getLatency().getMean()));
                writer.write("      },\n");
                writer.write("      \"memory\": {\n");
                writer.write(String.format("        \"heap_used_mb\": %.2f,\n", result.getMemory().getHeapUsedMB()));
                writer.write(String.format("        \"heap_max_mb\": %.2f,\n", result.getMemory().getHeapMaxMB()));
                writer.write(String.format("        \"offheap_mb\": %.2f\n", result.getMemory().getOffHeapUsedMB()));
                writer.write("      },\n");
                writer.write(String.format("      \"recall_at_10\": %.4f,\n", result.getRecallAt10()));
                writer.write(String.format("      \"timestamp\": %d\n", result.getTimestamp()));
                writer.write("    }");

                if (i < results.size() - 1) {
                    writer.write(",");
                }
                writer.write("\n");
            }

            writer.write("  ]\n");
            writer.write("}\n");
        }
    }

    /**
     * Appends a single benchmark result to an existing CSV file.
     *
     * @param result Benchmark result to append
     * @param outputPath Path to CSV file
     * @throws IOException If file write fails
     */
    public static void appendToCSV(StandardBenchmarkSuite.ComprehensiveBenchmarkResult result, Path outputPath) throws IOException {
        boolean fileExists = Files.exists(outputPath);

        try (BufferedWriter writer = Files.newBufferedWriter(outputPath, StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            // Write header if file is new
            if (!fileExists) {
                writer.write("dataset,vectors,dimensions,k,metric,throughput_qps,latency_p50_ms,latency_p95_ms,latency_p99_ms,latency_min_ms,latency_max_ms,latency_mean_ms,build_time_ms,recall_at_10,heap_used_mb,heap_max_mb,offheap_mb,timestamp\n");
            }

            // Write data row
            writer.write(String.format("%s,%d,%d,%d,%s,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%.4f,%.2f,%.2f,%.2f,%d\n",
                escapeCsv(result.getDatasetName()),
                result.getVectorCount(),
                result.getDimensions(),
                result.getK(),
                result.getMetric(),
                result.getThroughputQPS(),
                result.getLatency().getP50(),
                result.getLatency().getP95(),
                result.getLatency().getP99(),
                result.getLatency().getMin(),
                result.getLatency().getMax(),
                result.getLatency().getMean(),
                result.getBuildTimeMs(),
                result.getRecallAt10(),
                result.getMemory().getHeapUsedMB(),
                result.getMemory().getHeapMaxMB(),
                result.getMemory().getOffHeapUsedMB(),
                result.getTimestamp()
            ));
        }
    }

    private static String escapeCsv(String value) {
        if (value == null) {
            return "";
        }
        if (value.contains(",") || value.contains("\"") || value.contains("\n")) {
            return "\"" + value.replace("\"", "\"\"") + "\"";
        }
        return value;
    }

    private static String escapeJson(String value) {
        if (value == null) {
            return "";
        }
        return value.replace("\\", "\\\\")
                    .replace("\"", "\\\"")
                    .replace("\n", "\\n")
                    .replace("\r", "\\r")
                    .replace("\t", "\\t");
    }
}
