package com.vindex.jvectorcuda.benchmark;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Comprehensive performance metrics logger for JVectorCUDA operations.
 * 
 * <p>Provides detailed timing breakdowns for:
 * <ul>
 *   <li>Memory allocation and deallocation</li>
 *   <li>Host-to-device and device-to-host transfers</li>
 *   <li>Kernel execution time</li>
 *   <li>CPU fallback operations</li>
 *   <li>End-to-end operation timing</li>
 * </ul>
 * 
 * <p>Supports multiple output formats:
 * <ul>
 *   <li>SLF4J logging (DEBUG/INFO/TRACE levels)</li>
 *   <li>CSV export for analysis</li>
 *   <li>Console summary reports</li>
 * </ul>
 * 
 * <p>Example usage:
 * <pre>{@code
 * PerformanceMetricsLogger logger = PerformanceMetricsLogger.getInstance();
 * 
 * // Record individual operations
 * logger.recordOperation("GPU_UPLOAD", 15.5);
 * logger.recordOperation("KERNEL_EXEC", 3.2);
 * logger.recordOperation("GPU_DOWNLOAD", 8.1);
 * 
 * // Or use timing blocks
 * try (PerformanceMetricsLogger.TimingBlock block = logger.startTiming("SEARCH")) {
 *     // ... operation ...
 * } // Automatically records timing when block closes
 * 
 * // Export results
 * logger.exportToCsv(Path.of("metrics.csv"));
 * logger.printSummary();
 * }</pre>
 * 
 * @author JVectorCUDA (AI-assisted, Human-verified)
 * @since 1.0.0
 */
public class PerformanceMetricsLogger {

    private static final Logger logger = LoggerFactory.getLogger(PerformanceMetricsLogger.class);

    // Singleton instance
    private static volatile PerformanceMetricsLogger instance;
    private static final Object LOCK = new Object();

    // Operation categories
    public static final String OP_GPU_UPLOAD = "GPU_UPLOAD";
    public static final String OP_GPU_DOWNLOAD = "GPU_DOWNLOAD";
    public static final String OP_KERNEL_EXEC = "KERNEL_EXEC";
    public static final String OP_CPU_COMPUTE = "CPU_COMPUTE";
    public static final String OP_MEMORY_ALLOC = "MEMORY_ALLOC";
    public static final String OP_MEMORY_FREE = "MEMORY_FREE";
    public static final String OP_TOTAL_SEARCH = "TOTAL_SEARCH";
    public static final String OP_TOTAL_ADD = "TOTAL_ADD";

    // Storage for metrics
    private final Map<String, List<MetricEntry>> metricsByOperation;
    private final List<MetricEntry> allMetrics;
    private volatile boolean enabled = true;
    private volatile LogLevel logLevel = LogLevel.INFO;

    /**
     * Log levels for metric output.
     */
    public enum LogLevel {
        TRACE,  // Log every operation
        DEBUG,  // Log operations with timing details
        INFO,   // Log only summaries
        NONE    // Disable logging
    }

    /**
     * Private constructor for singleton pattern.
     */
    private PerformanceMetricsLogger() {
        this.metricsByOperation = new ConcurrentHashMap<>();
        this.allMetrics = Collections.synchronizedList(new ArrayList<>());
    }

    /**
     * Gets the singleton instance of the logger.
     * 
     * @return the logger instance
     */
    public static PerformanceMetricsLogger getInstance() {
        if (instance == null) {
            synchronized (LOCK) {
                if (instance == null) {
                    instance = new PerformanceMetricsLogger();
                }
            }
        }
        return instance;
    }

    /**
     * Records a timed operation.
     * 
     * @param operation operation name (use constants like OP_GPU_UPLOAD)
     * @param durationMs duration in milliseconds
     */
    public void recordOperation(String operation, double durationMs) {
        recordOperation(operation, durationMs, null);
    }

    /**
     * Records a timed operation with additional details.
     * 
     * @param operation operation name
     * @param durationMs duration in milliseconds
     * @param details additional details (e.g., "vectors=10000, dims=384")
     */
    public void recordOperation(String operation, double durationMs, String details) {
        if (!enabled) {
            return;
        }

        MetricEntry entry = new MetricEntry(operation, durationMs, details, Instant.now());
        
        metricsByOperation.computeIfAbsent(operation, k -> Collections.synchronizedList(new ArrayList<>()))
            .add(entry);
        allMetrics.add(entry);
        
        logEntry(entry);
    }

    /**
     * Starts a timing block that automatically records duration when closed.
     * 
     * @param operation operation name
     * @return timing block (use with try-with-resources)
     */
    public TimingBlock startTiming(String operation) {
        return new TimingBlock(this, operation, null);
    }

    /**
     * Starts a timing block with additional details.
     * 
     * @param operation operation name
     * @param details additional context
     * @return timing block
     */
    public TimingBlock startTiming(String operation, String details) {
        return new TimingBlock(this, operation, details);
    }

    /**
     * Gets statistics for a specific operation.
     * 
     * @param operation operation name
     * @return operation statistics, or null if no data
     */
    public OperationStats getStats(String operation) {
        List<MetricEntry> entries = metricsByOperation.get(operation);
        if (entries == null || entries.isEmpty()) {
            return null;
        }
        return calculateStats(operation, entries);
    }

    /**
     * Gets statistics for all operations.
     * 
     * @return map of operation name to statistics
     */
    public Map<String, OperationStats> getAllStats() {
        Map<String, OperationStats> stats = new ConcurrentHashMap<>();
        for (Map.Entry<String, List<MetricEntry>> entry : metricsByOperation.entrySet()) {
            stats.put(entry.getKey(), calculateStats(entry.getKey(), entry.getValue()));
        }
        return stats;
    }

    /**
     * Prints a summary of all recorded metrics.
     */
    public void printSummary() {
        System.out.println("\n=== Performance Metrics Summary ===\n");
        System.out.printf("%-20s %10s %10s %10s %10s %10s%n",
            "Operation", "Count", "Total(ms)", "Avg(ms)", "Min(ms)", "Max(ms)");
        System.out.println("-".repeat(75));
        
        Map<String, OperationStats> allStats = getAllStats();
        
        // Sort by total time descending
        allStats.entrySet().stream()
            .sorted((a, b) -> Double.compare(b.getValue().totalMs, a.getValue().totalMs))
            .forEach(entry -> {
                OperationStats stats = entry.getValue();
                System.out.printf("%-20s %10d %10.2f %10.2f %10.2f %10.2f%n",
                    entry.getKey(),
                    stats.count,
                    stats.totalMs,
                    stats.avgMs,
                    stats.minMs,
                    stats.maxMs);
            });
        
        System.out.println("-".repeat(75));
        System.out.printf("Total operations: %d%n", allMetrics.size());
        System.out.printf("Total time: %.2f ms%n", 
            allStats.values().stream().mapToDouble(s -> s.totalMs).sum());
    }

    /**
     * Exports all metrics to CSV format.
     * 
     * @param outputPath path to output file
     * @throws IOException if writing fails
     */
    public void exportToCsv(Path outputPath) throws IOException {
        Files.createDirectories(outputPath.getParent());
        
        try (PrintWriter writer = new PrintWriter(new FileWriter(outputPath.toFile()))) {
            // Header
            writer.println("timestamp,operation,duration_ms,details");
            
            // Data
            for (MetricEntry entry : allMetrics) {
                writer.printf("%s,%s,%.6f,%s%n",
                    entry.timestamp,
                    entry.operation,
                    entry.durationMs,
                    entry.details != null ? "\"" + entry.details.replace("\"", "\"\"") + "\"" : "");
            }
        }
        
        logger.info("Exported {} metrics to {}", allMetrics.size(), outputPath);
    }

    /**
     * Exports summary statistics to CSV.
     * 
     * @param outputPath path to output file
     * @throws IOException if writing fails
     */
    public void exportStatsToCsv(Path outputPath) throws IOException {
        Files.createDirectories(outputPath.getParent());
        
        try (PrintWriter writer = new PrintWriter(new FileWriter(outputPath.toFile()))) {
            // Header
            writer.println("operation,count,total_ms,avg_ms,min_ms,max_ms,std_dev_ms");
            
            // Data
            for (Map.Entry<String, OperationStats> entry : getAllStats().entrySet()) {
                OperationStats stats = entry.getValue();
                writer.printf("%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f%n",
                    entry.getKey(),
                    stats.count,
                    stats.totalMs,
                    stats.avgMs,
                    stats.minMs,
                    stats.maxMs,
                    stats.stdDevMs);
            }
        }
        
        logger.info("Exported statistics to {}", outputPath);
    }

    /**
     * Clears all recorded metrics.
     */
    public void clear() {
        metricsByOperation.clear();
        allMetrics.clear();
        logger.debug("Metrics cleared");
    }

    /**
     * Sets whether metrics collection is enabled.
     * 
     * @param enabled true to enable, false to disable
     */
    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    /**
     * Returns whether metrics collection is enabled.
     * 
     * @return true if enabled
     */
    public boolean isEnabled() {
        return enabled;
    }

    /**
     * Sets the log level for metric output.
     * 
     * @param level desired log level
     */
    public void setLogLevel(LogLevel level) {
        this.logLevel = level;
    }

    /**
     * Returns the current log level.
     * 
     * @return current log level
     */
    public LogLevel getLogLevel() {
        return logLevel;
    }

    /**
     * Returns the total number of recorded metrics.
     * 
     * @return metric count
     */
    public int getMetricCount() {
        return allMetrics.size();
    }

    // ==================== Private Methods ====================

    private void logEntry(MetricEntry entry) {
        switch (logLevel) {
            case TRACE:
                logger.trace("[{}] {} took {:.3f} ms {}",
                    entry.timestamp, entry.operation, entry.durationMs,
                    entry.details != null ? "(" + entry.details + ")" : "");
                break;
            case DEBUG:
                logger.debug("{}: {:.3f} ms", entry.operation, entry.durationMs);
                break;
            case INFO:
            case NONE:
                // No per-operation logging
                break;
        }
    }

    private OperationStats calculateStats(String operation, List<MetricEntry> entries) {
        if (entries.isEmpty()) {
            return new OperationStats(operation, 0, 0, 0, 0, 0, 0);
        }
        
        double total = 0;
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        
        for (MetricEntry entry : entries) {
            total += entry.durationMs;
            min = Math.min(min, entry.durationMs);
            max = Math.max(max, entry.durationMs);
        }
        
        double avg = total / entries.size();
        
        // Calculate standard deviation
        double sumSquaredDiff = 0;
        for (MetricEntry entry : entries) {
            double diff = entry.durationMs - avg;
            sumSquaredDiff += diff * diff;
        }
        double stdDev = Math.sqrt(sumSquaredDiff / entries.size());
        
        return new OperationStats(operation, entries.size(), total, avg, min, max, stdDev);
    }

    // ==================== Inner Classes ====================

    /**
     * A timing block for automatic duration measurement.
     * Use with try-with-resources.
     */
    public static class TimingBlock implements AutoCloseable {
        private final PerformanceMetricsLogger metricsLogger;
        private final String operation;
        private final String details;
        private final long startNanos;

        TimingBlock(PerformanceMetricsLogger logger, String operation, String details) {
            this.metricsLogger = logger;
            this.operation = operation;
            this.details = details;
            this.startNanos = System.nanoTime();
        }

        @Override
        public void close() {
            double durationMs = (System.nanoTime() - startNanos) / 1_000_000.0;
            metricsLogger.recordOperation(operation, durationMs, details);
        }

        /**
         * Returns the elapsed time so far without closing the block.
         * 
         * @return elapsed time in milliseconds
         */
        public double getElapsedMs() {
            return (System.nanoTime() - startNanos) / 1_000_000.0;
        }
    }

    /**
     * A single metric entry.
     */
    private static class MetricEntry {
        final String operation;
        final double durationMs;
        final String details;
        final Instant timestamp;

        MetricEntry(String operation, double durationMs, String details, Instant timestamp) {
            this.operation = operation;
            this.durationMs = durationMs;
            this.details = details;
            this.timestamp = timestamp;
        }
    }

    /**
     * Statistics for a specific operation type.
     */
    public static class OperationStats {
        public final String operation;
        public final int count;
        public final double totalMs;
        public final double avgMs;
        public final double minMs;
        public final double maxMs;
        public final double stdDevMs;

        OperationStats(String operation, int count, double totalMs, double avgMs, 
                      double minMs, double maxMs, double stdDevMs) {
            this.operation = operation;
            this.count = count;
            this.totalMs = totalMs;
            this.avgMs = avgMs;
            this.minMs = minMs;
            this.maxMs = maxMs;
            this.stdDevMs = stdDevMs;
        }

        @Override
        public String toString() {
            return String.format("%s: count=%d, avg=%.3fms, min=%.3fms, max=%.3fms",
                operation, count, avgMs, minMs, maxMs);
        }
    }
}
