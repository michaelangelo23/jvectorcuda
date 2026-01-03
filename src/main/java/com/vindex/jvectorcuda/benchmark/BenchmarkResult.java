package com.vindex.jvectorcuda.benchmark;

import java.time.Instant;
import java.util.Objects;

// Immutable benchmark results with timing and memory metrics.
public final class BenchmarkResult {

    private final double cpuTimeMs;
    private final double gpuTimeMs;
    private final double gpuTransferTimeMs;
    private final double gpuComputeTimeMs;

    private final long gpuMemoryUsedBytes;
    private final long cpuMemoryUsedBytes;

    private final int vectorCount;
    private final int dimensions;
    private final int queryCount;
    private final int k;
    private final String distanceMetric;

    private final Instant timestamp;
    private final String gpuName;
    private final int warmupIterations;
    private final int measuredIterations;

    private BenchmarkResult(Builder builder) {
        this.cpuTimeMs = builder.cpuTimeMs;
        this.gpuTimeMs = builder.gpuTimeMs;
        this.gpuTransferTimeMs = builder.gpuTransferTimeMs;
        this.gpuComputeTimeMs = builder.gpuComputeTimeMs;
        this.gpuMemoryUsedBytes = builder.gpuMemoryUsedBytes;
        this.cpuMemoryUsedBytes = builder.cpuMemoryUsedBytes;
        this.vectorCount = builder.vectorCount;
        this.dimensions = builder.dimensions;
        this.queryCount = builder.queryCount;
        this.k = builder.k;
        this.distanceMetric = builder.distanceMetric;
        this.timestamp = builder.timestamp;
        this.gpuName = builder.gpuName;
        this.warmupIterations = builder.warmupIterations;
        this.measuredIterations = builder.measuredIterations;
    }

    // Creates a new builder
    public static Builder builder() {
        return new Builder();
    }

    // GPU speedup over CPU (>1 = GPU faster, <1 = CPU faster)
    public double getSpeedup() {
        if (gpuTimeMs <= 0) {
            return 0.0;
        }
        return cpuTimeMs / gpuTimeMs;
    }

    // True if GPU was faster
    public boolean isGpuFaster() {
        return getSpeedup() > 1.0;
    }

    // Transfer overhead as percentage of total GPU time
    public double getTransferOverheadPercent() {
        if (gpuTimeMs <= 0) {
            return 0.0;
        }
        return (gpuTransferTimeMs / gpuTimeMs) * 100.0;
    }

    // GPU queries per second
    public double getGpuThroughput() {
        if (gpuTimeMs <= 0 || queryCount <= 0) {
            return 0.0;
        }
        return (queryCount * 1000.0) / gpuTimeMs;
    }

    // CPU queries per second
    public double getCpuThroughput() {
        if (cpuTimeMs <= 0 || queryCount <= 0) {
            return 0.0;
        }
        return (queryCount * 1000.0) / cpuTimeMs;
    }

    // GPU memory usage in MB
    public double getGpuMemoryUsedMB() {
        return gpuMemoryUsedBytes / (1024.0 * 1024.0);
    }

    public double getCpuTimeMs() {
        return cpuTimeMs;
    }

    public double getGpuTimeMs() {
        return gpuTimeMs;
    }

    public double getGpuTransferTimeMs() {
        return gpuTransferTimeMs;
    }

    public double getGpuComputeTimeMs() {
        return gpuComputeTimeMs;
    }

    public long getGpuMemoryUsedBytes() {
        return gpuMemoryUsedBytes;
    }

    public long getCpuMemoryUsedBytes() {
        return cpuMemoryUsedBytes;
    }

    public int getVectorCount() {
        return vectorCount;
    }

    public int getDimensions() {
        return dimensions;
    }

    public int getQueryCount() {
        return queryCount;
    }

    public int getK() {
        return k;
    }

    public String getDistanceMetric() {
        return distanceMetric;
    }

    public Instant getTimestamp() {
        return timestamp;
    }

    public String getGpuName() {
        return gpuName;
    }

    public int getWarmupIterations() {
        return warmupIterations;
    }

    public int getMeasuredIterations() {
        return measuredIterations;
    }

    // Multi-line summary for logging
    public String toSummaryString() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Benchmark Results ===\n");
        sb.append(String.format("Timestamp: %s%n", timestamp));
        sb.append(String.format("GPU: %s%n", gpuName));
        sb.append(String.format("Configuration: %,d vectors x %d dimensions%n", vectorCount, dimensions));
        sb.append(String.format("Queries: %d x k=%d%n", queryCount, k));
        sb.append(String.format("Distance Metric: %s%n", distanceMetric));
        sb.append(String.format("Iterations: %d warmup + %d measured%n", warmupIterations, measuredIterations));
        sb.append("\n--- Timing ---\n");
        sb.append(String.format("CPU Time: %.2f ms%n", cpuTimeMs));
        sb.append(String.format("GPU Time: %.2f ms (Transfer: %.2f ms, Compute: %.2f ms)%n",
                gpuTimeMs, gpuTransferTimeMs, gpuComputeTimeMs));
        sb.append(String.format("Speedup: %.2fx (%s)%n", getSpeedup(), isGpuFaster() ? "GPU wins" : "CPU wins"));
        sb.append(String.format("Transfer Overhead: %.1f%%%n", getTransferOverheadPercent()));
        sb.append("\n--- Throughput ---\n");
        sb.append(String.format("CPU: %.1f queries/sec%n", getCpuThroughput()));
        sb.append(String.format("GPU: %.1f queries/sec%n", getGpuThroughput()));
        sb.append("\n--- Memory ---\n");
        sb.append(String.format("GPU Memory: %.2f MB%n", getGpuMemoryUsedMB()));
        return sb.toString();
    }

    // Single-line CSV format
    public String toCsvLine() {
        return String.format("%s,%s,%d,%d,%d,%d,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%.1f,%.1f",
                timestamp, gpuName, vectorCount, dimensions, queryCount, k, distanceMetric,
                cpuTimeMs, gpuTimeMs, gpuTransferTimeMs, gpuComputeTimeMs, getSpeedup(), gpuMemoryUsedBytes,
                getCpuThroughput(), getGpuThroughput());
    }

    // CSV header line
    public static String getCsvHeader() {
        return "timestamp,gpu_name,vector_count,dimensions,query_count,k,metric,cpu_ms,gpu_ms,transfer_ms,compute_ms,speedup,gpu_memory_bytes,cpu_qps,gpu_qps";
    }

    @Override
    public String toString() {
        return String.format("BenchmarkResult[vectors=%d, dims=%d, speedup=%.2fx]",
                vectorCount, dimensions, getSpeedup());
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        BenchmarkResult that = (BenchmarkResult) o;
        return Double.compare(cpuTimeMs, that.cpuTimeMs) == 0 &&
                Double.compare(gpuTimeMs, that.gpuTimeMs) == 0 &&
                vectorCount == that.vectorCount &&
                dimensions == that.dimensions;
    }

    @Override
    public int hashCode() {
        return Objects.hash(cpuTimeMs, gpuTimeMs, vectorCount, dimensions);
    }

    // Builder for BenchmarkResult
    public static final class Builder {
        private double cpuTimeMs;
        private double gpuTimeMs;
        private double gpuTransferTimeMs;
        private double gpuComputeTimeMs;
        private long gpuMemoryUsedBytes;
        private long cpuMemoryUsedBytes;
        private int vectorCount;
        private int dimensions;
        private int queryCount = 1;
        private int k = 10;
        private String distanceMetric = "EUCLIDEAN";
        private Instant timestamp = Instant.now();
        private String gpuName = "Unknown";
        private int warmupIterations = 5;
        private int measuredIterations = 10;

        private Builder() {
        }

        public Builder cpuTimeMs(double cpuTimeMs) {
            this.cpuTimeMs = cpuTimeMs;
            return this;
        }

        public Builder gpuTimeMs(double gpuTimeMs) {
            this.gpuTimeMs = gpuTimeMs;
            return this;
        }

        public Builder gpuTransferTimeMs(double gpuTransferTimeMs) {
            this.gpuTransferTimeMs = gpuTransferTimeMs;
            return this;
        }

        public Builder gpuComputeTimeMs(double gpuComputeTimeMs) {
            this.gpuComputeTimeMs = gpuComputeTimeMs;
            return this;
        }

        public Builder gpuMemoryUsedBytes(long gpuMemoryUsedBytes) {
            this.gpuMemoryUsedBytes = gpuMemoryUsedBytes;
            return this;
        }

        public Builder cpuMemoryUsedBytes(long cpuMemoryUsedBytes) {
            this.cpuMemoryUsedBytes = cpuMemoryUsedBytes;
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

        public Builder queryCount(int queryCount) {
            this.queryCount = queryCount;
            return this;
        }

        public Builder k(int k) {
            this.k = k;
            return this;
        }

        public Builder distanceMetric(String distanceMetric) {
            this.distanceMetric = distanceMetric;
            return this;
        }

        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public Builder gpuName(String gpuName) {
            this.gpuName = gpuName;
            return this;
        }

        public Builder warmupIterations(int warmupIterations) {
            this.warmupIterations = warmupIterations;
            return this;
        }

        public Builder measuredIterations(int measuredIterations) {
            this.measuredIterations = measuredIterations;
            return this;
        }

        // Builds the immutable result
        public BenchmarkResult build() {
            if (vectorCount <= 0) {
                throw new IllegalStateException("vectorCount must be positive");
            }
            if (dimensions <= 0) {
                throw new IllegalStateException("dimensions must be positive");
            }
            return new BenchmarkResult(this);
        }
    }
}
