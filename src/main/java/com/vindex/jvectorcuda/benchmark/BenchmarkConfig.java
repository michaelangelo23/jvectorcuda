package com.vindex.jvectorcuda.benchmark;

import com.vindex.jvectorcuda.DistanceMetric;

import java.util.Objects;

// Immutable benchmark configuration. Use builder() to create.
public final class BenchmarkConfig {

    // Quick benchmark defaults
    public static final BenchmarkConfig DEFAULT = builder()
        .vectorCount(10_000)
        .dimensions(384)
        .queryCount(10)
        .k(10)
        .warmupIterations(3)
        .measuredIterations(10)
        .build();

    // Comprehensive benchmark config
    public static final BenchmarkConfig COMPREHENSIVE = builder()
        .vectorCount(100_000)
        .dimensions(384)
        .queryCount(100)
        .k(10)
        .warmupIterations(5)
        .measuredIterations(50)
        .build();

    // Stress test config
    public static final BenchmarkConfig STRESS_TEST = builder()
        .vectorCount(500_000)
        .dimensions(384)
        .queryCount(1000)
        .k(100)
        .warmupIterations(10)
        .measuredIterations(100)
        .build();

    private final int vectorCount;
    private final int dimensions;
    private final int queryCount;
    private final int k;
    private final DistanceMetric distanceMetric;
    private final int warmupIterations;
    private final int measuredIterations;
    private final boolean includeMemoryProfiling;
    private final boolean includeTransferTiming;
    private final long randomSeed;

    private BenchmarkConfig(Builder builder) {
        this.vectorCount = builder.vectorCount;
        this.dimensions = builder.dimensions;
        this.queryCount = builder.queryCount;
        this.k = builder.k;
        this.distanceMetric = builder.distanceMetric;
        this.warmupIterations = builder.warmupIterations;
        this.measuredIterations = builder.measuredIterations;
        this.includeMemoryProfiling = builder.includeMemoryProfiling;
        this.includeTransferTiming = builder.includeTransferTiming;
        this.randomSeed = builder.randomSeed;
    }

    // Creates a new builder
    public static Builder builder() {
        return new Builder();
    }

    // Calculates estimated GPU memory usage in bytes
    public long getEstimatedMemoryBytes() {
        long databaseBytes = (long) vectorCount * dimensions * Float.BYTES;
        long distancesBytes = (long) vectorCount * Float.BYTES;
        long queryBytes = (long) dimensions * Float.BYTES;
        return databaseBytes + distancesBytes + queryBytes;
    }

    // Returns estimated memory in MB
    public double getEstimatedMemoryMB() {
        return getEstimatedMemoryBytes() / (1024.0 * 1024.0);
    }

    // Checks if config fits in GPU memory (with 20% safety margin)
    public boolean fitsInMemory(double availableMemoryMB) {
        double requiredMB = getEstimatedMemoryMB() * 1.2;
        return requiredMB <= availableMemoryMB;
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

    public DistanceMetric getDistanceMetric() {
        return distanceMetric;
    }

    public int getWarmupIterations() {
        return warmupIterations;
    }

    public int getMeasuredIterations() {
        return measuredIterations;
    }

    public boolean isIncludeMemoryProfiling() {
        return includeMemoryProfiling;
    }

    public boolean isIncludeTransferTiming() {
        return includeTransferTiming;
    }

    public long getRandomSeed() {
        return randomSeed;
    }

    @Override
    public String toString() {
        return String.format(
            "BenchmarkConfig[vectors=%,d, dims=%d, queries=%d, k=%d, metric=%s, " +
            "warmup=%d, measured=%d, memory=%.1fMB]",
            vectorCount, dimensions, queryCount, k, distanceMetric,
            warmupIterations, measuredIterations, getEstimatedMemoryMB());
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        BenchmarkConfig that = (BenchmarkConfig) o;
        return vectorCount == that.vectorCount &&
               dimensions == that.dimensions &&
               queryCount == that.queryCount &&
               k == that.k &&
               distanceMetric == that.distanceMetric;
    }

    @Override
    public int hashCode() {
        return Objects.hash(vectorCount, dimensions, queryCount, k, distanceMetric);
    }

    // Builder for BenchmarkConfig
    public static final class Builder {
        private int vectorCount = 10_000;
        private int dimensions = 384;
        private int queryCount = 10;
        private int k = 10;
        private DistanceMetric distanceMetric = DistanceMetric.EUCLIDEAN;
        private int warmupIterations = 3;
        private int measuredIterations = 10;
        private boolean includeMemoryProfiling = true;
        private boolean includeTransferTiming = true;
        private long randomSeed = 42L;

        private Builder() {}

        public Builder vectorCount(int vectorCount) {
            if (vectorCount <= 0) {
                throw new IllegalArgumentException("vectorCount must be positive, got: " + vectorCount);
            }
            this.vectorCount = vectorCount;
            return this;
        }

        public Builder dimensions(int dimensions) {
            if (dimensions <= 0) {
                throw new IllegalArgumentException("dimensions must be positive, got: " + dimensions);
            }
            this.dimensions = dimensions;
            return this;
        }

        public Builder queryCount(int queryCount) {
            if (queryCount <= 0) {
                throw new IllegalArgumentException("queryCount must be positive, got: " + queryCount);
            }
            this.queryCount = queryCount;
            return this;
        }

        public Builder k(int k) {
            if (k <= 0) {
                throw new IllegalArgumentException("k must be positive, got: " + k);
            }
            this.k = k;
            return this;
        }

        public Builder distanceMetric(DistanceMetric distanceMetric) {
            this.distanceMetric = Objects.requireNonNull(distanceMetric, "distanceMetric cannot be null");
            return this;
        }

        public Builder warmupIterations(int warmupIterations) {
            if (warmupIterations < 0) {
                throw new IllegalArgumentException("warmupIterations cannot be negative, got: " + warmupIterations);
            }
            this.warmupIterations = warmupIterations;
            return this;
        }

        public Builder measuredIterations(int measuredIterations) {
            if (measuredIterations <= 0) {
                throw new IllegalArgumentException("measuredIterations must be positive, got: " + measuredIterations);
            }
            this.measuredIterations = measuredIterations;
            return this;
        }

        public Builder includeMemoryProfiling(boolean includeMemoryProfiling) {
            this.includeMemoryProfiling = includeMemoryProfiling;
            return this;
        }

        public Builder includeTransferTiming(boolean includeTransferTiming) {
            this.includeTransferTiming = includeTransferTiming;
            return this;
        }

        public Builder randomSeed(long randomSeed) {
            this.randomSeed = randomSeed;
            return this;
        }

        // Builds the immutable config
        public BenchmarkConfig build() {
            return new BenchmarkConfig(this);
        }
    }
}
