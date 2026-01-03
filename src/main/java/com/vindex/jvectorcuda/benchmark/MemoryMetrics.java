package com.vindex.jvectorcuda.benchmark;

/**
 * Tracks memory usage for benchmarks (heap and off-heap).
 */
public class MemoryMetrics {

    private final long heapUsedBytes;
    private final long heapMaxBytes;
    private final long offHeapUsedBytes;
    private final long totalMemoryBytes;

    /**
     * Creates memory metrics snapshot.
     *
     * @param heapUsedBytes   Heap memory used in bytes
     * @param heapMaxBytes    Maximum heap memory in bytes
     * @param offHeapUsedBytes Off-heap memory used in bytes (e.g., GPU memory)
     * @param totalMemoryBytes Total system memory in bytes
     */
    public MemoryMetrics(long heapUsedBytes, long heapMaxBytes, long offHeapUsedBytes, long totalMemoryBytes) {
        this.heapUsedBytes = heapUsedBytes;
        this.heapMaxBytes = heapMaxBytes;
        this.offHeapUsedBytes = offHeapUsedBytes;
        this.totalMemoryBytes = totalMemoryBytes;
    }

    /**
     * Captures current memory usage from JVM.
     *
     * @param offHeapUsedBytes Off-heap memory used (e.g., GPU memory)
     * @return Memory metrics snapshot
     */
    public static MemoryMetrics capture(long offHeapUsedBytes) {
        Runtime runtime = Runtime.getRuntime();
        long heapUsed = runtime.totalMemory() - runtime.freeMemory();
        long heapMax = runtime.maxMemory();
        long totalMemory = runtime.totalMemory();

        return new MemoryMetrics(heapUsed, heapMax, offHeapUsedBytes, totalMemory);
    }

    /**
     * @return Heap memory used in bytes
     */
    public long getHeapUsedBytes() {
        return heapUsedBytes;
    }

    /**
     * @return Heap memory used in MB
     */
    public double getHeapUsedMB() {
        return heapUsedBytes / (1024.0 * 1024.0);
    }

    /**
     * @return Maximum heap memory in bytes
     */
    public long getHeapMaxBytes() {
        return heapMaxBytes;
    }

    /**
     * @return Maximum heap memory in MB
     */
    public double getHeapMaxMB() {
        return heapMaxBytes / (1024.0 * 1024.0);
    }

    /**
     * @return Off-heap memory used in bytes (e.g., GPU memory)
     */
    public long getOffHeapUsedBytes() {
        return offHeapUsedBytes;
    }

    /**
     * @return Off-heap memory used in MB
     */
    public double getOffHeapUsedMB() {
        return offHeapUsedBytes / (1024.0 * 1024.0);
    }

    /**
     * @return Total memory in bytes
     */
    public long getTotalMemoryBytes() {
        return totalMemoryBytes;
    }

    /**
     * @return Total memory in MB
     */
    public double getTotalMemoryMB() {
        return totalMemoryBytes / (1024.0 * 1024.0);
    }

    /**
     * @return Heap usage percentage (0-100)
     */
    public double getHeapUsagePercent() {
        if (heapMaxBytes == 0) {
            return 0.0;
        }
        return (heapUsedBytes * 100.0) / heapMaxBytes;
    }

    @Override
    public String toString() {
        return String.format(
            "MemoryMetrics[heap=%.2f/%.2f MB (%.1f%%), offHeap=%.2f MB, total=%.2f MB]",
            getHeapUsedMB(), getHeapMaxMB(), getHeapUsagePercent(),
            getOffHeapUsedMB(), getTotalMemoryMB());
    }
}
