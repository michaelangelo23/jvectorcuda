package com.vindex.jvectorcuda.benchmark;

import java.util.Arrays;

/**
 * Calculates percentile metrics from a collection of measurements.
 * Used for latency analysis (P50, P95, P99).
 */
public class PercentileMetrics {

    private final double[] sortedValues;
    private final double p50;
    private final double p95;
    private final double p99;
    private final double min;
    private final double max;
    private final double mean;

    /**
     * Creates percentile metrics from raw measurements.
     *
     * @param measurements Raw timing measurements in milliseconds
     */
    public PercentileMetrics(double[] measurements) {
        if (measurements == null || measurements.length == 0) {
            throw new IllegalArgumentException("measurements cannot be null or empty");
        }

        this.sortedValues = measurements.clone();
        Arrays.sort(sortedValues);

        this.p50 = calculatePercentile(50);
        this.p95 = calculatePercentile(95);
        this.p99 = calculatePercentile(99);
        this.min = sortedValues[0];
        this.max = sortedValues[sortedValues.length - 1];
        this.mean = calculateMean();
    }

    private double calculatePercentile(int percentile) {
        if (percentile < 0 || percentile > 100) {
            throw new IllegalArgumentException("percentile must be between 0 and 100");
        }

        double index = (percentile / 100.0) * (sortedValues.length - 1);
        int lowerIndex = (int) Math.floor(index);
        int upperIndex = (int) Math.ceil(index);

        if (lowerIndex == upperIndex) {
            return sortedValues[lowerIndex];
        }

        double lowerValue = sortedValues[lowerIndex];
        double upperValue = sortedValues[upperIndex];
        double fraction = index - lowerIndex;

        return lowerValue + (upperValue - lowerValue) * fraction;
    }

    private double calculateMean() {
        double sum = 0.0;
        for (double value : sortedValues) {
            sum += value;
        }
        return sum / sortedValues.length;
    }

    /**
     * @return Median (50th percentile) latency in milliseconds
     */
    public double getP50() {
        return p50;
    }

    /**
     * @return 95th percentile latency in milliseconds
     */
    public double getP95() {
        return p95;
    }

    /**
     * @return 99th percentile latency in milliseconds
     */
    public double getP99() {
        return p99;
    }

    /**
     * @return Minimum latency in milliseconds
     */
    public double getMin() {
        return min;
    }

    /**
     * @return Maximum latency in milliseconds
     */
    public double getMax() {
        return max;
    }

    /**
     * @return Mean latency in milliseconds
     */
    public double getMean() {
        return mean;
    }

    /**
     * @return Number of measurements
     */
    public int getSampleCount() {
        return sortedValues.length;
    }

    @Override
    public String toString() {
        return String.format(
            "PercentileMetrics[samples=%d, min=%.2f, p50=%.2f, p95=%.2f, p99=%.2f, max=%.2f, mean=%.2f]",
            getSampleCount(), min, p50, p95, p99, max, mean);
    }
}
