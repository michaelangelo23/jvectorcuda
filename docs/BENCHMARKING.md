# Benchmarking Guide

This guide explains how to run benchmarks and analyze performance of JVectorCUDA.

## Table of Contents

- [Quick Start](#quick-start)
- [Standard Benchmark Suite](#standard-benchmark-suite)
- [Custom Benchmarks](#custom-benchmarks)
- [Interpreting Results](#interpreting-results)
- [Exporting Results](#exporting-results)
- [Best Practices](#best-practices)

## Quick Start

Run the built-in benchmark suite:

```bash
./gradlew benchmark
```

This will run standard benchmarks and generate:
- `benchmarkTests/benchmark-report.md` - Human-readable report
- `benchmarkTests/benchmark-results-TIMESTAMP.csv` - CSV for regression tracking
- `benchmarkTests/benchmark-results-TIMESTAMP.json` - JSON for programmatic analysis

## Standard Benchmark Suite

The `StandardBenchmarkSuite` provides reproducible benchmarks with comprehensive metrics.

### Using StandardBenchmarkSuite

```java
import com.vindex.jvectorcuda.benchmark.StandardBenchmarkSuite;
import com.vindex.jvectorcuda.cpu.CPUVectorIndex;

// Create benchmark suite
StandardBenchmarkSuite suite = new StandardBenchmarkSuite();

// Create dataset (1000 vectors, 384 dimensions)
StandardBenchmarkSuite.Dataset dataset = 
    StandardBenchmarkSuite.createSyntheticDataset(1000, 384, 100, 42L);

// Create index to benchmark
CPUVectorIndex index = new CPUVectorIndex(384);

// Run benchmark
StandardBenchmarkSuite.ComprehensiveBenchmarkResult result = 
    suite.run(index, dataset, 10);

// Print results
System.out.println(result);
```

### Standard Datasets

The suite includes pre-configured datasets:

- **Small** (1,000 vectors × 384 dims): Quick iteration testing
- **Medium** (10,000 vectors × 384 dims): Development benchmarks
- **Large** (100,000 vectors × 384 dims): Pre-production testing

Access them via:

```java
StandardBenchmarkSuite.Dataset[] datasets = StandardBenchmarkSuite.STANDARD_DATASETS;
```

## Metrics Collected

### Throughput
- **Queries Per Second (QPS)**: Total queries processed per second
- Higher is better

### Latency Percentiles
- **P50 (Median)**: 50% of queries complete faster than this
- **P95**: 95% of queries complete faster than this
- **P99**: 99% of queries complete faster than this
- Lower is better

### Memory Usage
- **Heap Used**: JVM heap memory consumption
- **Heap Max**: Maximum heap memory available
- **Off-Heap**: GPU memory usage (for GPU indices)

### Build Time
- Time taken to add vectors to the index
- Measured in milliseconds

## Custom Benchmarks

### Creating Custom Datasets

```java
// Create synthetic dataset with specific parameters
StandardBenchmarkSuite.Dataset customDataset = 
    StandardBenchmarkSuite.createSyntheticDataset(
        50000,     // vector count
        768,       // dimensions
        200,       // query count
        12345L     // random seed for reproducibility
    );
```

### Benchmarking Different Configurations

```java
import com.vindex.jvectorcuda.benchmark.StandardBenchmarkSuite;
import com.vindex.jvectorcuda.cpu.CPUVectorIndex;
import com.vindex.jvectorcuda.gpu.GPUVectorIndex;
import com.vindex.jvectorcuda.DistanceMetric;

StandardBenchmarkSuite suite = new StandardBenchmarkSuite();
StandardBenchmarkSuite.Dataset dataset = /* ... */;

// Benchmark CPU
CPUVectorIndex cpuIndex = new CPUVectorIndex(384);
var cpuResult = suite.run(cpuIndex, dataset, 10);

// Benchmark GPU (if available)
GPUVectorIndex gpuIndex = new GPUVectorIndex(384, DistanceMetric.EUCLIDEAN);
var gpuResult = suite.run(gpuIndex, dataset, 10);

// Compare results
System.out.printf("CPU QPS: %.2f%n", cpuResult.getThroughputQPS());
System.out.printf("GPU QPS: %.2f%n", gpuResult.getThroughputQPS());
System.out.printf("Speedup: %.2fx%n", 
    gpuResult.getThroughputQPS() / cpuResult.getThroughputQPS());
```

## Interpreting Results

### Example Output

```
BenchmarkResult[dataset=Synthetic[1,000×384], vectors=1,000, dims=384, 
                QPS=15234.5, p50=0.06ms, p95=0.12ms, p99=0.18ms]
```

### What to Look For

1. **High QPS**: Indicates good throughput
2. **Low P99**: Indicates consistent performance (no outliers)
3. **Small P95-P50 gap**: Indicates stable latency
4. **Reasonable memory usage**: Should fit in available RAM/VRAM

### GPU vs CPU Performance

GPU is faster when:
- Dataset is large (>10,000 vectors)
- Batch queries (multiple queries at once)
- High dimensionality (>256 dimensions)

CPU is faster when:
- Dataset is small (<5,000 vectors)
- Single queries
- Low latency requirements

## Exporting Results

### Export to CSV

```java
import com.vindex.jvectorcuda.benchmark.BenchmarkResultExporter;
import java.nio.file.Paths;
import java.util.List;

List<StandardBenchmarkSuite.ComprehensiveBenchmarkResult> results = /* ... */;

// Export all results to CSV
BenchmarkResultExporter.exportToCSV(
    results, 
    Paths.get("benchmark-results.csv")
);
```

CSV format includes:
- dataset, vectors, dimensions, k, metric
- throughput_qps
- latency_p50_ms, latency_p95_ms, latency_p99_ms
- build_time_ms, recall_at_10
- heap_used_mb, heap_max_mb, offheap_mb
- timestamp

### Export to JSON

```java
BenchmarkResultExporter.exportToJSON(
    results, 
    Paths.get("benchmark-results.json")
);
```

### Append Results

For continuous benchmarking, append results over time:

```java
// Append single result to existing CSV
BenchmarkResultExporter.appendToCSV(result, Paths.get("benchmark-log.csv"));
```

This is useful for:
- Tracking performance over time
- Regression detection
- A/B testing different configurations

## Best Practices

### 1. Use Consistent Datasets

Always use the same dataset for before/after comparisons:

```java
// Use fixed seed for reproducibility
long seed = 42L;
var dataset = StandardBenchmarkSuite.createSyntheticDataset(10000, 384, 100, seed);
```

### 2. Warm Up the JVM

The benchmark suite includes warmup iterations (5 by default), but for manual benchmarks:

```java
// Warmup phase
for (int i = 0; i < 5; i++) {
    for (var query : queries) {
        index.search(query, 10);
    }
}

// Now measure
long start = System.nanoTime();
// ... run benchmark
long elapsed = System.nanoTime() - start;
```

### 3. Run Multiple Iterations

Single measurements can be noisy. Run multiple iterations:

```java
double[] latencies = new double[100];
for (int i = 0; i < 100; i++) {
    long start = System.nanoTime();
    index.search(query, 10);
    latencies[i] = (System.nanoTime() - start) / 1_000_000.0;
}

PercentileMetrics metrics = new PercentileMetrics(latencies);
System.out.printf("P50: %.2fms, P95: %.2fms, P99: %.2fms%n",
    metrics.getP50(), metrics.getP95(), metrics.getP99());
```

### 4. Monitor System Resources

Ensure your system is not under load during benchmarking:
- Close unnecessary applications
- Disable background processes
- Monitor CPU/GPU usage

### 5. Document Your Environment

Always record:
- Hardware (CPU, GPU model)
- OS and version
- Java version
- CUDA version
- JCuda version
- Dataset characteristics

### 6. Use Percentiles, Not Averages

P95 and P99 are more meaningful than averages for understanding tail latency:

```java
// Good: Shows latency distribution
System.out.printf("Latency - P50: %.2fms, P95: %.2fms, P99: %.2fms%n",
    metrics.getP50(), metrics.getP95(), metrics.getP99());

// Less useful: Average can hide outliers
System.out.printf("Average latency: %.2fms%n", metrics.getMean());
```

## Comparing Before/After Changes

### 1. Baseline Benchmark

Before making changes:

```bash
./gradlew benchmark
# Results saved to benchmarkTests/
cp benchmarkTests/benchmark-results-*.csv benchmarkTests/baseline.csv
```

### 2. After Changes

After your optimization:

```bash
./gradlew benchmark
# Compare new results with baseline
```

### 3. Compare Results

```bash
# Using any CSV comparison tool or spreadsheet
diff benchmarkTests/baseline.csv benchmarkTests/benchmark-results-*.csv
```

Or programmatically:

```java
var baseline = /* load baseline results */;
var optimized = /* load optimized results */;

double speedup = optimized.getThroughputQPS() / baseline.getThroughputQPS();
System.out.printf("Throughput improvement: %.2fx%n", speedup);

double latencyImprovement = baseline.getLatency().getP95() / optimized.getLatency().getP95();
System.out.printf("P95 latency improvement: %.2fx%n", latencyImprovement);
```

## Example: Full Benchmark Workflow

```java
import com.vindex.jvectorcuda.benchmark.*;
import com.vindex.jvectorcuda.cpu.CPUVectorIndex;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class ComprehensiveBenchmark {
    public static void main(String[] args) throws Exception {
        StandardBenchmarkSuite suite = new StandardBenchmarkSuite();
        List<StandardBenchmarkSuite.ComprehensiveBenchmarkResult> results = new ArrayList<>();
        
        // Test different dataset sizes
        int[] vectorCounts = {1000, 10000, 100000};
        
        for (int vectorCount : vectorCounts) {
            var dataset = StandardBenchmarkSuite.createSyntheticDataset(
                vectorCount, 384, 100, 42L
            );
            
            CPUVectorIndex index = new CPUVectorIndex(384);
            var result = suite.run(index, dataset, 10);
            results.add(result);
            index.close();
            
            System.out.println(result);
        }
        
        // Export to CSV
        BenchmarkResultExporter.exportToCSV(results, Paths.get("results.csv"));
        
        // Export to JSON
        BenchmarkResultExporter.exportToJSON(results, Paths.get("results.json"));
        
        System.out.println("Benchmark complete! Results saved to results.csv and results.json");
    }
}
```

## Troubleshooting

### High Variance in Results

If you see inconsistent results:
- Ensure system is idle
- Increase warmup iterations
- Run more measurement iterations
- Check for thermal throttling

### Out of Memory Errors

For large datasets:
- Reduce dataset size
- Increase JVM heap: `-Xmx8g`
- Use batch processing
- Monitor memory with `MemoryMetrics`

### GPU Not Available

If GPU benchmarks fail:
```bash
./gradlew test --tests CudaAvailabilityTest
```

Check diagnostics:
```java
import com.vindex.jvectorcuda.diagnostics.GPUDiagnostics;

GPUDiagnostics.printDiagnostics();
```

## See Also

- [README.md](README.md) - General usage
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guide
- [API Documentation](https://javadoc.io/doc/io.github.michaelangelo23/jvectorcuda)
