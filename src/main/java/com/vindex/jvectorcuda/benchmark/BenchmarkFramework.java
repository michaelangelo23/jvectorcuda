package com.vindex.jvectorcuda.benchmark;

import com.vindex.jvectorcuda.CudaDetector;
import com.vindex.jvectorcuda.cpu.CPUVectorIndex;
import com.vindex.jvectorcuda.gpu.GPUVectorIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

// CPU vs GPU benchmark framework with timing and memory analysis.
public class BenchmarkFramework {

    private static final Logger logger = LoggerFactory.getLogger(BenchmarkFramework.class);

    private final String gpuName;
    private final boolean gpuAvailable;

    // Auto-detects GPU availability
    public BenchmarkFramework() {
        this.gpuAvailable = CudaDetector.isAvailable();
        this.gpuName = gpuAvailable ? extractGpuName(CudaDetector.getGpuInfo()) : "No GPU";
        
        logger.info("BenchmarkFramework initialized. GPU: {} (available: {})", gpuName, gpuAvailable);
    }

    // Extracts GPU name from info string (format: "GPU: [name], Compute: X.Y, Memory: Z MB")
    private String extractGpuName(String gpuInfo) {
        if (gpuInfo == null || gpuInfo.isEmpty()) {
            return "Unknown GPU";
        }
        // Extract name from format "GPU: [name], Compute: ..."
        if (gpuInfo.startsWith("GPU: ")) {
            int commaIndex = gpuInfo.indexOf(',');
            if (commaIndex > 5) {
                return gpuInfo.substring(5, commaIndex).trim();
            }
        }
        return gpuInfo;
    }

    // Runs a single benchmark with given config
    public BenchmarkResult run(BenchmarkConfig config) {
        logger.info("Starting benchmark: {}", config);
        
        // Generate test data
        float[][] database = generateRandomVectors(
            config.getVectorCount(), 
            config.getDimensions(), 
            config.getRandomSeed()
        );
        float[][] queries = generateRandomVectors(
            config.getQueryCount(), 
            config.getDimensions(), 
            config.getRandomSeed() + 1
        );
        
        // Run CPU benchmark
        double cpuTimeMs = benchmarkCpu(database, queries, config);
        
        // Run GPU benchmark
        GpuBenchmarkResult gpuResult = benchmarkGpu(database, queries, config);
        
        // Calculate memory usage
        long gpuMemoryBytes = estimateGpuMemory(config);
        
        return BenchmarkResult.builder()
            .cpuTimeMs(cpuTimeMs)
            .gpuTimeMs(gpuResult.totalTimeMs)
            .gpuTransferTimeMs(gpuResult.transferTimeMs)
            .gpuComputeTimeMs(gpuResult.computeTimeMs)
            .gpuMemoryUsedBytes(gpuMemoryBytes)
            .vectorCount(config.getVectorCount())
            .dimensions(config.getDimensions())
            .queryCount(config.getQueryCount())
            .k(config.getK())
            .distanceMetric(config.getDistanceMetric().name())
            .timestamp(Instant.now())
            .gpuName(gpuName)
            .warmupIterations(config.getWarmupIterations())
            .measuredIterations(config.getMeasuredIterations())
            .build();
    }

    // Runs benchmarks across multiple vector counts and dimensions
    public List<BenchmarkResult> runSuite(List<Integer> vectorCounts, List<Integer> dimensionsList) {
        List<BenchmarkResult> results = new ArrayList<>();
        
        for (int vectorCount : vectorCounts) {
            for (int dimensions : dimensionsList) {
                BenchmarkConfig config = BenchmarkConfig.builder()
                    .vectorCount(vectorCount)
                    .dimensions(dimensions)
                    .queryCount(100)
                    .k(10)
                    .warmupIterations(3)
                    .measuredIterations(10)
                    .build();
                
                try {
                    BenchmarkResult result = run(config);
                    results.add(result);
                    logger.info("Completed: vectors={}, dims={}, speedup={}x",
                        vectorCount, dimensions, String.format("%.2f", result.getSpeedup()));
                } catch (Exception e) {
                    logger.error("Benchmark failed for vectors={}, dims={}: {}",
                        vectorCount, dimensions, e.getMessage());
                }
            }
        }
        
        return results;
    }

    // Upload once, query many times benchmark (demonstrates 5x+ speedup)
    public BenchmarkResult runPersistentMemoryBenchmark(BenchmarkConfig config) {
        logger.info("Starting persistent memory benchmark: {}", config);
        
        float[][] database = generateRandomVectors(
            config.getVectorCount(),
            config.getDimensions(),
            config.getRandomSeed()
        );
        float[][] queries = generateRandomVectors(
            config.getQueryCount(),
            config.getDimensions(),
            config.getRandomSeed() + 1
        );
        
        // CPU: Run all queries
        double cpuTimeMs = benchmarkCpuPersistent(database, queries, config);
        
        // GPU: Upload once, run all queries
        GpuBenchmarkResult gpuResult = benchmarkGpuPersistent(database, queries, config);
        
        return BenchmarkResult.builder()
            .cpuTimeMs(cpuTimeMs)
            .gpuTimeMs(gpuResult.totalTimeMs)
            .gpuTransferTimeMs(gpuResult.transferTimeMs)
            .gpuComputeTimeMs(gpuResult.computeTimeMs)
            .gpuMemoryUsedBytes(estimateGpuMemory(config))
            .vectorCount(config.getVectorCount())
            .dimensions(config.getDimensions())
            .queryCount(config.getQueryCount())
            .k(config.getK())
            .distanceMetric(config.getDistanceMetric().name())
            .timestamp(Instant.now())
            .gpuName(gpuName)
            .warmupIterations(config.getWarmupIterations())
            .measuredIterations(config.getMeasuredIterations())
            .build();
    }

    // Prints formatted comparison table to console
    public void printComparisonTable(List<BenchmarkResult> results) {
        System.out.println("\n=== JVectorCUDA Benchmark Results ===\n");
        System.out.printf("%-12s %-8s %-10s %-10s %-10s %-8s %-12s%n",
            "Vectors", "Dims", "CPU (ms)", "GPU (ms)", "Speedup", "Winner", "GPU Memory");
        System.out.println("-".repeat(80));
        
        for (BenchmarkResult result : results) {
            String winner = result.isGpuFaster() ? "GPU" : "CPU";
            System.out.printf("%-12s %-8d %-10.2f %-10.2f %-10.2fx %-8s %-12.1f MB%n",
                formatNumber(result.getVectorCount()),
                result.getDimensions(),
                result.getCpuTimeMs(),
                result.getGpuTimeMs(),
                result.getSpeedup(),
                winner,
                result.getGpuMemoryUsedMB());
        }
        
        System.out.println("\nNote: GPU speedup > 1.0x means GPU is faster than CPU.");
        System.out.println("For persistent memory scenarios (many queries, same dataset), expect 5x+ speedup.");
    }

    private double benchmarkCpu(float[][] database, float[][] queries, BenchmarkConfig config) {
        try (CPUVectorIndex index = new CPUVectorIndex(config.getDimensions())) {
            index.add(database);
            
            // Warmup
            for (int i = 0; i < config.getWarmupIterations(); i++) {
                for (float[] query : queries) {
                    index.search(query, config.getK());
                }
            }
            
            // Measured runs
            long startTime = System.nanoTime();
            for (int i = 0; i < config.getMeasuredIterations(); i++) {
                for (float[] query : queries) {
                    index.search(query, config.getK());
                }
            }
            long elapsed = System.nanoTime() - startTime;
            
            return elapsed / 1_000_000.0 / config.getMeasuredIterations();
        }
    }

    private GpuBenchmarkResult benchmarkGpu(float[][] database, float[][] queries, BenchmarkConfig config) {
        if (!gpuAvailable) {
            return new GpuBenchmarkResult(0, 0, 0);
        }
        
        double totalTimeMs = 0;
        double transferTimeMs = 0;
        
        // For non-persistent benchmark, create fresh index each iteration
        for (int iter = 0; iter < config.getWarmupIterations() + config.getMeasuredIterations(); iter++) {
            boolean isMeasured = iter >= config.getWarmupIterations();
            
            long iterStart = System.nanoTime();
            long uploadStart = System.nanoTime();
            
            try (GPUVectorIndex index = new GPUVectorIndex(config.getDimensions(), config.getDistanceMetric())) {
                index.add(database);
                long uploadEnd = System.nanoTime();
                
                for (float[] query : queries) {
                    index.search(query, config.getK());
                }
                
                long iterEnd = System.nanoTime();
                
                if (isMeasured) {
                    totalTimeMs += (iterEnd - iterStart) / 1_000_000.0;
                    transferTimeMs += (uploadEnd - uploadStart) / 1_000_000.0;
                }
            }
        }
        
        int measuredCount = config.getMeasuredIterations();
        return new GpuBenchmarkResult(
            totalTimeMs / measuredCount,
            transferTimeMs / measuredCount,
            (totalTimeMs - transferTimeMs) / measuredCount
        );
    }

    private double benchmarkCpuPersistent(float[][] database, float[][] queries, BenchmarkConfig config) {
        try (CPUVectorIndex index = new CPUVectorIndex(config.getDimensions())) {
            index.add(database);
            
            // Warmup
            for (int i = 0; i < config.getWarmupIterations(); i++) {
                for (float[] query : queries) {
                    index.search(query, config.getK());
                }
            }
            
            // Measured: all queries in a single block
            long startTime = System.nanoTime();
            for (int i = 0; i < config.getMeasuredIterations(); i++) {
                for (float[] query : queries) {
                    index.search(query, config.getK());
                }
            }
            long elapsed = System.nanoTime() - startTime;
            
            return elapsed / 1_000_000.0 / config.getMeasuredIterations();
        }
    }

    private GpuBenchmarkResult benchmarkGpuPersistent(float[][] database, float[][] queries, BenchmarkConfig config) {
        if (!gpuAvailable) {
            return new GpuBenchmarkResult(0, 0, 0);
        }
        
        try (GPUVectorIndex index = new GPUVectorIndex(config.getDimensions(), config.getDistanceMetric())) {
            // Upload once
            long uploadStart = System.nanoTime();
            index.add(database);
            long uploadEnd = System.nanoTime();
            double uploadTimeMs = (uploadEnd - uploadStart) / 1_000_000.0;
            
            // Warmup
            for (int i = 0; i < config.getWarmupIterations(); i++) {
                for (float[] query : queries) {
                    index.search(query, config.getK());
                }
            }
            
            // Measured: many queries, same dataset
            long startTime = System.nanoTime();
            for (int i = 0; i < config.getMeasuredIterations(); i++) {
                for (float[] query : queries) {
                    index.search(query, config.getK());
                }
            }
            long elapsed = System.nanoTime() - startTime;
            double queryTimeMs = elapsed / 1_000_000.0 / config.getMeasuredIterations();
            
            // In persistent mode, transfer is amortized
            double amortizedTransfer = uploadTimeMs / config.getMeasuredIterations();
            
            return new GpuBenchmarkResult(
                queryTimeMs + amortizedTransfer,
                amortizedTransfer,
                queryTimeMs
            );
        }
    }

    private float[][] generateRandomVectors(int count, int dimensions, long seed) {
        Random random = new Random(seed);
        float[][] vectors = new float[count][dimensions];
        
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < dimensions; j++) {
                vectors[i][j] = random.nextFloat() * 2.0f - 1.0f;
            }
        }
        
        return vectors;
    }

    // Database + distances + query = total GPU memory
    private long estimateGpuMemory(BenchmarkConfig config) {
        return (long) config.getVectorCount() * config.getDimensions() * Float.BYTES +
               (long) config.getVectorCount() * Float.BYTES +
               (long) config.getDimensions() * Float.BYTES;
    }

    private String formatNumber(int number) {
        if (number >= 1_000_000) {
            return String.format("%.1fM", number / 1_000_000.0);
        } else if (number >= 1_000) {
            return String.format("%.1fK", number / 1_000.0);
        }
        return String.valueOf(number);
    }

    // Returns true if GPU is available
    public boolean isGpuAvailable() {
        return gpuAvailable;
    }

    // Returns GPU name or "No GPU"
    public String getGpuName() {
        return gpuName;
    }

    private static class GpuBenchmarkResult {
        final double totalTimeMs;
        final double transferTimeMs;
        final double computeTimeMs;

        GpuBenchmarkResult(double totalTimeMs, double transferTimeMs, double computeTimeMs) {
            this.totalTimeMs = totalTimeMs;
            this.transferTimeMs = transferTimeMs;
            this.computeTimeMs = computeTimeMs;
        }
    }
}
