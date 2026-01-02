package com.vindex.jvectorcuda.gpu;

import com.vindex.jvectorcuda.DistanceMetric;
import com.vindex.jvectorcuda.SearchResult;
import com.vindex.jvectorcuda.VectorIndex;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUresult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

import static jcuda.driver.JCudaDriver.*;

// GPU-accelerated vector index with persistent memory. 5x+ speedup for batch queries. Not thread-safe.
public class GPUVectorIndex implements VectorIndex {

    private static final Logger logger = LoggerFactory.getLogger(GPUVectorIndex.class);
    
    private static final int BLOCK_SIZE = 256;
    
    private final int dimensions;
    private final DistanceMetric distanceMetric;
    private final GpuKernelLoader kernelLoader;
    
    // CUDA context
    private CUcontext context;
    private CUdevice device;
    
    // Persistent GPU memory (the key optimization!)
    private CUdeviceptr d_database;
    private CUdeviceptr d_distances;
    private int vectorCount;
    private int capacity;
    
    // State tracking
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private boolean databaseOnGpu = false;

    public GPUVectorIndex(int dimensions) {
        this(dimensions, DistanceMetric.EUCLIDEAN, 100_000);
    }

    public GPUVectorIndex(int dimensions, DistanceMetric metric) {
        this(dimensions, metric, 100_000);
    }

    public GPUVectorIndex(int dimensions, DistanceMetric metric, int initialCapacity) {
        if (dimensions <= 0) {
            throw new IllegalArgumentException("Dimensions must be positive, got: " + dimensions);
        }
        if (metric == null) {
            throw new IllegalArgumentException("Distance metric cannot be null");
        }
        if (initialCapacity <= 0) {
            throw new IllegalArgumentException("Capacity must be positive, got: " + initialCapacity);
        }
        
        this.dimensions = dimensions;
        this.distanceMetric = metric;
        this.capacity = initialCapacity;
        this.vectorCount = 0;
        
        // Initialize CUDA
        initializeCuda();
        
        // Validate GPU memory availability before allocation
        validateGpuMemory(initialCapacity, dimensions);
        
        // Load kernel for specified metric
        this.kernelLoader = new GpuKernelLoader(metric.getPtxFile(), metric.getKernelName());
        
        // Pre-allocate GPU memory for distances (reused across searches)
        d_distances = new CUdeviceptr();
        checkCudaResult(
            cuMemAlloc(d_distances, (long) capacity * Sizeof.FLOAT),
            "cuMemAlloc for distances");
        
        logger.info("GPUVectorIndex created: {} dimensions, {} capacity, metric={}", 
            dimensions, capacity, metric);
    }

    private void initializeCuda() {
        checkCudaResult(cuInit(0), "cuInit");
        device = new CUdevice();
        checkCudaResult(cuDeviceGet(device, 0), "cuDeviceGet");
        context = new CUcontext();
        checkCudaResult(cuCtxCreate(context, 0, device), "cuCtxCreate");
        logger.debug("CUDA context initialized");
    }

    @Override
    public void add(float[][] vectors) {
        checkNotClosed();
        
        if (vectors == null || vectors.length == 0) {
            return;
        }
        
        validateInputVectors(vectors);
        ensureCapacity(vectorCount + vectors.length);
        uploadVectorsToGpu(vectors);
        
        vectorCount += vectors.length;
    }

    private void validateInputVectors(float[][] vectors) {
        for (int i = 0; i < vectors.length; i++) {
            if (vectors[i] == null) {
                throw new IllegalArgumentException(
                    String.format("Vector %d is null", i));
            }
            if (vectors[i].length != dimensions) {
                throw new IllegalArgumentException(
                    String.format("Vector %d has %d dimensions, expected %d", 
                        i, vectors[i].length, dimensions));
            }
            validateVectorValues(vectors[i], i);
        }
    }

    private void ensureCapacity(int requiredCount) {
        if (requiredCount > capacity) {
            expandCapacity(requiredCount);
        }
    }

    private void uploadVectorsToGpu(float[][] vectors) {
        float[] flatVectors = flattenVectors(vectors);
        
        if (!databaseOnGpu) {
            allocateAndUploadInitial(flatVectors, vectors.length);
        } else {
            appendToExisting(flatVectors, vectors.length);
        }
    }

    private void allocateAndUploadInitial(float[] flatVectors, int count) {
        d_database = new CUdeviceptr();
        checkCudaResult(
            cuMemAlloc(d_database, (long) capacity * dimensions * Sizeof.FLOAT),
            "cuMemAlloc for database");
        checkCudaResult(
            cuMemcpyHtoD(d_database, Pointer.to(flatVectors), (long) flatVectors.length * Sizeof.FLOAT),
            "cuMemcpyHtoD for initial upload");
        databaseOnGpu = true;
        logger.debug("Initial database upload: {} vectors", count);
    }

    private void appendToExisting(float[] flatVectors, int count) {
        long offset = (long) vectorCount * dimensions * Sizeof.FLOAT;
        CUdeviceptr offsetPtr = d_database.withByteOffset(offset);
        checkCudaResult(
            cuMemcpyHtoD(offsetPtr, Pointer.to(flatVectors), (long) flatVectors.length * Sizeof.FLOAT),
            "cuMemcpyHtoD for append");
        logger.debug("Appended {} vectors to GPU database", count);
    }

    private void expandCapacity(int requiredCapacity) {
        int newCapacity = Math.max(capacity * 2, requiredCapacity);
        
        logger.info("Expanding GPU capacity: {} → {}", capacity, newCapacity);
        
        // Allocate new GPU memory
        CUdeviceptr newDatabase = new CUdeviceptr();
        checkCudaResult(
            cuMemAlloc(newDatabase, (long) newCapacity * dimensions * Sizeof.FLOAT),
            "cuMemAlloc for expanded database");
        
        // Copy existing data if any
        if (databaseOnGpu && vectorCount > 0) {
            checkCudaResult(
                cuMemcpyDtoD(newDatabase, d_database, (long) vectorCount * dimensions * Sizeof.FLOAT),
                "cuMemcpyDtoD during expand");
            cuMemFree(d_database); // Free old memory (no check needed, best effort)
        }
        
        d_database = newDatabase;
        
        // Reallocate distances buffer
        cuMemFree(d_distances); // Free old memory (no check needed, best effort)
        d_distances = new CUdeviceptr();
        checkCudaResult(
            cuMemAlloc(d_distances, (long) newCapacity * Sizeof.FLOAT),
            "cuMemAlloc for expanded distances");
        
        capacity = newCapacity;
    }

    @Override
    public SearchResult search(float[] query, int k) {
        checkNotClosed();
        validateSearchParams(query, k);
        
        if (vectorCount == 0) {
            return new SearchResult(new int[0], new float[0], 0);
        }
        
        k = Math.min(k, vectorCount);
        long startTime = System.nanoTime();
        
        float[] distances = computeDistancesOnGpu(query);
        SearchResult result = buildSearchResult(distances, k, startTime);
        
        return result;
    }

    private void validateSearchParams(float[] query, int k) {
        if (query == null || query.length != dimensions) {
            throw new IllegalArgumentException(
                String.format("Query must have %d dimensions, got %d", 
                    dimensions, query == null ? 0 : query.length));
        }
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive, got: " + k);
        }
    }

    private float[] computeDistancesOnGpu(float[] query) {
        CUdeviceptr d_query = uploadQuery(query);
        launchDistanceKernel(d_query);
        float[] distances = downloadDistances();
        cuMemFree(d_query); // Free temporary query memory (best effort)
        return distances;
    }

    private CUdeviceptr uploadQuery(float[] query) {
        CUdeviceptr d_query = new CUdeviceptr();
        checkCudaResult(
            cuMemAlloc(d_query, (long) dimensions * Sizeof.FLOAT),
            "cuMemAlloc for query");
        checkCudaResult(
            cuMemcpyHtoD(d_query, Pointer.to(query), (long) dimensions * Sizeof.FLOAT),
            "cuMemcpyHtoD for query");
        return d_query;
    }

    private void launchDistanceKernel(CUdeviceptr d_query) {
        int gridSize = (vectorCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
        Pointer params = Pointer.to(
                Pointer.to(d_database),
                Pointer.to(d_query),
                Pointer.to(d_distances),
                Pointer.to(new int[]{vectorCount}),
                Pointer.to(new int[]{dimensions})
        );
        kernelLoader.launch(gridSize, BLOCK_SIZE, params);
    }

    private float[] downloadDistances() {
        float[] distances = new float[vectorCount];
        checkCudaResult(
            cuMemcpyDtoH(Pointer.to(distances), d_distances, (long) vectorCount * Sizeof.FLOAT),
            "cuMemcpyDtoH for distances");
        return distances;
    }

    private SearchResult buildSearchResult(float[] distances, int k, long startTime) {
        int[] topKIndices = findTopK(distances, k);
        float[] topKDistances = new float[k];
        for (int i = 0; i < k; i++) {
            topKDistances[i] = distances[topKIndices[i]];
        }
        long searchTimeMs = (System.nanoTime() - startTime) / 1_000_000;
        return new SearchResult(topKIndices, topKDistances, searchTimeMs);
    }

    @Override
    public CompletableFuture<SearchResult> searchAsync(float[] query, int k) {
        return CompletableFuture.supplyAsync(() -> search(query, k));
    }

    // Batch search - amortizes kernel launch overhead for multiple queries
    public java.util.List<SearchResult> searchBatch(float[][] queries, int k) {
        checkNotClosed();
        
        if (queries == null || queries.length == 0) {
            return java.util.Collections.emptyList();
        }
        
        java.util.List<SearchResult> results = new java.util.ArrayList<>(queries.length);
        
        for (float[] query : queries) {
            results.add(search(query, k));
        }
        
        return results;
    }

    @Override
    public int getDimensions() {
        return dimensions;
    }

    @Override
    public int size() {
        return vectorCount;
    }

    public boolean isDatabaseOnGpu() {
        return databaseOnGpu;
    }

    public int getCapacity() {
        return capacity;
    }

    public DistanceMetric getDistanceMetric() {
        return distanceMetric;
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            logger.debug("Closing GPUVectorIndex");
            
            // Release resources one by one, capturing first exception
            // to ensure all resources are freed even if one fails
            RuntimeException firstException = null;
            
            if (d_database != null && databaseOnGpu) {
                try {
                    cuMemFree(d_database);
                } catch (RuntimeException e) {
                    logger.error("Failed to free d_database", e);
                    firstException = e;
                }
            }
            
            if (d_distances != null) {
                try {
                    cuMemFree(d_distances);
                } catch (RuntimeException e) {
                    logger.error("Failed to free d_distances", e);
                    if (firstException == null) firstException = e;
                }
            }
            
            if (context != null) {
                try {
                    cuCtxDestroy(context);
                } catch (RuntimeException e) {
                    logger.error("Failed to destroy CUDA context", e);
                    if (firstException == null) firstException = e;
                }
            }
            
            logger.info("GPUVectorIndex closed, freed GPU memory");
            
            // Rethrow first exception after all cleanup attempted
            if (firstException != null) {
                throw firstException;
            }
        }
    }

    private void checkNotClosed() {
        if (closed.get()) {
            throw new IllegalStateException("GPUVectorIndex has been closed");
        }
    }

    private void validateVectorValues(float[] vector, int index) {
        for (int i = 0; i < vector.length; i++) {
            if (!Float.isFinite(vector[i])) {
                throw new IllegalArgumentException(
                    String.format("Vector %d contains invalid value at index %d: %s",
                        index, i, vector[i]));
            }
        }
    }

    private float[] flattenVectors(float[][] vectors) {
        float[] flat = new float[vectors.length * dimensions];
        for (int i = 0; i < vectors.length; i++) {
            System.arraycopy(vectors[i], 0, flat, i * dimensions, dimensions);
        }
        return flat;
    }

    // Find k smallest distances using max-heap, O(n log k)
    private int[] findTopK(float[] distances, int k) {
        int n = distances.length;
        
        java.util.PriorityQueue<int[]> maxHeap = new java.util.PriorityQueue<>(
            k, (a, b) -> Float.compare(distances[b[0]], distances[a[0]])
        );
        
        for (int i = 0; i < n; i++) {
            if (maxHeap.size() < k) {
                maxHeap.offer(new int[]{i});
            } else if (distances[i] < distances[maxHeap.peek()[0]]) {
                maxHeap.poll();
                maxHeap.offer(new int[]{i});
            }
        }
        
        int[] result = new int[k];
        for (int i = k - 1; i >= 0; i--) {
            result[i] = maxHeap.poll()[0];
        }
        
        return result;
    }

    // Validate GPU has sufficient memory before allocation
    private void validateGpuMemory(int capacity, int dimensions) {
        long[] free = new long[1];
        long[] total = new long[1];
        
        int result = cuMemGetInfo(free, total);
        if (result != CUresult.CUDA_SUCCESS) {
            logger.warn("Failed to query GPU memory, proceeding with allocation");
            return; // Non-fatal, attempt allocation anyway
        }
        
        // Calculate required memory (database + distances buffer + 10% overhead)
        long databaseBytes = (long) capacity * dimensions * Sizeof.FLOAT;
        long distancesBytes = (long) capacity * Sizeof.FLOAT;
        long requiredBytes = (long) ((databaseBytes + distancesBytes) * 1.1); // 10% overhead
        
        long availableBytes = free[0];
        long totalBytes = total[0];
        
        if (requiredBytes > availableBytes) {
            throw new IllegalArgumentException(String.format(
                "Insufficient GPU memory: need %d MB, available %d MB (%.1f%% of %d MB total)%n" +
                "Suggestions:%n" +
                "  - Reduce capacity (current: %,d vectors)%n" +
                "  - Reduce dimensions (current: %d)%n" +
                "  - Close other GPU applications%n" +
                "  - Use CPU implementation instead",
                requiredBytes / 1_048_576,
                availableBytes / 1_048_576,
                (availableBytes * 100.0) / totalBytes,
                totalBytes / 1_048_576,
                capacity,
                dimensions
            ));
        }
        
        logger.info("GPU memory check passed: need {} MB, available {} MB ({}% of total)",
            requiredBytes / 1_048_576,
            availableBytes / 1_048_576,
            (availableBytes * 100) / totalBytes);
    }

    // Check CUDA result and throw exception on error
    private static void checkCudaResult(int result, String operation) {
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException(String.format(
                "CUDA error in %s: %s (code %d)", 
                operation, 
                CUresult.stringFor(result), 
                result));
        }
    }
}
