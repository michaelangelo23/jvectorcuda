package com.vindex.jvectorcuda.gpu;

import com.vindex.jvectorcuda.DistanceMetric;
import com.vindex.jvectorcuda.SearchResult;
import com.vindex.jvectorcuda.VectorIndex;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
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
        
        // Load kernel for specified metric
        this.kernelLoader = new GpuKernelLoader(metric.getPtxFile(), metric.getKernelName());
        
        // Pre-allocate GPU memory for distances (reused across searches)
        d_distances = new CUdeviceptr();
        cuMemAlloc(d_distances, (long) capacity * Sizeof.FLOAT);
        
        logger.info("GPUVectorIndex created: {} dimensions, {} capacity, metric={}", 
            dimensions, capacity, metric);
    }

    private void initializeCuda() {
        cuInit(0);
        device = new CUdevice();
        cuDeviceGet(device, 0);
        context = new CUcontext();
        cuCtxCreate(context, 0, device);
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
        cuMemAlloc(d_database, (long) capacity * dimensions * Sizeof.FLOAT);
        cuMemcpyHtoD(d_database, Pointer.to(flatVectors), (long) flatVectors.length * Sizeof.FLOAT);
        databaseOnGpu = true;
        logger.debug("Initial database upload: {} vectors", count);
    }

    private void appendToExisting(float[] flatVectors, int count) {
        long offset = (long) vectorCount * dimensions * Sizeof.FLOAT;
        CUdeviceptr offsetPtr = d_database.withByteOffset(offset);
        cuMemcpyHtoD(offsetPtr, Pointer.to(flatVectors), (long) flatVectors.length * Sizeof.FLOAT);
        logger.debug("Appended {} vectors to GPU database", count);
    }

    private void expandCapacity(int requiredCapacity) {
        int newCapacity = Math.max(capacity * 2, requiredCapacity);
        
        logger.info("Expanding GPU capacity: {} → {}", capacity, newCapacity);
        
        // Allocate new GPU memory
        CUdeviceptr newDatabase = new CUdeviceptr();
        cuMemAlloc(newDatabase, (long) newCapacity * dimensions * Sizeof.FLOAT);
        
        // Copy existing data if any
        if (databaseOnGpu && vectorCount > 0) {
            cuMemcpyDtoD(newDatabase, d_database, (long) vectorCount * dimensions * Sizeof.FLOAT);
            cuMemFree(d_database);
        }
        
        d_database = newDatabase;
        
        // Reallocate distances buffer
        cuMemFree(d_distances);
        d_distances = new CUdeviceptr();
        cuMemAlloc(d_distances, (long) newCapacity * Sizeof.FLOAT);
        
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
        cuMemFree(d_query);
        return distances;
    }

    private CUdeviceptr uploadQuery(float[] query) {
        CUdeviceptr d_query = new CUdeviceptr();
        cuMemAlloc(d_query, (long) dimensions * Sizeof.FLOAT);
        cuMemcpyHtoD(d_query, Pointer.to(query), (long) dimensions * Sizeof.FLOAT);
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
        cuMemcpyDtoH(Pointer.to(distances), d_distances, (long) vectorCount * Sizeof.FLOAT);
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
            
            if (d_database != null && databaseOnGpu) {
                cuMemFree(d_database);
            }
            if (d_distances != null) {
                cuMemFree(d_distances);
            }
            if (context != null) {
                cuCtxDestroy(context);
            }
            
            logger.info("GPUVectorIndex closed, freed GPU memory");
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
}
