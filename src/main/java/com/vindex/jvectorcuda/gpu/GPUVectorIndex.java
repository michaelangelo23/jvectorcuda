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

import java.util.Arrays;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

import static jcuda.driver.JCudaDriver.*;

/**
 * GPU-accelerated vector index using persistent memory architecture.
 * 
 * <p>Key optimization: Database vectors are uploaded to GPU once and kept persistent,
 * achieving 5x+ speedup by avoiding repeated memory transfers.
 * 
 * <p>Architecture:
 * <pre>
 * Initial add():     Host → GPU (one-time cost)
 * Each search():     Query upload → Kernel → Results download
 * Database stays:    Persistent on GPU VRAM
 * </pre>
 * 
 * <p>Supported distance metrics:
 * <ul>
 *   <li>EUCLIDEAN - L2 distance (default)</li>
 *   <li>COSINE - 1 - cosine_similarity</li>
 *   <li>INNER_PRODUCT - negative dot product</li>
 * </ul>
 * 
 * @author JVectorCUDA (AI-assisted, Human-verified)
 * @since 1.0.0
 */
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

    /**
     * Creates a new GPU vector index with specified dimensions.
     * Uses Euclidean distance by default.
     * 
     * @param dimensions number of dimensions per vector
     * @throws IllegalArgumentException if dimensions <= 0
     */
    public GPUVectorIndex(int dimensions) {
        this(dimensions, DistanceMetric.EUCLIDEAN, 100_000);
    }

    /**
     * Creates a new GPU vector index with specified dimensions and distance metric.
     * 
     * @param dimensions number of dimensions per vector
     * @param metric distance metric to use
     * @throws IllegalArgumentException if dimensions <= 0 or metric is null
     */
    public GPUVectorIndex(int dimensions, DistanceMetric metric) {
        this(dimensions, metric, 100_000);
    }

    /**
     * Creates a new GPU vector index with specified dimensions, metric, and capacity.
     * 
     * @param dimensions number of dimensions per vector
     * @param metric distance metric to use
     * @param initialCapacity initial capacity for vectors
     * @throws IllegalArgumentException if dimensions <= 0, metric is null, or capacity <= 0
     */
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

    /**
     * Validates dimensions and values of input vectors.
     */
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

    /**
     * Ensures GPU memory can hold the required number of vectors.
     */
    private void ensureCapacity(int requiredCount) {
        if (requiredCount > capacity) {
            expandCapacity(requiredCount);
        }
    }

    /**
     * Uploads vectors to GPU, either as initial upload or append.
     */
    private void uploadVectorsToGpu(float[][] vectors) {
        float[] flatVectors = flattenVectors(vectors);
        
        if (!databaseOnGpu) {
            allocateAndUploadInitial(flatVectors, vectors.length);
        } else {
            appendToExisting(flatVectors, vectors.length);
        }
    }

    /**
     * Initial database upload - allocates GPU memory and copies data.
     */
    private void allocateAndUploadInitial(float[] flatVectors, int count) {
        d_database = new CUdeviceptr();
        cuMemAlloc(d_database, (long) capacity * dimensions * Sizeof.FLOAT);
        cuMemcpyHtoD(d_database, Pointer.to(flatVectors), (long) flatVectors.length * Sizeof.FLOAT);
        databaseOnGpu = true;
        logger.debug("Initial database upload: {} vectors", count);
    }

    /**
     * Appends vectors to existing GPU database.
     */
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

    /**
     * Validates search parameters.
     */
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

    /**
     * Computes distances from query to all vectors using GPU kernel.
     */
    private float[] computeDistancesOnGpu(float[] query) {
        CUdeviceptr d_query = uploadQuery(query);
        launchDistanceKernel(d_query);
        float[] distances = downloadDistances();
        cuMemFree(d_query);
        return distances;
    }

    /**
     * Uploads query vector to GPU.
     */
    private CUdeviceptr uploadQuery(float[] query) {
        CUdeviceptr d_query = new CUdeviceptr();
        cuMemAlloc(d_query, (long) dimensions * Sizeof.FLOAT);
        cuMemcpyHtoD(d_query, Pointer.to(query), (long) dimensions * Sizeof.FLOAT);
        return d_query;
    }

    /**
     * Launches the distance kernel on GPU.
     */
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

    /**
     * Downloads computed distances from GPU.
     */
    private float[] downloadDistances() {
        float[] distances = new float[vectorCount];
        cuMemcpyDtoH(Pointer.to(distances), d_distances, (long) vectorCount * Sizeof.FLOAT);
        return distances;
    }

    /**
     * Builds search result with top-k neighbors.
     */
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

    /**
     * Returns whether the database is currently loaded on GPU.
     * 
     * @return true if database is on GPU
     */
    public boolean isDatabaseOnGpu() {
        return databaseOnGpu;
    }

    /**
     * Returns the current capacity of the index.
     * 
     * @return maximum number of vectors before expansion
     */
    public int getCapacity() {
        return capacity;
    }

    /**
     * Returns the distance metric used by this index.
     * 
     * @return the distance metric (EUCLIDEAN, COSINE, or INNER_PRODUCT)
     */
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

    /**
     * Validates that vector values are finite (no NaN or Infinity).
     * 
     * @param vector the vector to validate
     * @param index the vector index for error messages
     * @throws IllegalArgumentException if vector contains NaN or Infinity
     */
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

    /**
     * Find indices of k smallest distances using partial sort.
     */
    private int[] findTopK(float[] distances, int k) {
        // Create index-distance pairs
        int n = distances.length;
        int[][] pairs = new int[n][1];
        float[] dists = new float[n];
        for (int i = 0; i < n; i++) {
            pairs[i][0] = i;
            dists[i] = distances[i];
        }
        
        // Simple selection for small k (optimize later if needed)
        int[] result = new int[k];
        boolean[] used = new boolean[n];
        
        for (int i = 0; i < k; i++) {
            float minDist = Float.MAX_VALUE;
            int minIdx = -1;
            for (int j = 0; j < n; j++) {
                if (!used[j] && dists[j] < minDist) {
                    minDist = dists[j];
                    minIdx = j;
                }
            }
            result[i] = minIdx;
            used[minIdx] = true;
        }
        
        return result;
    }
}
