package com.vindex.jvectorcuda.cpu;

import com.vindex.jvectorcuda.SearchResult;
import com.vindex.jvectorcuda.VectorIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * CPU-based vector index using brute-force search.
 * 
 * <p>This serves as the fallback when GPU is not available or not beneficial.
 * For small datasets or single queries, CPU is often faster due to no memory transfer overhead.
 * 
 * <p>Future optimization: Integrate JVector's HNSW for approximate nearest neighbor search
 * on larger datasets where O(n) brute-force becomes too slow.
 * 
 * @author JVectorCUDA (AI-assisted, Human-verified)
 * @since 1.0.0
 */
public class CPUVectorIndex implements VectorIndex {

    private static final Logger logger = LoggerFactory.getLogger(CPUVectorIndex.class);
    
    private final int dimensions;
    private final List<float[]> vectors;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    /**
     * Creates a new CPU vector index with specified dimensions.
     * 
     * @param dimensions number of dimensions per vector
     * @throws IllegalArgumentException if dimensions <= 0
     */
    public CPUVectorIndex(int dimensions) {
        if (dimensions <= 0) {
            throw new IllegalArgumentException("Dimensions must be positive, got: " + dimensions);
        }
        
        this.dimensions = dimensions;
        this.vectors = new ArrayList<>();
        
        logger.info("CPUVectorIndex created: {} dimensions", dimensions);
    }

    @Override
    public void add(float[][] newVectors) {
        checkNotClosed();
        
        if (newVectors == null || newVectors.length == 0) {
            return;
        }
        
        // Validate and add vectors
        for (int i = 0; i < newVectors.length; i++) {
            if (newVectors[i].length != dimensions) {
                throw new IllegalArgumentException(
                    String.format("Vector %d has %d dimensions, expected %d", 
                        i, newVectors[i].length, dimensions));
            }
            validateVectorValues(newVectors[i], i);
            // Copy to avoid external modification
            float[] copy = new float[dimensions];
            System.arraycopy(newVectors[i], 0, copy, 0, dimensions);
            vectors.add(copy);
        }
        
        logger.debug("Added {} vectors, total: {}", newVectors.length, vectors.size());
    }

    @Override
    public SearchResult search(float[] query, int k) {
        checkNotClosed();
        
        if (query == null || query.length != dimensions) {
            throw new IllegalArgumentException(
                String.format("Query must have %d dimensions, got %d", 
                    dimensions, query == null ? 0 : query.length));
        }
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive, got: " + k);
        }
        if (vectors.isEmpty()) {
            return new SearchResult(new int[0], new float[0], 0);
        }
        
        k = Math.min(k, vectors.size());
        
        long startTime = System.nanoTime();
        
        // Compute all distances
        float[] distances = new float[vectors.size()];
        for (int i = 0; i < vectors.size(); i++) {
            distances[i] = euclideanDistance(query, vectors.get(i));
        }
        
        // Find top-k
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
        return vectors.size();
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            vectors.clear();
            logger.info("CPUVectorIndex closed");
        }
    }

    private void checkNotClosed() {
        if (closed.get()) {
            throw new IllegalStateException("CPUVectorIndex has been closed");
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

    /**
     * Computes Euclidean distance between two vectors.
     */
    private float euclideanDistance(float[] a, float[] b) {
        float sum = 0.0f;
        for (int i = 0; i < dimensions; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return (float) Math.sqrt(sum);
    }

    /**
     * Find indices of k smallest distances using partial selection.
     */
    private int[] findTopK(float[] distances, int k) {
        int n = distances.length;
        int[] result = new int[k];
        boolean[] used = new boolean[n];
        
        for (int i = 0; i < k; i++) {
            float minDist = Float.MAX_VALUE;
            int minIdx = -1;
            for (int j = 0; j < n; j++) {
                if (!used[j] && distances[j] < minDist) {
                    minDist = distances[j];
                    minIdx = j;
                }
            }
            result[i] = minIdx;
            used[minIdx] = true;
        }
        
        return result;
    }
}
