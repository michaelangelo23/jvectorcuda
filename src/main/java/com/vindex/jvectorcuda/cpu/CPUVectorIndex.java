package com.vindex.jvectorcuda.cpu;

import com.vindex.jvectorcuda.SearchResult;
import com.vindex.jvectorcuda.VectorIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

// CPU-based brute-force vector index. Fallback when GPU unavailable. Not thread-safe.
public class CPUVectorIndex implements VectorIndex {

    private static final Logger logger = LoggerFactory.getLogger(CPUVectorIndex.class);
    
    private final int dimensions;
    private final List<float[]> vectors;
    private final AtomicBoolean closed = new AtomicBoolean(false);

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

    // Validates that vector values are finite
    private void validateVectorValues(float[] vector, int index) {
        for (int i = 0; i < vector.length; i++) {
            if (!Float.isFinite(vector[i])) {
                throw new IllegalArgumentException(
                    String.format("Vector %d contains invalid value at index %d: %s",
                        index, i, vector[i]));
            }
        }
    }

    // Euclidean distance between two vectors
    private float euclideanDistance(float[] a, float[] b) {
        float sum = 0.0f;
        for (int i = 0; i < dimensions; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return (float) Math.sqrt(sum);
    }

    // Find k smallest distances using max-heap, O(n log k)
    private int[] findTopK(float[] distances, int k) {
        int n = distances.length;
        
        // Max-heap of size k to track smallest distances
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
        
        // Extract indices sorted by distance
        int[] result = new int[k];
        for (int i = k - 1; i >= 0; i--) {
            result[i] = maxHeap.poll()[0];
        }
        
        return result;
    }
}
