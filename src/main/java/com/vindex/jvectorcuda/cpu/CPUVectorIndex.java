package com.vindex.jvectorcuda.cpu;

import com.vindex.jvectorcuda.DistanceMetric;
import com.vindex.jvectorcuda.SearchResult;
import com.vindex.jvectorcuda.VectorIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * CPU-based vector index using brute-force search for exact nearest neighbor queries.
 *
 * <p>This implementation stores vectors in heap memory and performs linear scan
 * for each search. It serves as a fallback when GPU is unavailable and is suitable
 * for smaller datasets or when GPU overhead isn't justified.
 *
 * <h2>Performance Characteristics</h2>
 * <table border="1">
 *   <tr><th>Scenario</th><th>CPU vs GPU</th><th>Recommendation</th></tr>
 *   <tr><td>Single query, small dataset (&lt;10K)</td><td>1-2x faster</td><td>Use CPU</td></tr>
 *   <tr><td>Single query, large dataset (50K+)</td><td>~0.5x slower</td><td>Consider GPU</td></tr>
 *   <tr><td>Batch queries (10+)</td><td>0.1-0.2x slower</td><td>Use GPU</td></tr>
 *   <tr><td>Memory constrained</td><td>N/A</td><td>Use CPU</td></tr>
 * </table>
 *
 * <h2>Algorithm</h2>
 * <p>Uses <b>brute-force exact search</b>:
 * <ul>
 *   <li>Linear scan through all vectors computing distances</li>
 *   <li>Max-heap for top-k selection (O(n log k))</li>
 *   <li>Returns exact nearest neighbors (no approximation)</li>
 * </ul>
 *
 * <h2>Memory Management</h2>
 * <ul>
 *   <li>Vectors stored in Java heap (ArrayList of float[])</li>
 *   <li>Vectors are copied on add to prevent external modification</li>
 *   <li>Memory grows dynamically with ArrayList defaults</li>
 * </ul>
 *
 * <h2>Thread Safety</h2>
 * <p><b>NOT thread-safe.</b> For concurrent access, wrap with
 * {@link com.vindex.jvectorcuda.ThreadSafeVectorIndex} or use
 * {@link com.vindex.jvectorcuda.VectorIndexFactory#cpuThreadSafe(int)}.
 *
 * <h2>Example</h2>
 * <pre>{@code
 * try (CPUVectorIndex index = new CPUVectorIndex(384)) {
 *     index.add(embeddings);
 *     SearchResult result = index.search(query, 10);
 *     System.out.println("Closest match: " + result.indices()[0]);
 * }
 * }</pre>
 *
 * @see VectorIndex
 * @see com.vindex.jvectorcuda.VectorIndexFactory#cpu(int)
 * @see DistanceMetric
 * @since 1.0.0
 */
public class CPUVectorIndex implements VectorIndex {

    private static final Logger logger = LoggerFactory.getLogger(CPUVectorIndex.class);
    
    private final int dimensions;
    private final List<float[]> vectors;
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final DistanceMetric distanceMetric;

    /**
     * Creates a CPU vector index with Euclidean distance.
     *
     * @param dimensions the dimensionality of vectors (e.g., 384, 768, 1536)
     * @throws IllegalArgumentException if dimensions &le; 0
     */
    public CPUVectorIndex(int dimensions) {
        this(dimensions, DistanceMetric.EUCLIDEAN);
    }

    /**
     * Creates a CPU vector index with specified distance metric.
     *
     * @param dimensions the dimensionality of vectors
     * @param metric the distance metric for similarity calculations
     * @throws IllegalArgumentException if dimensions &le; 0 or metric is null
     */
    public CPUVectorIndex(int dimensions, DistanceMetric metric) {
        if (dimensions <= 0) {
            throw new IllegalArgumentException("Dimensions must be positive, got: " + dimensions);
        }
        if (metric == null) {
            throw new IllegalArgumentException("Distance metric cannot be null");
        }
        
        this.dimensions = dimensions;
        this.vectors = new ArrayList<>();
        this.distanceMetric = metric;
        
        logger.info("CPUVectorIndex created: {} dimensions, metric={}", dimensions, metric);
    }

    /**
     * {@inheritDoc}
     *
     * <p>Vectors are copied before storage to prevent external modification.
     *
     * @throws IllegalArgumentException if vectors is null, contains null vectors,
     *         vectors have wrong dimensions, or contain NaN/Infinity values
     * @throws IllegalStateException if the index has been closed
     */
    @Override
    public void add(float[][] newVectors) {
        checkNotClosed();
        
        if (newVectors == null) {
            throw new IllegalArgumentException("Vectors array cannot be null");
        }
        if (newVectors.length == 0) {
            return;
        }
        
        // Validate and add vectors
        for (int i = 0; i < newVectors.length; i++) {
            if (newVectors[i] == null) {
                throw new IllegalArgumentException(
                    String.format("Vector %d is null", i));
            }
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

    /**
     * {@inheritDoc}
     *
     * <p>Performs linear scan through all vectors. For large datasets with
     * single queries, consider GPU implementation for better performance.
     */
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
        
        // Compute all distances using configured metric
        float[] distances = new float[vectors.size()];
        for (int i = 0; i < vectors.size(); i++) {
            distances[i] = computeDistance(query, vectors.get(i));
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

    /** {@inheritDoc} */
    @Override
    public CompletableFuture<SearchResult> searchAsync(float[] query, int k) {
        return CompletableFuture.supplyAsync(() -> search(query, k));
    }

    /**
     * {@inheritDoc}
     *
     * <p>Processes queries sequentially. For batch workloads, GPU implementation
     * offers significantly better performance.
     */
    @Override
    public List<SearchResult> searchBatch(float[][] queries, int k) {
        checkNotClosed();
        
        if (queries == null || queries.length == 0) {
            return java.util.Collections.emptyList();
        }
        
        List<SearchResult> results = new ArrayList<>(queries.length);
        
        for (float[] query : queries) {
            results.add(search(query, k));
        }
        
        return results;
    }

    /** {@inheritDoc} */
    @Override
    public int getDimensions() {
        return dimensions;
    }

    /** {@inheritDoc} */
    @Override
    public int size() {
        return vectors.size();
    }

    /**
     * {@inheritDoc}
     *
     * <p>Clears the internal vector storage. This method is idempotent.
     */
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

    // Compute distance based on configured metric
    private float computeDistance(float[] a, float[] b) {
        return switch (distanceMetric) {
            case EUCLIDEAN -> euclideanDistance(a, b);
            case COSINE -> cosineDistance(a, b);
            case INNER_PRODUCT -> innerProductDistance(a, b);
        };
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

    // Cosine distance: 1 - cosine_similarity (smaller = more similar)
    private float cosineDistance(float[] a, float[] b) {
        float dotProduct = 0.0f;
        float normA = 0.0f;
        float normB = 0.0f;
        
        for (int i = 0; i < dimensions; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        
        float denominator = (float) (Math.sqrt(normA) * Math.sqrt(normB));
        if (denominator == 0.0f) {
            return 1.0f; // Maximum distance for zero vectors
        }
        
        float cosineSimilarity = dotProduct / denominator;
        return 1.0f - cosineSimilarity;
    }

    // Inner product distance: -dot_product (smaller = more similar, i.e., higher dot product)
    private float innerProductDistance(float[] a, float[] b) {
        float dotProduct = 0.0f;
        for (int i = 0; i < dimensions; i++) {
            dotProduct += a[i] * b[i];
        }
        // Negate so that higher dot product = smaller distance
        return -dotProduct;
    }

    /**
     * Returns the distance metric used for similarity calculations.
     *
     * @return the configured distance metric
     */
    public DistanceMetric getDistanceMetric() {
        return distanceMetric;
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
