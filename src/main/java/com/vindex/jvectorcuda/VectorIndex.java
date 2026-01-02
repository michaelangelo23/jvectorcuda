package com.vindex.jvectorcuda;

import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * Core interface for GPU/CPU-accelerated vector similarity search.
 *
 * <p>JVectorCUDA provides high-performance nearest neighbor search using NVIDIA CUDA
 * for GPU acceleration, with automatic fallback to CPU when GPU is unavailable.
 *
 * <h2>Quick Start</h2>
 * <pre>{@code
 * // Create index with automatic GPU/CPU detection
 * try (VectorIndex index = VectorIndex.auto(384)) {
 *     // Add vectors (e.g., embeddings from a model)
 *     float[][] vectors = loadEmbeddings();
 *     index.add(vectors);
 *
 *     // Search for nearest neighbors
 *     float[] query = getQueryVector();
 *     SearchResult result = index.search(query, 10);
 *
 *     // Process results
 *     for (int i = 0; i < result.indices().length; i++) {
 *         System.out.printf("Match %d: index=%d, distance=%.4f%n",
 *             i, result.indices()[i], result.distances()[i]);
 *     }
 * }
 * }</pre>
 *
 * <h2>Implementation Notes</h2>
 * <ul>
 *   <li><b>GPU:</b> Uses brute-force exact search with CUDA acceleration.
 *       Best for batch queries where GPU upload cost is amortized.</li>
 *   <li><b>CPU:</b> Uses JVector's HNSW approximate search.
 *       Better for single queries or when GPU is unavailable.</li>
 * </ul>
 *
 * <h2>Thread Safety</h2>
 * <p>Implementations of this interface are <b>NOT thread-safe</b>. For concurrent
 * access, wrap with {@link ThreadSafeVectorIndex} or use factory methods like
 * {@link VectorIndexFactory#autoThreadSafe(int)}.
 *
 * <h2>Distance Metrics</h2>
 * <p>Default distance metric is Euclidean. Use {@link VectorIndexFactory} to
 * specify alternative metrics (Cosine Similarity, Inner Product).
 *
 * @see VectorIndexFactory
 * @see ThreadSafeVectorIndex
 * @see SearchResult
 * @see DistanceMetric
 * @since 1.0.0
 */
public interface VectorIndex extends AutoCloseable {

    /**
     * Adds vectors to the index.
     *
     * <p>Vectors are stored and indexed for subsequent similarity searches.
     * Each vector must have the same dimensionality as specified when creating the index.
     *
     * <h3>Performance Notes</h3>
     * <ul>
     *   <li>GPU: Vectors are uploaded to GPU memory. For best performance, add all
     *       vectors in a single call rather than multiple small batches.</li>
     *   <li>CPU: Vectors are indexed using HNSW algorithm.</li>
     * </ul>
     *
     * @param vectors 2D array of vectors to add, where {@code vectors[i]} is the i-th vector
     *                and {@code vectors[i].length} must equal {@link #getDimensions()}
     * @throws IllegalArgumentException if vectors is null, empty, or contains vectors
     *         with incorrect dimensions
     * @throws IllegalStateException if the index has been closed
     */
    void add(float[][] vectors);

    /**
     * Searches for the k nearest neighbors to the query vector.
     *
     * <p>Returns indices and distances of the k most similar vectors in the index.
     * Results are sorted by distance in ascending order (closest first).
     *
     * <h3>Example</h3>
     * <pre>{@code
     * SearchResult result = index.search(queryVector, 10);
     * int closestIndex = result.indices()[0];
     * float closestDistance = result.distances()[0];
     * }</pre>
     *
     * @param query the query vector with length equal to {@link #getDimensions()}
     * @param k the number of nearest neighbors to return (must be positive and &le; {@link #size()})
     * @return a {@link SearchResult} containing arrays of indices and distances
     * @throws IllegalArgumentException if query is null, has wrong dimensions, or k is invalid
     * @throws IllegalStateException if the index is empty or has been closed
     */
    SearchResult search(float[] query, int k);

    /**
     * Searches for the k nearest neighbors for multiple query vectors.
     *
     * <p>This is more efficient than calling {@link #search(float[], int)} multiple times,
     * especially on GPU where the kernel launch overhead is amortized across all queries.
     *
     * <h3>GPU Performance</h3>
     * <p>Batch queries can achieve 5-10x speedup compared to sequential single queries
     * because GPU memory transfer and kernel launch costs are paid only once.
     *
     * <h3>Example</h3>
     * <pre>{@code
     * float[][] queries = new float[100][384]; // 100 query vectors
     * List<SearchResult> results = index.searchBatch(queries, 10);
     * for (int i = 0; i < results.size(); i++) {
     *     System.out.printf("Query %d: closest=%d%n", i, results.get(i).indices()[0]);
     * }
     * }</pre>
     *
     * @param queries 2D array of query vectors
     * @param k the number of nearest neighbors to return per query
     * @return list of {@link SearchResult} objects, one per query
     * @throws IllegalArgumentException if queries is null/empty, has wrong dimensions, or k is invalid
     * @throws IllegalStateException if the index is empty or has been closed
     */
    List<SearchResult> searchBatch(float[][] queries, int k);

    /**
     * Asynchronously searches for the k nearest neighbors.
     *
     * <p>Returns immediately with a {@link CompletableFuture} that will be completed
     * when the search finishes. Useful for non-blocking operations in async applications.
     *
     * <h3>Example</h3>
     * <pre>{@code
     * CompletableFuture<SearchResult> future = index.searchAsync(query, 10);
     * // Do other work while search runs...
     * SearchResult result = future.get(); // Block when you need the result
     * }</pre>
     *
     * @param query the query vector
     * @param k the number of nearest neighbors to return
     * @return a CompletableFuture that will contain the search result
     * @throws IllegalArgumentException if query is null or has wrong dimensions
     */
    CompletableFuture<SearchResult> searchAsync(float[] query, int k);

    /**
     * Returns the dimensionality of vectors in this index.
     *
     * <p>All vectors added to this index must have exactly this many dimensions.
     *
     * @return the number of dimensions (e.g., 384, 768, 1536 for common embedding models)
     */
    int getDimensions();

    /**
     * Returns the number of vectors currently in the index.
     *
     * @return the count of indexed vectors, or 0 if empty
     */
    int size();

    /**
     * Creates an index with automatic GPU/CPU detection.
     *
     * <p>Attempts to use GPU acceleration if CUDA is available and sufficient
     * GPU memory exists. Falls back to CPU if GPU is unavailable.
     *
     * @param dimensions the dimensionality of vectors to be indexed
     * @return a new VectorIndex instance
     * @throws IllegalArgumentException if dimensions &le; 0
     * @see VectorIndexFactory#auto(int)
     */
    static VectorIndex auto(int dimensions) {
        return VectorIndexFactory.auto(dimensions);
    }

    /**
     * Creates a hybrid index with intelligent CPU/GPU routing.
     *
     * <p><b>This is the recommended default for most applications.</b> The hybrid index
     * automatically routes queries to the optimal backend:
     * <ul>
     *   <li>Single queries → CPU (lowest latency)</li>
     *   <li>Batch queries (10+) → GPU (5x+ speedup)</li>
     *   <li>Large datasets (50K+) → GPU (persistent memory)</li>
     * </ul>
     *
     * @param dimensions the dimensionality of vectors to be indexed
     * @return a new HybridVectorIndex instance
     * @throws IllegalArgumentException if dimensions &le; 0
     * @see VectorIndexFactory#hybrid(int)
     * @see HybridVectorIndex
     */
    static VectorIndex hybrid(int dimensions) {
        return VectorIndexFactory.hybrid(dimensions);
    }

    /**
     * Creates a CPU-only index using JVector's HNSW algorithm.
     *
     * <p>Use this when GPU is not needed or for workloads where approximate
     * nearest neighbor search is preferred (single queries, memory constraints).
     *
     * @param dimensions the dimensionality of vectors to be indexed
     * @return a new CPU-based VectorIndex instance
     * @throws IllegalArgumentException if dimensions &le; 0
     * @see VectorIndexFactory#cpu(int)
     */
    static VectorIndex cpu(int dimensions) {
        return VectorIndexFactory.cpu(dimensions);
    }

    /**
     * Creates a GPU-only index using CUDA brute-force search.
     *
     * <p>Use this when you need exact nearest neighbor search and have a compatible
     * NVIDIA GPU. Best performance is achieved with batch queries.
     *
     * @param dimensions the dimensionality of vectors to be indexed
     * @return a new GPU-based VectorIndex instance
     * @throws IllegalArgumentException if dimensions &le; 0
     * @throws IllegalStateException if CUDA is not available or GPU memory is insufficient
     * @see VectorIndexFactory#gpu(int)
     */
    static VectorIndex gpu(int dimensions) {
        return VectorIndexFactory.gpu(dimensions);
    }

    /**
     * Releases resources held by this index.
     *
     * <p>For GPU indices, this frees GPU memory. Always call close() or use
     * try-with-resources to prevent memory leaks.
     *
     * <p>After calling close(), any operations on this index will throw
     * {@link IllegalStateException}.
     */
    @Override
    void close();
}
