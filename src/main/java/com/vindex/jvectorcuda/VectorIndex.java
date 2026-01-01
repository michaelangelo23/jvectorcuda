package com.vindex.jvectorcuda;

import java.util.concurrent.CompletableFuture;

/**
 * Interface for GPU/CPU-accelerated vector similarity search.
 * 
 * <p>
 * Implementations automatically detect GPU availability and fall back to CPU
 * when needed.
 * 
 * <p>
 * Example usage:
 * 
 * <pre>{@code
 * VectorIndex index = VectorIndex.auto(384);
 * index.add(vectors);
 * SearchResult result = index.search(query, 10);
 * }</pre>
 *
 * @author JVectorCUDA Team
 * @since 1.0.0
 */
public interface VectorIndex extends AutoCloseable {

    /**
     * Adds vectors to the index.
     *
     * @param vectors array of vectors to add, each must have dimensions matching
     *                index
     * @throws IllegalArgumentException if vector dimensions don't match
     */
    void add(float[][] vectors);

    /**
     * Searches for k nearest neighbors to the query vector.
     *
     * @param query query vector (must match index dimensions)
     * @param k     number of nearest neighbors to return
     * @return search results containing IDs and distances
     * @throws IllegalArgumentException if query dimensions don't match or k <= 0
     */
    SearchResult search(float[] query, int k);

    /**
     * Asynchronously searches for k nearest neighbors.
     *
     * @param query query vector
     * @param k     number of nearest neighbors
     * @return future containing search results
     */
    CompletableFuture<SearchResult> searchAsync(float[] query, int k);

    /**
     * Returns the number of dimensions for vectors in this index.
     *
     * @return vector dimensions
     */
    int getDimensions();

    /**
     * Returns the number of vectors currently in the index.
     *
     * @return vector count
     */
    int size();

    /**
     * Creates an index that auto-detects GPU and falls back to CPU.
     *
     * @param dimensions number of dimensions for vectors
     * @return new VectorIndex instance
     */
    static VectorIndex auto(int dimensions) {
        return VectorIndexFactory.auto(dimensions);
    }

    /**
     * Creates a CPU-only index.
     *
     * @param dimensions number of dimensions for vectors
     * @return new CPU-based VectorIndex
     */
    static VectorIndex cpu(int dimensions) {
        return VectorIndexFactory.cpu(dimensions);
    }

    /**
     * Creates a GPU-only index.
     *
     * @param dimensions number of dimensions for vectors
     * @return new GPU-based VectorIndex
     * @throws UnsupportedOperationException if CUDA is not available
     */
    static VectorIndex gpu(int dimensions) {
        return VectorIndexFactory.gpu(dimensions);
    }

    @Override
    void close();
}
