package com.vindex.jvectorcuda;

import java.util.Arrays;

/**
 * Immutable result of a vector similarity search.
 *
 * <p>Contains the indices and distances of the k nearest neighbors found,
 * sorted by distance in ascending order (closest first).
 *
 * <h2>Example</h2>
 * <pre>{@code
 * SearchResult result = index.search(query, 10);
 *
 * // Access results
 * int[] neighbors = result.indices();   // Indices in original dataset
 * float[] distances = result.distances(); // Distances from query
 * long timeMs = result.searchTimeMs();   // Search duration
 *
 * // Iterate through results
 * for (int i = 0; i < result.size(); i++) {
 *     System.out.printf("Rank %d: index=%d, distance=%.4f%n",
 *         i, result.indices()[i], result.distances()[i]);
 * }
 * }</pre>
 *
 * <h2>Distance Interpretation</h2>
 * <ul>
 *   <li><b>Euclidean:</b> Lower = more similar (0 = identical)</li>
 *   <li><b>Cosine:</b> Lower = more similar (0 = identical, 2 = opposite)</li>
 *   <li><b>Inner Product:</b> Lower = more similar (negative values mean higher similarity)</li>
 * </ul>
 *
 * <p>This class is immutable and thread-safe. Defensive copies are made of
 * all arrays on construction and access.
 *
 * @see VectorIndex#search(float[], int)
 * @see VectorIndex#searchBatch(float[][], int)
 * @since 1.0.0
 */
public final class SearchResult {

    private final int[] ids;
    private final float[] distances;
    private final long searchTimeMs;

    /**
     * Creates a new search result.
     *
     * @param ids array of indices pointing to the nearest neighbors in the index
     * @param distances array of distances from the query to each neighbor
     * @param searchTimeMs time taken to perform the search in milliseconds
     * @throws IllegalArgumentException if ids and distances have different lengths
     */
    public SearchResult(int[] ids, float[] distances, long searchTimeMs) {
        if (ids.length != distances.length) {
            throw new IllegalArgumentException("IDs and distances arrays must have same length");
        }
        this.ids = Arrays.copyOf(ids, ids.length);
        this.distances = Arrays.copyOf(distances, distances.length);
        this.searchTimeMs = searchTimeMs;
    }

    /**
     * Returns the indices of the nearest neighbors.
     *
     * <p>Indices point to positions in the original dataset (order of vectors added).
     * Results are sorted by distance (closest neighbor at index 0).
     *
     * @return defensive copy of the indices array
     */
    public int[] getIds() {
        return Arrays.copyOf(ids, ids.length);
    }

    /**
     * Returns the indices of the nearest neighbors (alias for {@link #getIds()}).
     *
     * @return defensive copy of the indices array
     */
    public int[] indices() {
        return getIds();
    }

    /**
     * Returns the distances from the query to each neighbor.
     *
     * <p>Distance values depend on the metric used:
     * <ul>
     *   <li>Euclidean: L2 distance</li>
     *   <li>Cosine: 1 - cosine_similarity</li>
     *   <li>Inner Product: -dot_product (negated)</li>
     * </ul>
     *
     * @return defensive copy of the distances array
     */
    public float[] getDistances() {
        return Arrays.copyOf(distances, distances.length);
    }

    /**
     * Returns the distances from the query to each neighbor (alias for {@link #getDistances()}).
     *
     * @return defensive copy of the distances array
     */
    public float[] distances() {
        return getDistances();
    }

    /**
     * Returns the time taken to perform the search.
     *
     * @return search duration in milliseconds
     */
    public long getSearchTimeMs() {
        return searchTimeMs;
    }

    /**
     * Returns the time taken to perform the search (alias for {@link #getSearchTimeMs()}).
     *
     * @return search duration in milliseconds
     */
    public long searchTimeMs() {
        return searchTimeMs;
    }

    /**
     * Returns the number of results (k value used in search).
     *
     * @return number of nearest neighbors in this result
     */
    public int size() {
        return ids.length;
    }

    @Override
    public String toString() {
        return String.format("SearchResult{size=%d, timeMs=%d}", size(), searchTimeMs);
    }
}
