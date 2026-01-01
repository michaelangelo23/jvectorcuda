package com.vindex.jvectorcuda;

import java.util.Arrays;

/**
 * Result of a vector search operation containing nearest neighbor IDs and
 * distances.
 *
 * @author JVectorCUDA Team
 * @since 1.0.0
 */
public final class SearchResult {

    private final int[] ids;
    private final float[] distances;
    private final long searchTimeMs;

    /**
     * Creates a new search result.
     *
     * @param ids          vector IDs of nearest neighbors (ordered by distance)
     * @param distances    distances to query vector
     * @param searchTimeMs time taken for search in milliseconds
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
     * Returns the IDs of nearest neighbors.
     *
     * @return array of vector IDs
     */
    public int[] getIds() {
        return Arrays.copyOf(ids, ids.length);
    }

    /**
     * Returns the distances to the query vector.
     *
     * @return array of distances
     */
    public float[] getDistances() {
        return Arrays.copyOf(distances, distances.length);
    }

    /**
     * Returns the search execution time.
     *
     * @return time in milliseconds
     */
    public long getSearchTimeMs() {
        return searchTimeMs;
    }

    /**
     * Returns the number of results.
     *
     * @return result count
     */
    public int size() {
        return ids.length;
    }

    @Override
    public String toString() {
        return String.format("SearchResult{size=%d, timeMs=%d}", size(), searchTimeMs);
    }
}
