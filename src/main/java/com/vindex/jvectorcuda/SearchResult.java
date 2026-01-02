package com.vindex.jvectorcuda;

import java.util.Arrays;

// Result of vector search containing nearest neighbor IDs and distances.
public final class SearchResult {

    private final int[] ids;
    private final float[] distances;
    private final long searchTimeMs;

    public SearchResult(int[] ids, float[] distances, long searchTimeMs) {
        if (ids.length != distances.length) {
            throw new IllegalArgumentException("IDs and distances arrays must have same length");
        }
        this.ids = Arrays.copyOf(ids, ids.length);
        this.distances = Arrays.copyOf(distances, distances.length);
        this.searchTimeMs = searchTimeMs;
    }

    public int[] getIds() {
        return Arrays.copyOf(ids, ids.length);
    }

    public float[] getDistances() {
        return Arrays.copyOf(distances, distances.length);
    }

    public long getSearchTimeMs() {
        return searchTimeMs;
    }

    public int size() {
        return ids.length;
    }

    @Override
    public String toString() {
        return String.format("SearchResult{size=%d, timeMs=%d}", size(), searchTimeMs);
    }
}
