package com.vindex.jvectorcuda;

import java.util.concurrent.CompletableFuture;

// Interface for GPU/CPU-accelerated vector similarity search.
public interface VectorIndex extends AutoCloseable {

    // Adds vectors to the index
    void add(float[][] vectors);

    // Searches for k nearest neighbors
    SearchResult search(float[] query, int k);

    // Async search
    CompletableFuture<SearchResult> searchAsync(float[] query, int k);

    int getDimensions();

    int size();

    // Creates index with auto GPU/CPU detection
    static VectorIndex auto(int dimensions) {
        return VectorIndexFactory.auto(dimensions);
    }

    // Creates CPU-only index
    static VectorIndex cpu(int dimensions) {
        return VectorIndexFactory.cpu(dimensions);
    }

    // Creates GPU-only index (throws if CUDA unavailable)
    static VectorIndex gpu(int dimensions) {
        return VectorIndexFactory.gpu(dimensions);
    }

    @Override
    void close();
}
