package com.vindex.jvectorcuda.examples;

import com.vindex.jvectorcuda.SearchResult;
import com.vindex.jvectorcuda.VectorIndex;
import com.vindex.jvectorcuda.VectorIndexFactory;

import java.util.List;
import java.util.Random;

/**
 * Basic JVectorCUDA Example - 50 lines, copy-paste ready.
 * 
 * Demonstrates:
 * - Creating a hybrid index (auto GPU/CPU routing)
 * - Adding vectors
 * - Single query (auto-routes to CPU for low latency)
 * - Batch queries (auto-routes to GPU for high throughput)
 */
public class BasicExample {

    public static void main(String[] args) {
        int dimensions = 384; // Common embedding size (e.g., MiniLM)
        int numVectors = 10_000;
        int numQueries = 100;
        int k = 10; // Top-k results

        // Generate random test data
        float[][] embeddings = generateRandomVectors(numVectors, dimensions);
        float[][] queries = generateRandomVectors(numQueries, dimensions);

        // Create hybrid index (auto GPU/CPU routing)
        try (VectorIndex index = VectorIndexFactory.hybrid(dimensions)) {

            // Add vectors (uploaded to GPU once if available)
            index.add(embeddings);
            System.out.printf("Added %,d vectors (%d dimensions)%n", numVectors, dimensions);

            // Single query - auto-routes to CPU (lower latency)
            SearchResult single = index.search(queries[0], k);
            System.out.printf("Single query: found %d neighbors%n", single.indices().length);

            // Batch queries - auto-routes to GPU (higher throughput)
            List<SearchResult> batch = index.searchBatch(queries, k);
            System.out.printf("Batch query: processed %d queries%n", batch.size());

            // Access results
            System.out.println("\nTop 3 neighbors for first query:");
            for (int i = 0; i < Math.min(3, single.indices().length); i++) {
                System.out.printf("  #%d: index=%d, distance=%.4f%n",
                        i + 1, single.indices()[i], single.distances()[i]);
            }
        }
    }

    private static float[][] generateRandomVectors(int count, int dims) {
        Random random = new Random(42);
        float[][] vectors = new float[count][dims];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < dims; j++) {
                vectors[i][j] = random.nextFloat() * 2 - 1;
            }
        }
        return vectors;
    }
}
