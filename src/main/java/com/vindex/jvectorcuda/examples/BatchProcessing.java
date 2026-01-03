package com.vindex.jvectorcuda.examples;

import com.vindex.jvectorcuda.DistanceMetric;
import com.vindex.jvectorcuda.SearchResult;
import com.vindex.jvectorcuda.VectorIndex;
import com.vindex.jvectorcuda.VectorIndexFactory;

import java.util.List;
import java.util.Random;

/**
 * Batch Processing Example - Real ML training use case.
 * 
 * Demonstrates GPU speedup for batch operations:
 * - Upload embeddings once to GPU
 * - Process many queries (5-10x faster than CPU for batches)
 * - Use cosine similarity for normalized embeddings
 * 
 * This is the PRIMARY use case for JVectorCUDA:
 * ML training pipelines that need to find similar examples.
 */
public class BatchProcessing {

    public static void main(String[] args) {
        int dimensions = 384;
        int databaseSize = 50_000;
        int batchSize = 100;
        int k = 10;

        System.out.println("=== JVectorCUDA Batch Processing Demo ===\n");

        // Simulate loading embeddings from a file
        float[][] database = loadEmbeddings(databaseSize, dimensions);
        float[][] queryBatch = loadEmbeddings(batchSize, dimensions);

        // Use cosine similarity (common for normalized embeddings)
        try (VectorIndex index = VectorIndexFactory.hybrid(dimensions, DistanceMetric.COSINE)) {

            // Upload database once (stays in GPU memory)
            long uploadStart = System.currentTimeMillis();
            index.add(database);
            long uploadTime = System.currentTimeMillis() - uploadStart;
            System.out.printf("Database upload: %,d vectors in %d ms%n", databaseSize, uploadTime);

            // Process batch queries (GPU shines here)
            long batchStart = System.currentTimeMillis();
            List<SearchResult> results = index.searchBatch(queryBatch, k);
            long batchTime = System.currentTimeMillis() - batchStart;

            System.out.printf("Batch search: %d queries in %d ms (%.1f ms/query)%n",
                    batchSize, batchTime, (double) batchTime / batchSize);

            // Calculate throughput
            double qps = batchSize * 1000.0 / batchTime;
            System.out.printf("Throughput: %.1f queries/second%n", qps);

            // Show sample results
            System.out.println("\nSample result (query #0):");
            SearchResult first = results.get(0);
            for (int i = 0; i < Math.min(5, first.indices().length); i++) {
                System.out.printf("  Neighbor %d: index=%d, cosine_distance=%.4f%n",
                        i + 1, first.indices()[i], first.distances()[i]);
            }
        }

        System.out.println("\n=== End Demo ===");
    }

    // Simulate loading embeddings (replace with actual file I/O)
    private static float[][] loadEmbeddings(int count, int dims) {
        Random random = new Random(12345);
        float[][] vectors = new float[count][dims];
        for (int i = 0; i < count; i++) {
            float norm = 0;
            for (int j = 0; j < dims; j++) {
                vectors[i][j] = random.nextFloat() * 2 - 1;
                norm += vectors[i][j] * vectors[i][j];
            }
            // Normalize for cosine similarity
            norm = (float) Math.sqrt(norm);
            for (int j = 0; j < dims; j++) {
                vectors[i][j] /= norm;
            }
        }
        return vectors;
    }
}
