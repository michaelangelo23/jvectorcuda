/**
 * Cosine Similarity kernel for vector similarity search.
 * 
 * Computes cosine similarity between a query vector and all database vectors.
 * Each thread computes the similarity for one database vector.
 * 
 * Cosine Similarity = (A · B) / (||A|| * ||B||)
 * 
 * Note: Returns 1 - cosine_similarity as distance (0 = identical, 2 = opposite)
 * This allows sorting by distance (smaller = more similar).
 * 
 * Performance Characteristics:
 * - Coalesced memory access for database vectors
 * - 3 passes per vector: dot product, norm_a, norm_b
 * - Memory bandwidth bound
 * 
 * @param database Database vectors [numVectors * dimensions] (row-major)
 * @param query Query vector [dimensions]
 * @param distances Output distances [numVectors] (1 - cosine_similarity)
 * @param numVectors Number of database vectors
 * @param dimensions Vector dimensionality
 * 
 * @author JVectorCUDA (AI-assisted, Human-verified)
 * @since 1.0.0
 */
extern "C"
__global__ void cosineSimilarity(
    const float* database,
    const float* query,
    float* distances,
    int numVectors,
    int dimensions
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numVectors) {
        float dotProduct = 0.0f;
        float normA = 0.0f;
        float normB = 0.0f;
        
        // Compute dot product and norms in single pass
        for (int d = 0; d < dimensions; d++) {
            float a = database[idx * dimensions + d];
            float b = query[d];
            dotProduct += a * b;
            normA += a * a;
            normB += b * b;
        }
        
        // Compute cosine similarity
        float normProduct = sqrtf(normA) * sqrtf(normB);
        float cosineSim;
        
        // Handle zero vectors (avoid division by zero)
        if (normProduct < 1e-8f) {
            cosineSim = 0.0f;  // Zero vectors have no similarity
        } else {
            cosineSim = dotProduct / normProduct;
            // Clamp to [-1, 1] to handle floating point errors
            cosineSim = fminf(1.0f, fmaxf(-1.0f, cosineSim));
        }
        
        // Return distance (1 - similarity) so smaller = more similar
        distances[idx] = 1.0f - cosineSim;
    }
}

/**
 * Optimized version using shared memory for query vector.
 * Reduces global memory reads by caching query in shared memory.
 * 
 * Block size should be 256 threads for optimal occupancy.
 */
extern "C"
__global__ void cosineSimilarityShared(
    const float* database,
    const float* query,
    float* distances,
    int numVectors,
    int dimensions
) {
    extern __shared__ float sharedQuery[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Cooperatively load query into shared memory
    for (int d = threadIdx.x; d < dimensions; d += blockDim.x) {
        sharedQuery[d] = query[d];
    }
    __syncthreads();
    
    if (idx < numVectors) {
        float dotProduct = 0.0f;
        float normA = 0.0f;
        float normB = 0.0f;
        
        for (int d = 0; d < dimensions; d++) {
            float a = database[idx * dimensions + d];
            float b = sharedQuery[d];
            dotProduct += a * b;
            normA += a * a;
            normB += b * b;
        }
        
        float normProduct = sqrtf(normA) * sqrtf(normB);
        float cosineSim = (normProduct < 1e-8f) ? 0.0f : dotProduct / normProduct;
        cosineSim = fminf(1.0f, fmaxf(-1.0f, cosineSim));
        
        distances[idx] = 1.0f - cosineSim;
    }
}
