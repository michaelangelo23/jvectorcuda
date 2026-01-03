// Cosine Similarity kernel. Returns 1 - cosine_similarity as distance (0 = identical).
// Optimized with 4-way loop unrolling for better instruction-level parallelism.
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
        const float* vec = database + idx * dimensions;
        float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
        float normA0 = 0.0f, normA1 = 0.0f, normA2 = 0.0f, normA3 = 0.0f;
        float normB0 = 0.0f, normB1 = 0.0f, normB2 = 0.0f, normB3 = 0.0f;
        
        // Process 4 elements at a time
        int d = 0;
        int limit = dimensions - 3;
        for (; d < limit; d += 4) {
            float a0 = vec[d], a1 = vec[d + 1], a2 = vec[d + 2], a3 = vec[d + 3];
            float b0 = query[d], b1 = query[d + 1], b2 = query[d + 2], b3 = query[d + 3];
            
            dot0 += a0 * b0;
            dot1 += a1 * b1;
            dot2 += a2 * b2;
            dot3 += a3 * b3;
            
            normA0 += a0 * a0;
            normA1 += a1 * a1;
            normA2 += a2 * a2;
            normA3 += a3 * a3;
            
            normB0 += b0 * b0;
            normB1 += b1 * b1;
            normB2 += b2 * b2;
            normB3 += b3 * b3;
        }
        
        // Combine partial sums
        float dotProduct = dot0 + dot1 + dot2 + dot3;
        float normA = normA0 + normA1 + normA2 + normA3;
        float normB = normB0 + normB1 + normB2 + normB3;
        
        // Handle remaining elements
        for (; d < dimensions; d++) {
            float a = vec[d];
            float b = query[d];
            dotProduct += a * b;
            normA += a * a;
            normB += b * b;
        }
        
        float normProduct = sqrtf(normA) * sqrtf(normB);
        float cosineSim;
        
        if (normProduct < 1e-8f) {
            cosineSim = 0.0f;
        } else {
            cosineSim = dotProduct / normProduct;
            cosineSim = fminf(1.0f, fmaxf(-1.0f, cosineSim));
        }
        
        distances[idx] = 1.0f - cosineSim;
    }
}

// Shared memory version - caches query for faster access
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
    
    // Load query into shared memory
    for (int d = threadIdx.x; d < dimensions; d += blockDim.x) {
        sharedQuery[d] = query[d];
    }
    __syncthreads();
    
    if (idx < numVectors) {
        const float* vec = database + idx * dimensions;
        float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
        float normA0 = 0.0f, normA1 = 0.0f, normA2 = 0.0f, normA3 = 0.0f;
        float normB0 = 0.0f, normB1 = 0.0f, normB2 = 0.0f, normB3 = 0.0f;
        
        // Process 4 elements at a time
        int d = 0;
        int limit = dimensions - 3;
        for (; d < limit; d += 4) {
            float a0 = vec[d], a1 = vec[d + 1], a2 = vec[d + 2], a3 = vec[d + 3];
            float b0 = sharedQuery[d], b1 = sharedQuery[d + 1], b2 = sharedQuery[d + 2], b3 = sharedQuery[d + 3];
            
            dot0 += a0 * b0;
            dot1 += a1 * b1;
            dot2 += a2 * b2;
            dot3 += a3 * b3;
            
            normA0 += a0 * a0;
            normA1 += a1 * a1;
            normA2 += a2 * a2;
            normA3 += a3 * a3;
            
            normB0 += b0 * b0;
            normB1 += b1 * b1;
            normB2 += b2 * b2;
            normB3 += b3 * b3;
        }
        
        float dotProduct = dot0 + dot1 + dot2 + dot3;
        float normA = normA0 + normA1 + normA2 + normA3;
        float normB = normB0 + normB1 + normB2 + normB3;
        
        for (; d < dimensions; d++) {
            float a = vec[d];
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
