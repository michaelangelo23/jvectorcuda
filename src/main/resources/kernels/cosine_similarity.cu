// Cosine Similarity kernel. Returns 1 - cosine_similarity as distance (0 = identical).
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
        
        for (int d = 0; d < dimensions; d++) {
            float a = database[idx * dimensions + d];
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
