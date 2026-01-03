// Batch Cosine similarity kernel
// Processes N queries simultaneously against the entire database.
// Grid: (ceil(numVectors/blockDim.x), numQueries)
// Output is cosine distance: 1.0 - cosine_similarity

extern "C"
__global__ void cosineSimilarityBatch(
    const float* database,      // [numVectors x dimensions] - row major
    const float* queries,       // [numQueries x dimensions] - row major
    float* distances,           // [numQueries x numVectors] - row major output
    int numVectors,
    int numQueries,
    int dimensions
) {
    int vectorIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int queryIdx = blockIdx.y;
    
    if (vectorIdx >= numVectors || queryIdx >= numQueries) {
        return;
    }
    
    const float* vec = database + vectorIdx * dimensions;
    const float* query = queries + queryIdx * dimensions;
    
    // Compute dot product and norms with 4-way unrolling
    float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
    float normV0 = 0.0f, normV1 = 0.0f, normV2 = 0.0f, normV3 = 0.0f;
    float normQ0 = 0.0f, normQ1 = 0.0f, normQ2 = 0.0f, normQ3 = 0.0f;
    
    int d = 0;
    const int limit = dimensions - 3;
    for (; d < limit; d += 4) {
        float v0 = vec[d], v1 = vec[d+1], v2 = vec[d+2], v3 = vec[d+3];
        float q0 = query[d], q1 = query[d+1], q2 = query[d+2], q3 = query[d+3];
        
        dot0 += v0 * q0;
        dot1 += v1 * q1;
        dot2 += v2 * q2;
        dot3 += v3 * q3;
        
        normV0 += v0 * v0;
        normV1 += v1 * v1;
        normV2 += v2 * v2;
        normV3 += v3 * v3;
        
        normQ0 += q0 * q0;
        normQ1 += q1 * q1;
        normQ2 += q2 * q2;
        normQ3 += q3 * q3;
    }
    
    // Handle remaining elements
    float dotProduct = dot0 + dot1 + dot2 + dot3;
    float normVec = normV0 + normV1 + normV2 + normV3;
    float normQuery = normQ0 + normQ1 + normQ2 + normQ3;
    
    for (; d < dimensions; d++) {
        float v = vec[d];
        float q = query[d];
        dotProduct += v * q;
        normVec += v * v;
        normQuery += q * q;
    }
    
    // Compute cosine similarity and clamp to valid range
    float denominator = sqrtf(normVec) * sqrtf(normQuery);
    float cosineSim = (denominator > 0.0f) ? (dotProduct / denominator) : 0.0f;
    
    // Clamp to handle floating point precision issues
    cosineSim = fmaxf(-1.0f, fminf(1.0f, cosineSim));
    
    // Cosine distance = 1 - cosine_similarity
    distances[queryIdx * numVectors + vectorIdx] = 1.0f - cosineSim;
}

// Shared memory version for better query cache utilization
extern "C"
__global__ void cosineSimilarityBatchShared(
    const float* database,
    const float* queries,
    float* distances,
    int numVectors,
    int numQueries,
    int dimensions
) {
    extern __shared__ float sharedQuery[];
    
    int vectorIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int queryIdx = blockIdx.y;
    
    if (queryIdx >= numQueries) {
        return;
    }
    
    // Cooperatively load query into shared memory
    const float* query = queries + queryIdx * dimensions;
    for (int d = threadIdx.x; d < dimensions; d += blockDim.x) {
        sharedQuery[d] = query[d];
    }
    __syncthreads();
    
    if (vectorIdx >= numVectors) {
        return;
    }
    
    const float* vec = database + vectorIdx * dimensions;
    
    // Compute with shared query
    float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
    float normV0 = 0.0f, normV1 = 0.0f, normV2 = 0.0f, normV3 = 0.0f;
    float normQ0 = 0.0f, normQ1 = 0.0f, normQ2 = 0.0f, normQ3 = 0.0f;
    
    int d = 0;
    const int limit = dimensions - 3;
    for (; d < limit; d += 4) {
        float v0 = vec[d], v1 = vec[d+1], v2 = vec[d+2], v3 = vec[d+3];
        float q0 = sharedQuery[d], q1 = sharedQuery[d+1], q2 = sharedQuery[d+2], q3 = sharedQuery[d+3];
        
        dot0 += v0 * q0;
        dot1 += v1 * q1;
        dot2 += v2 * q2;
        dot3 += v3 * q3;
        
        normV0 += v0 * v0;
        normV1 += v1 * v1;
        normV2 += v2 * v2;
        normV3 += v3 * v3;
        
        normQ0 += q0 * q0;
        normQ1 += q1 * q1;
        normQ2 += q2 * q2;
        normQ3 += q3 * q3;
    }
    
    float dotProduct = dot0 + dot1 + dot2 + dot3;
    float normVec = normV0 + normV1 + normV2 + normV3;
    float normQuery = normQ0 + normQ1 + normQ2 + normQ3;
    
    for (; d < dimensions; d++) {
        float v = vec[d];
        float q = sharedQuery[d];
        dotProduct += v * q;
        normVec += v * v;
        normQuery += q * q;
    }
    
    float denominator = sqrtf(normVec) * sqrtf(normQuery);
    float cosineSim = (denominator > 0.0f) ? (dotProduct / denominator) : 0.0f;
    cosineSim = fmaxf(-1.0f, fminf(1.0f, cosineSim));
    
    distances[queryIdx * numVectors + vectorIdx] = 1.0f - cosineSim;
}
