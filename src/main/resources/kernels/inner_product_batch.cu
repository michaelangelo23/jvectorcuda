// Batch Inner Product kernel
// Processes N queries simultaneously against the entire database.
// Grid: (ceil(numVectors/blockDim.x), numQueries)
// Output is negative inner product (for distance-like ordering: lower = more similar)

extern "C"
__global__ void innerProductBatch(
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
    
    // Compute inner product with 4-way unrolling
    float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
    
    int d = 0;
    const int limit = dimensions - 3;
    for (; d < limit; d += 4) {
        dot0 += vec[d] * query[d];
        dot1 += vec[d + 1] * query[d + 1];
        dot2 += vec[d + 2] * query[d + 2];
        dot3 += vec[d + 3] * query[d + 3];
    }
    
    // Handle remaining elements
    float dotProduct = dot0 + dot1 + dot2 + dot3;
    for (; d < dimensions; d++) {
        dotProduct += vec[d] * query[d];
    }
    
    // Negative inner product for distance ordering (lower = more similar)
    distances[queryIdx * numVectors + vectorIdx] = -dotProduct;
}

// Shared memory version for better query cache utilization
extern "C"
__global__ void innerProductBatchShared(
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
    
    // Compute inner product with shared query
    float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
    
    int d = 0;
    const int limit = dimensions - 3;
    for (; d < limit; d += 4) {
        dot0 += vec[d] * sharedQuery[d];
        dot1 += vec[d + 1] * sharedQuery[d + 1];
        dot2 += vec[d + 2] * sharedQuery[d + 2];
        dot3 += vec[d + 3] * sharedQuery[d + 3];
    }
    
    float dotProduct = dot0 + dot1 + dot2 + dot3;
    for (; d < dimensions; d++) {
        dotProduct += vec[d] * sharedQuery[d];
    }
    
    distances[queryIdx * numVectors + vectorIdx] = -dotProduct;
}
