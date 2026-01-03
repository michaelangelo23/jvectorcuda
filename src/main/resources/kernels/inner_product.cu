// Inner Product kernel. Returns negative dot product so smaller = more similar.
// Optimized with 4-way loop unrolling for better instruction-level parallelism.
extern "C"
__global__ void innerProduct(
    const float* database,
    const float* query,
    float* distances,
    int numVectors,
    int dimensions
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numVectors) {
        const float* vec = database + idx * dimensions;
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        
        // Process 4 elements at a time
        int d = 0;
        int limit = dimensions - 3;
        for (; d < limit; d += 4) {
            sum0 += vec[d] * query[d];
            sum1 += vec[d + 1] * query[d + 1];
            sum2 += vec[d + 2] * query[d + 2];
            sum3 += vec[d + 3] * query[d + 3];
        }
        
        // Handle remaining elements
        float dotProduct = sum0 + sum1 + sum2 + sum3;
        for (; d < dimensions; d++) {
            dotProduct += vec[d] * query[d];
        }
        
        distances[idx] = -dotProduct;
    }
}

// Shared memory version - caches query for faster access
extern "C"
__global__ void innerProductShared(
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
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        
        // Process 4 elements at a time
        int d = 0;
        int limit = dimensions - 3;
        for (; d < limit; d += 4) {
            sum0 += vec[d] * sharedQuery[d];
            sum1 += vec[d + 1] * sharedQuery[d + 1];
            sum2 += vec[d + 2] * sharedQuery[d + 2];
            sum3 += vec[d + 3] * sharedQuery[d + 3];
        }
        
        float dotProduct = sum0 + sum1 + sum2 + sum3;
        for (; d < dimensions; d++) {
            dotProduct += vec[d] * sharedQuery[d];
        }
        
        distances[idx] = -dotProduct;
    }
}
