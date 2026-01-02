// Inner Product kernel. Returns negative dot product so smaller = more similar.
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
        float dotProduct = 0.0f;
        
        for (int d = 0; d < dimensions; d++) {
            dotProduct += database[idx * dimensions + d] * query[d];
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
        float dotProduct = 0.0f;
        
        for (int d = 0; d < dimensions; d++) {
            dotProduct += database[idx * dimensions + d] * sharedQuery[d];
        }
        
        distances[idx] = -dotProduct;
    }
}
