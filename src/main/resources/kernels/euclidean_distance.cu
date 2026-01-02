// Euclidean distance kernel. Each thread computes distance for one database vector.
extern "C"
__global__ void euclideanDistance(
    const float* database,
    const float* query,
    float* distances,
    int numVectors,
    int dimensions
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numVectors) {
        float sum = 0.0f;
        
        for (int d = 0; d < dimensions; d++) {
            float diff = database[idx * dimensions + d] - query[d];
            sum += diff * diff;
        }
        
        distances[idx] = sqrtf(sum);
    }
}

// Shared memory version - caches query for faster access
extern "C"
__global__ void euclideanDistanceShared(
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
        float sum = 0.0f;
        
        for (int d = 0; d < dimensions; d++) {
            float diff = database[idx * dimensions + d] - sharedQuery[d];
            sum += diff * diff;
        }
        
        distances[idx] = sqrtf(sum);
    }
}
