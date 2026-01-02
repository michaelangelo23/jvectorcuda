/**
 * Brute-force Euclidean distance kernel for vector similarity search.
 * 
 * Computes distances between a query vector and all database vectors.
 * Each thread computes the distance for one database vector.
 * 
 * Performance Characteristics:
 * - Coalesced memory access for database vectors
 * - ~1,153 operations per vector pair (384 dimensions)
 * - Memory bandwidth bound, not compute bound
 * 
 * @param database Database vectors [numVectors * dimensions] (row-major)
 * @param query Query vector [dimensions]
 * @param distances Output distances [numVectors]
 * @param numVectors Number of database vectors
 * @param dimensions Vector dimensionality
 */
extern "C"
__global__ void euclideanDistance(
    const float* database,
    const float* query,
    float* distances,
    int numVectors,
    int dimensions
) {
    // Global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numVectors) {
        float sum = 0.0f;
        
        // Compute squared Euclidean distance
        // Each thread reads one row of database (coalesced across warps)
        for (int d = 0; d < dimensions; d++) {
            float diff = database[idx * dimensions + d] - query[d];
            sum += diff * diff;  // Fused multiply-add (FMA)
        }
        
        // Store distance (sqrt can be deferred to CPU if needed)
        distances[idx] = sqrtf(sum);
    }
}

/**
 * Optimized version using shared memory for query vector.
 * Reduces global memory reads by caching query in shared memory.
 * 
 * Block size should be 256 threads for optimal occupancy.
 */
extern "C"
__global__ void euclideanDistanceShared(
    const float* database,
    const float* query,
    float* distances,
    int numVectors,
    int dimensions
) {
    // Shared memory for query vector (max 2048 floats = 8KB)
    extern __shared__ float sharedQuery[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Cooperatively load query into shared memory
    for (int d = threadIdx.x; d < dimensions; d += blockDim.x) {
        sharedQuery[d] = query[d];
    }
    __syncthreads();
    
    if (idx < numVectors) {
        float sum = 0.0f;
        
        // Use shared memory for query (faster access)
        for (int d = 0; d < dimensions; d++) {
            float diff = database[idx * dimensions + d] - sharedQuery[d];
            sum += diff * diff;
        }
        
        distances[idx] = sqrtf(sum);
    }
}
