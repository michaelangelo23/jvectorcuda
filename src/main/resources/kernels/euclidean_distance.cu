// Euclidean distance kernel. Each thread computes distance for one database vector.
// Optimized with 4-way loop unrolling for better instruction-level parallelism.
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
        const float* vec = database + idx * dimensions;
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        
        // Process 4 elements at a time for better GPU pipeline utilization
        // Note: When dimensions < 4, limit is negative and unrolled loop is skipped
        int d = 0;
        const int limit = dimensions - 3;
        for (; d < limit; d += 4) {
            float diff0 = vec[d] - query[d];
            float diff1 = vec[d + 1] - query[d + 1];
            float diff2 = vec[d + 2] - query[d + 2];
            float diff3 = vec[d + 3] - query[d + 3];
            sum0 += diff0 * diff0;
            sum1 += diff1 * diff1;
            sum2 += diff2 * diff2;
            sum3 += diff3 * diff3;
        }
        
        // Handle remaining elements (and all elements when dimensions < 4)
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; d < dimensions; d++) {
            float diff = vec[d] - query[d];
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
        const float* vec = database + idx * dimensions;
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        
        // Process 4 elements at a time
        // Note: When dimensions < 4, limit is negative and unrolled loop is skipped
        int d = 0;
        const int limit = dimensions - 3;
        for (; d < limit; d += 4) {
            float diff0 = vec[d] - sharedQuery[d];
            float diff1 = vec[d + 1] - sharedQuery[d + 1];
            float diff2 = vec[d + 2] - sharedQuery[d + 2];
            float diff3 = vec[d + 3] - sharedQuery[d + 3];
            sum0 += diff0 * diff0;
            sum1 += diff1 * diff1;
            sum2 += diff2 * diff2;
            sum3 += diff3 * diff3;
        }
        
        // Handle remaining elements (and all elements when dimensions < 4)
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; d < dimensions; d++) {
            float diff = vec[d] - sharedQuery[d];
            sum += diff * diff;
        }
        
        distances[idx] = sqrtf(sum);
    }
}

// Batch kernel for processing multiple queries with a single kernel launch.
// Uses 2D grid: x-dimension = query index, y-dimension = vector blocks.
// Output: distances[queryIdx * numVectors + vectorIdx]
extern "C"
__global__ void euclideanDistanceBatch(
    const float* database,      // [numVectors x dimensions] flattened
    const float* queries,       // [numQueries x dimensions] flattened
    float* distances,           // [numQueries x numVectors] flattened output
    int numVectors,
    int numQueries,
    int dimensions
) {
    // Query index from grid x-dimension
    int queryIdx = blockIdx.x;
    // Vector index from grid y-dimension and thread
    int vectorIdx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (queryIdx < numQueries && vectorIdx < numVectors) {
        const float* query = queries + queryIdx * dimensions;
        const float* vec = database + vectorIdx * dimensions;
        
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        
        // Unrolled loop for better GPU pipeline utilization
        int d = 0;
        const int limit = dimensions - 3;
        for (; d < limit; d += 4) {
            float diff0 = vec[d] - query[d];
            float diff1 = vec[d + 1] - query[d + 1];
            float diff2 = vec[d + 2] - query[d + 2];
            float diff3 = vec[d + 3] - query[d + 3];
            sum0 += diff0 * diff0;
            sum1 += diff1 * diff1;
            sum2 += diff2 * diff2;
            sum3 += diff3 * diff3;
        }
        
        // Handle remaining elements
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; d < dimensions; d++) {
            float diff = vec[d] - query[d];
            sum += diff * diff;
        }
        
        // Output layout: [queryIdx][vectorIdx] -> linear index
        distances[queryIdx * numVectors + vectorIdx] = sqrtf(sum);
    }
}
