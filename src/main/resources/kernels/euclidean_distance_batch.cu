// Batch Euclidean distance kernel
// Processes N queries simultaneously against the entire database.
// Grid: (ceil(numVectors/blockDim.x), numQueries)
// Each thread computes distance for one (query, vector) pair.

extern "C"
__global__ void euclideanDistanceBatch(
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
    
    // 4-way unrolled distance computation
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    
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
    
    // Output: distances[queryIdx * numVectors + vectorIdx]
    distances[queryIdx * numVectors + vectorIdx] = sqrtf(sum);
}

// Shared memory version for better query cache utilization
extern "C"
__global__ void euclideanDistanceBatchShared(
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
    
    // 4-way unrolled distance computation
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    
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
    
    // Handle remaining elements
    float sum = sum0 + sum1 + sum2 + sum3;
    for (; d < dimensions; d++) {
        float diff = vec[d] - sharedQuery[d];
        sum += diff * diff;
    }
    
    distances[queryIdx * numVectors + vectorIdx] = sqrtf(sum);
}
