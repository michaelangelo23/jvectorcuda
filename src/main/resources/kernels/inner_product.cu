/**
 * Inner Product (Dot Product) kernel for vector similarity search.
 * 
 * Computes the negative inner product between a query vector and all database vectors.
 * Each thread computes the inner product for one database vector.
 * 
 * Inner Product = A · B = Σ(a_i * b_i)
 * 
 * Note: Returns NEGATIVE inner product as distance so that:
 * - Higher similarity (larger dot product) → smaller distance
 * - This allows sorting by distance (smaller = more similar)
 * 
 * Use Case: Normalized embeddings (e.g., from sentence transformers) where
 * inner product equals cosine similarity and is faster to compute.
 * 
 * Performance Characteristics:
 * - Coalesced memory access for database vectors
 * - Single pass per vector (fastest distance metric)
 * - Memory bandwidth bound
 * 
 * @param database Database vectors [numVectors * dimensions] (row-major)
 * @param query Query vector [dimensions]
 * @param distances Output distances [numVectors] (negative inner product)
 * @param numVectors Number of database vectors
 * @param dimensions Vector dimensionality
 * 
 * @author JVectorCUDA (AI-assisted, Human-verified)
 * @since 1.0.0
 */
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
        
        // Compute dot product in single pass
        for (int d = 0; d < dimensions; d++) {
            dotProduct += database[idx * dimensions + d] * query[d];
        }
        
        // Return negative so smaller = more similar
        distances[idx] = -dotProduct;
    }
}

/**
 * Optimized version using shared memory for query vector.
 * Reduces global memory reads by caching query in shared memory.
 * 
 * Block size should be 256 threads for optimal occupancy.
 */
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
    
    // Cooperatively load query into shared memory
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
