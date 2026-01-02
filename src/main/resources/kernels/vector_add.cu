/**
 * Simple vector addition kernel for POC #2.
 * Adds two vectors element-wise: result[i] = a[i] + b[i]
 *
 * Thread organization:
 * - Block size: 256 threads (optimal for GTX 1080)
 * - Grid size: Calculated to cover all elements
 *
 * Memory access pattern:
 * - Coalesced reads/writes (each thread accesses consecutive memory)
 * - Global memory only (no shared memory optimization yet)
 *
 * @param a First input vector
 * @param b Second input vector
 * @param result Output vector (a + b)
 * @param n Number of elements
 */
extern "C"
__global__ void vectorAdd(const float* a, const float* b, float* result, int n) {
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check (handle non-multiple-of-blocksize array lengths)
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}
