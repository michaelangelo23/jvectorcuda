// Vector addition kernel: result[i] = a[i] + b[i]
extern "C"
__global__ void vectorAdd(const float* a, const float* b, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}
