package com.vindex.jvectorcuda;

import com.vindex.jvectorcuda.exception.GpuOutOfMemoryException;
import jcuda.driver.CUresult;

public class CudaUtils {

    /**
     * Checks the result of a CUDA operation and throws an exception if it failed.
     * Specifically handles OutOfMemory errors with a custom exception.
     *
     * @param result    the CUDA result code
     * @param operation description of the operation that was attempted
     * @throws GpuOutOfMemoryException if the error is CUDA_ERROR_OUT_OF_MEMORY
     * @throws RuntimeException        for other CUDA errors
     */
    public static void checkCudaResult(int result, String operation) {
        if (result == CUresult.CUDA_SUCCESS) {
            return;
        }

        if (result == CUresult.CUDA_ERROR_OUT_OF_MEMORY) {
            throw new GpuOutOfMemoryException(String.format(
                    "GPU Out of Memory during %s. Try reducing batch size, vector count, or dimensionality.",
                    operation));
        }

        throw new RuntimeException(String.format(
                "CUDA error in %s: %s (code %d)",
                operation,
                CUresult.stringFor(result),
                result));
    }
}
