package com.vindex.jvectorcuda.gpu;

import jcuda.Pointer;
import jcuda.driver.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;

import static jcuda.driver.JCudaDriver.*;

// Loads and executes CUDA kernels from PTX files.
public class GpuKernelLoader {
    
    private static final Logger logger = LoggerFactory.getLogger(GpuKernelLoader.class);
    
    private CUmodule module;
    private CUfunction function;
    
    public GpuKernelLoader(String ptxFileName, String kernelName) {
        // Initialize CUDA driver
        cuInit(0);
        
        // Load PTX file from resources
        byte[] ptxBytes = loadPTXFromResources(ptxFileName);
        
        // Create module from PTX
        module = new CUmodule();
        int result = cuModuleLoadData(module, ptxBytes);
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("Failed to load CUDA module: " + CUresult.stringFor(result));
        }
        
        // Get function reference
        function = new CUfunction();
        result = cuModuleGetFunction(function, module, kernelName);
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("Failed to get kernel function '" + kernelName + "': " 
                + CUresult.stringFor(result));
        }
        
        logger.info("Successfully loaded kernel '{}' from {}", kernelName, ptxFileName);
    }
    
    private byte[] loadPTXFromResources(String fileName) {
        String resourcePath = "/kernels/" + fileName;
        
        try (InputStream is = getClass().getResourceAsStream(resourcePath)) {
            if (is == null) {
                throw new RuntimeException("PTX file not found: " + resourcePath);
            }
            
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            byte[] buffer = new byte[8192];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                baos.write(buffer, 0, bytesRead);
            }
            
            // CUDA requires null-terminated PTX string
            baos.write(0);
            
            return baos.toByteArray();
            
        } catch (IOException e) {
            throw new RuntimeException("Failed to load PTX file: " + fileName, e);
        }
    }
    
    public void launch(int gridSize, int blockSize, Pointer kernelParameters) {
        int result = cuLaunchKernel(
            function,
            gridSize, 1, 1,      // Grid dimensions (x, y, z)
            blockSize, 1, 1,     // Block dimensions (x, y, z)
            0,                   // Shared memory size
            null,                // Stream (null = default stream)
            kernelParameters,    // Kernel parameters
            null                 // Extra options
        );
        
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("Kernel launch failed: " + CUresult.stringFor(result));
        }
        
        // Wait for kernel to complete
        cuCtxSynchronize();
    }
    
    public CUfunction getFunction() {
        return function;
    }
    
    public CUmodule getModule() {
        return module;
    }
}
