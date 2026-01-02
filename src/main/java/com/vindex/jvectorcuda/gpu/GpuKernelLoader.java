package com.vindex.jvectorcuda.gpu;

import jcuda.Pointer;
import jcuda.driver.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.ConcurrentHashMap;

import static jcuda.driver.JCudaDriver.*;

// Loads and executes CUDA kernels from PTX files with per-context caching.
public class GpuKernelLoader {
    
    private static final Logger logger = LoggerFactory.getLogger(GpuKernelLoader.class);
    
    // Static cache for loaded PTX bytes (key: ptxFileName)
    // PTX files are context-independent and safe to cache statically
    private static final ConcurrentHashMap<String, byte[]> PTX_CACHE = new ConcurrentHashMap<>();
    
    // Per-context cache for loaded modules (key: context handle -> ptxFileName -> module)
    // Modules are context-dependent and must be cached per-context to avoid INVALID_HANDLE errors
    private static final ConcurrentHashMap<Long, ConcurrentHashMap<String, CUmodule>> CONTEXT_MODULE_CACHE = new ConcurrentHashMap<>();
    
    private CUmodule module;
    private CUfunction function;
    
    public GpuKernelLoader(String ptxFileName, String kernelName) {
        // Initialize CUDA driver (safe to call multiple times)
        cuInit(0);
        
        // Get current context - use hashCode as identifier since getNativePointer() is protected
        CUcontext context = new CUcontext();
        cuCtxGetCurrent(context);
        long contextHandle = System.identityHashCode(context);
        
        // Load or retrieve cached PTX bytes
        byte[] ptxBytes = PTX_CACHE.computeIfAbsent(ptxFileName, this::loadPTXFromResources);
        
        // Get or create per-context module cache
        ConcurrentHashMap<String, CUmodule> moduleCache = CONTEXT_MODULE_CACHE.computeIfAbsent(
            contextHandle, 
            k -> new ConcurrentHashMap<>()
        );
        
        // Load or retrieve cached module for this context
        module = moduleCache.computeIfAbsent(ptxFileName, key -> {
            CUmodule newModule = new CUmodule();
            int result = cuModuleLoadData(newModule, ptxBytes);
            if (result != CUresult.CUDA_SUCCESS) {
                throw new RuntimeException("Failed to load CUDA module: " + CUresult.stringFor(result));
            }
            logger.info("Loaded and cached CUDA module from {} for context {}", ptxFileName, Long.toHexString(contextHandle));
            return newModule;
        });
        
        // Get function reference
        function = new CUfunction();
        int result = cuModuleGetFunction(function, module, kernelName);
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("Failed to get kernel function '" + kernelName + "': " 
                + CUresult.stringFor(result));
        }
        
        logger.debug("Retrieved kernel '{}' from cached module {} (context: {})", 
            kernelName, ptxFileName, Long.toHexString(contextHandle));
    }
    
    private byte[] loadPTXFromResources(String fileName) {
        String resourcePath = "/kernels/" + fileName;
        
        // Use class literal instead of getClass() to avoid issues with subclasses
        try (InputStream is = GpuKernelLoader.class.getResourceAsStream(resourcePath)) {
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
    
    /**
     * Clears all cached modules for a specific context.
     * Should be called when a CUDA context is being destroyed.
     *
     * @param contextHandle the context identifier to clear cache for
     */
    public static void clearContextCache(long contextHandle) {
        ConcurrentHashMap<String, CUmodule> removed = CONTEXT_MODULE_CACHE.remove(contextHandle);
        if (removed != null) {
            logger.debug("Cleared {} cached modules for context {}", removed.size(), Long.toHexString(contextHandle));
        }
    }
    
    /**
     * Clears all cached modules across all contexts.
     * Use with caution - only call when shutting down completely.
     */
    public static void clearAllCaches() {
        int contextCount = CONTEXT_MODULE_CACHE.size();
        CONTEXT_MODULE_CACHE.clear();
        PTX_CACHE.clear();
        logger.info("Cleared all kernel caches ({} contexts)", contextCount);
    }
    
    public CUmodule getModule() {
        return module;
    }
}
