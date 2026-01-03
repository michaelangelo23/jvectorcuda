package com.vindex.jvectorcuda.gpu;

import com.vindex.jvectorcuda.CudaUtils;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Manages a pool of reusable GPU memory allocations to minimize overhead.
 * 
 * <p>
 * Allocating GPU memory (cudaMalloc) is a synchronous and expensive operation
 * involving the OS and GPU driver. For frequent operations like single-vector
 * queries, this allocation overhead can dominate total execution time.
 * 
 * <p>
 * This pool maintains reusable {@link CUdeviceptr} buckets by size. When memory
 * is requested, it checks for an existing free block of the exact size. If none
 * exists, a new block is allocated. Released blocks are returned to the pool
 * for future reuse instead of being freed immediately.
 */
public class GpuMemoryPool implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(GpuMemoryPool.class);

    // Map from size (bytes) to queue of available pointers
    private final Map<Long, Queue<CUdeviceptr>> pool = new ConcurrentHashMap<>();

    /**
     * Acquires a pointer to a block of GPU memory of the specified size.
     * 
     * <p>
     * If a reusable block of this size is available in the pool, it is returned.
     * Otherwise, a new block is allocated on the GPU.
     * 
     * @param sizeBytes the size of the memory block in bytes
     * @return a pointer to the allocated GPU memory
     * @throws RuntimeException if GPU allocation fails
     */
    public CUdeviceptr acquire(long sizeBytes) {
        if (sizeBytes <= 0) {
            throw new IllegalArgumentException("Size must be positive: " + sizeBytes);
        }

        Queue<CUdeviceptr> queue = pool.get(sizeBytes);
        if (queue != null) {
            CUdeviceptr ptr = queue.poll();
            if (ptr != null) {
                return ptr; // Reusing existing block
            }
        }

        // Allocate new block if pool is empty
        CUdeviceptr ptr = new CUdeviceptr();
        CudaUtils.checkCudaResult(
                JCudaDriver.cuMemAlloc(ptr, sizeBytes),
                "cuMemAlloc from pool");

        return ptr;
    }

    /**
     * Releases a memory block back to the pool for reuse.
     * 
     * <p>
     * The caller must ensure the pointer is valid and matches the specified size.
     * 
     * @param ptr       the pointer to release
     * @param sizeBytes the size of the block (must match acquire size)
     */
    public void release(CUdeviceptr ptr, long sizeBytes) {
        if (ptr == null) {
            return;
        }

        pool.computeIfAbsent(sizeBytes, k -> new ConcurrentLinkedQueue<>()).offer(ptr);
    }

    // Map from size (bytes) to queue of available pinned host pointers
    private final Map<Long, Queue<Pointer>> pinnedMemoryPool = new ConcurrentHashMap<>();

    /**
     * Acquires a pointer to a block of Pinned (Page-Locked) CPU memory.
     * 
     * <p>
     * Pinned memory allows for "Zero-Copy" or faster DMA transfers to the GPU,
     * bypassing the driver's internal staging buffers.
     * 
     * @param sizeBytes the size of the memory block in bytes
     * @return a Generic Pointer to the allocated pinned host memory
     */
    public Pointer acquirePinnedMemory(long sizeBytes) {
        if (sizeBytes <= 0) {
            throw new IllegalArgumentException("Size must be positive: " + sizeBytes);
        }

        Queue<Pointer> queue = pinnedMemoryPool.get(sizeBytes);
        if (queue != null) {
            Pointer pooledPointer = queue.poll();
            if (pooledPointer != null) {
                return pooledPointer;
            }
        }

        // Allocate new pinned host memory
        Pointer hostPointer = new Pointer();
        CudaUtils.checkCudaResult(
                JCudaDriver.cuMemAllocHost(hostPointer, sizeBytes),
                "cuMemAllocHost Pinned Memory");

        return hostPointer;
    }

    /**
     * Releases a pinned memory block back to the pool.
     * 
     * @param hostPointer the pointer to release
     * @param sizeBytes   the size of the block
     */
    public void releasePinnedMemory(Pointer hostPointer, long sizeBytes) {
        if (hostPointer == null) {
            return;
        }
        pinnedMemoryPool.computeIfAbsent(sizeBytes, k -> new ConcurrentLinkedQueue<>()).offer(hostPointer);
    }

    /**
     * Frees all pooled memory on the GPU and Host.
     */
    @Override
    public void close() {
        // Cleanup Device Memory (VRAM)
        cleanupDeviceMemory();

        // Cleanup Host Pinned Memory (RAM)
        cleanupHostPinnedMemory();

        logger.debug("GpuMemoryPool closed. All resources freed.");
    }

    private void cleanupDeviceMemory() {
        int freedCount = 0;
        long freedBytes = 0;

        for (Map.Entry<Long, Queue<CUdeviceptr>> entry : pool.entrySet()) {
            long size = entry.getKey();
            Queue<CUdeviceptr> queue = entry.getValue();

            CUdeviceptr devicePointer;
            while ((devicePointer = queue.poll()) != null) {
                try {
                    JCudaDriver.cuMemFree(devicePointer);
                    freedCount++;
                    freedBytes += size;
                } catch (Exception e) {
                    logger.warn("Failed to free pooled GPU memory", e);
                }
            }
        }
        pool.clear();
        logger.debug("Freed {} device memory blocks, total {} bytes", freedCount, freedBytes);
    }

    private void cleanupHostPinnedMemory() {
        int freedCount = 0;
        long freedBytes = 0;

        for (Map.Entry<Long, Queue<Pointer>> entry : pinnedMemoryPool.entrySet()) {
            long size = entry.getKey();
            Queue<Pointer> queue = entry.getValue();

            Pointer hostPointer;
            while ((hostPointer = queue.poll()) != null) {
                try {
                    JCudaDriver.cuMemFreeHost(hostPointer);
                    freedCount++;
                    freedBytes += size;
                } catch (Exception e) {
                    logger.warn("Failed to free pinned host memory", e);
                }
            }
        }
        pinnedMemoryPool.clear();
        logger.debug("Freed {} pinned host memory blocks, total {} bytes", freedCount, freedBytes);
    }
}
