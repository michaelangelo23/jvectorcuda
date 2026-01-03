package com.vindex.jvectorcuda.exception;

/**
 * Thrown when the GPU runs out of memory (VRAM) or Pinned Host Memory.
 * 
 * <p>
 * This allows the application to catch this specific error and attempt recovery
 * steps,
 * such as:
 * <ul>
 * <li>Reducing batch size</li>
 * <li>Evicting persistent memory</li>
 * <li>Falling back to CPU execution</li>
 * </ul>
 */
public class GpuOutOfMemoryException extends RuntimeException {

    public GpuOutOfMemoryException(String message) {
        super(message);
    }

    public GpuOutOfMemoryException(String message, Throwable cause) {
        super(message, cause);
    }
}
