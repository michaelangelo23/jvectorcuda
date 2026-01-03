package com.vindex.jvectorcuda;

import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;

/**
 * Utility class for detecting CUDA availability and validating GPU
 * capabilities.
 *
 * <p>
 * This class checks whether the system has a compatible NVIDIA GPU and CUDA
 * runtime for GPU-accelerated vector operations. Results are cached after the
 * first check for performance.
 *
 * <h2>Minimum Requirements</h2>
 * <ul>
 * <li><b>CUDA Driver:</b> 11.8 or newer (NOT the full CUDA Toolkit)</li>
 * <li><b>GPU:</b> Compute Capability 6.1+ (GTX 1060 / Tesla P4 or newer)</li>
 * <li><b>JCuda:</b> 12.0.0 native libraries</li>
 * </ul>
 *
 * <h2>CUDA Compatibility</h2>
 * <p>
 * JVectorCUDA uses PTX (Portable Intermediate Representation) which is
 * <b>forward-compatible</b>:
 * <ul>
 * <li>PTX compiled with CUDA 11.8 works on drivers 11.8, 12.0, 12.3+</li>
 * <li>The driver JIT-compiles PTX to native GPU code at runtime</li>
 * <li>Users do NOT need the full CUDA Toolkit, only the driver</li>
 * </ul>
 *
 * <h2>GPU Compatibility Matrix</h2>
 * <table border="1">
 * <tr>
 * <th>GPU Generation</th>
 * <th>Compute</th>
 * <th>Examples</th>
 * <th>Supported</th>
 * </tr>
 * <tr>
 * <td>Pascal</td>
 * <td>6.1</td>
 * <td>GTX 1060/1070/1080</td>
 * <td>Yes</td>
 * </tr>
 * <tr>
 * <td>Volta</td>
 * <td>7.0</td>
 * <td>Tesla V100</td>
 * <td>Yes</td>
 * </tr>
 * <tr>
 * <td>Turing</td>
 * <td>7.5</td>
 * <td>RTX 2080, Tesla T4</td>
 * <td>Yes</td>
 * </tr>
 * <tr>
 * <td>Ampere</td>
 * <td>8.0/8.6</td>
 * <td>A100, RTX 3090</td>
 * <td>Yes</td>
 * </tr>
 * <tr>
 * <td>Ada Lovelace</td>
 * <td>8.9</td>
 * <td>RTX 4090</td>
 * <td>Yes</td>
 * </tr>
 * <tr>
 * <td>Maxwell</td>
 * <td>5.x</td>
 * <td>GTX 980</td>
 * <td>No</td>
 * </tr>
 * </table>
 *
 * <h2>Usage</h2>
 * 
 * <pre>{@code
 * // Check availability before creating GPU index
 * if (CudaDetector.isAvailable()) {
 *     VectorIndex index = VectorIndexFactory.gpu(384);
 *     System.out.println("Using: " + CudaDetector.getGpuInfo());
 *     System.out.println("Driver: " + CudaDetector.getDriverVersion());
 * } else {
 *     VectorIndex index = VectorIndexFactory.cpu(384);
 *     System.out.println("GPU not available, using CPU");
 * }
 * }</pre>
 *
 * <h2>Troubleshooting</h2>
 * <p>
 * If {@link #isAvailable()} returns false:
 * <ol>
 * <li>Verify NVIDIA GPU is installed: {@code nvidia-smi}</li>
 * <li>Check driver version: {@code nvidia-smi} (top right shows CUDA
 * version)</li>
 * <li>Update drivers if below 11.8:
 * <a href="https://www.nvidia.com/drivers">nvidia.com/drivers</a></li>
 * <li>Verify GPU compute capability meets minimum (6.1)</li>
 * </ol>
 *
 * @see VectorIndexFactory#auto(int)
 * @see VectorIndexFactory#gpu(int)
 * @since 1.0.0
 */
public final class CudaDetector {

    /**
     * Minimum required CUDA Compute Capability major version (Pascal architecture).
     */
    private static final int MIN_COMPUTE_MAJOR = 6;

    /** Minimum required CUDA Compute Capability minor version. */
    private static final int MIN_COMPUTE_MINOR = 1;

    /** Minimum required CUDA driver major version. */
    private static final int MIN_DRIVER_MAJOR = 11;

    /** Minimum required CUDA driver minor version. */
    private static final int MIN_DRIVER_MINOR = 8;

    /**
     * Cached result of CUDA availability check. Volatile for thread-safe lazy init.
     */
    private static volatile Boolean cudaAvailable = null;

    /** Cached driver version string. Volatile for thread-safe lazy init. */
    private static volatile String driverVersionString = null;

    private CudaDetector() {
        // Utility class - prevent instantiation
    }

    /**
     * Checks if CUDA is available and meets minimum requirements.
     *
     * <p>
     * This method checks for:
     * <ul>
     * <li>At least one CUDA-capable device</li>
     * <li>CUDA driver version 11.8+</li>
     * <li>Compute Capability 6.1+ (GTX 1060 or newer)</li>
     * <li>Functional JCuda runtime</li>
     * </ul>
     *
     * <p>
     * The result is cached after the first call for performance.
     * Thread-safe via synchronized access.
     *
     * @return true if a compatible GPU is available, false otherwise
     */
    public static synchronized boolean isAvailable() {
        if (cudaAvailable != null) {
            return cudaAvailable;
        }

        try {
            JCuda.setExceptionsEnabled(true);

            int[] deviceCount = new int[1];
            JCuda.cudaGetDeviceCount(deviceCount);

            if (deviceCount[0] == 0) {
                cudaAvailable = false;
                return false;
            }

            // Check CUDA driver version
            int[] driverVersion = new int[1];
            JCuda.cudaDriverGetVersion(driverVersion);
            int driverMajor = driverVersion[0] / 1000;
            int driverMinor = (driverVersion[0] % 1000) / 10;
            driverVersionString = driverMajor + "." + driverMinor;

            if (driverMajor < MIN_DRIVER_MAJOR ||
                    (driverMajor == MIN_DRIVER_MAJOR && driverMinor < MIN_DRIVER_MINOR)) {
                System.err.println(String.format(
                        "CUDA driver %d.%d is below minimum required %d.%d. " +
                                "Please update your NVIDIA drivers: https://www.nvidia.com/drivers",
                        driverMajor, driverMinor, MIN_DRIVER_MAJOR, MIN_DRIVER_MINOR));
                cudaAvailable = false;
                return false;
            }

            // Check compute capability
            cudaDeviceProp prop = new cudaDeviceProp();
            JCuda.cudaGetDeviceProperties(prop, 0);

            boolean compatible = prop.major > MIN_COMPUTE_MAJOR ||
                    (prop.major == MIN_COMPUTE_MAJOR && prop.minor >= MIN_COMPUTE_MINOR);

            if (!compatible) {
                System.err.println(String.format(
                        "GPU compute capability %d.%d is below minimum required %d.%d",
                        prop.major, prop.minor, MIN_COMPUTE_MAJOR, MIN_COMPUTE_MINOR));
                cudaAvailable = false;
                return false;
            }

            cudaAvailable = true;
            return true;

        } catch (Exception e) {
            cudaAvailable = false;
            return false;
        }
    }

    /**
     * Returns the CUDA driver version string.
     *
     * <p>
     * Note: This is the driver version, not the CUDA Toolkit version.
     * Users only need the driver installed, not the full toolkit.
     *
     * @return driver version (e.g., "12.3"), or "unknown" if not detected
     */
    public static String getDriverVersion() {
        if (driverVersionString == null) {
            isAvailable(); // Populate cached values
        }
        return driverVersionString != null ? driverVersionString : "unknown";
    }

    /**
     * Returns the minimum required CUDA driver version.
     *
     * @return minimum driver version string (e.g., "11.8")
     */
    public static String getMinDriverVersion() {
        return MIN_DRIVER_MAJOR + "." + MIN_DRIVER_MINOR;
    }

    /**
     * Returns the minimum required Compute Capability.
     *
     * @return minimum compute capability string (e.g., "6.1")
     */
    public static String getMinComputeCapability() {
        return MIN_COMPUTE_MAJOR + "." + MIN_COMPUTE_MINOR;
    }

    /**
     * Returns a brief GPU information string for display.
     *
     * <p>
     * Format: "GPU Name (Compute X.Y, XXXX MB VRAM, XXXX MHz)"
     *
     * @return GPU info string, or error message if unavailable
     */
    public static String getGpuInfo() {
        if (!isAvailable()) {
            return "CUDA not available";
        }

        try {
            cudaDeviceProp prop = new cudaDeviceProp();
            JCuda.cudaGetDeviceProperties(prop, 0);

            int clockMhz = prop.clockRate / 1000;
            return String.format("%s (Compute %d.%d, %d MB VRAM, %d MHz)",
                    prop.getName(),
                    prop.major,
                    prop.minor,
                    prop.totalGlobalMem / (1024 * 1024),
                    clockMhz);
        } catch (Exception e) {
            return "Error getting GPU info: " + e.getMessage();
        }
    }

    /**
     * Returns detailed GPU specifications for benchmarking and diagnostics.
     *
     * <p>
     * Includes: name, compute capability, VRAM, clock rate, multiprocessors,
     * warp size, and max threads per block.
     *
     * @return multi-line GPU specs string, or error message if unavailable
     */
    public static String getDetailedGpuInfo() {
        if (!isAvailable()) {
            return "CUDA not available";
        }

        try {
            cudaDeviceProp prop = new cudaDeviceProp();
            JCuda.cudaGetDeviceProperties(prop, 0);

            StringBuilder info = new StringBuilder();
            info.append(String.format("%s\n", prop.getName()));
            info.append(String.format("  - Compute Capability: %d.%d\n", prop.major, prop.minor));
            info.append(String.format("  - Total VRAM: %d MB\n", prop.totalGlobalMem / (1024 * 1024)));
            int clockMhz = prop.clockRate / 1000;
            info.append(String.format("  - Clock Rate: %d MHz\n", clockMhz));
            info.append(String.format("  - Multiprocessors: %d\n", prop.multiProcessorCount));
            info.append(String.format("  - Warp Size: %d\n", prop.warpSize));
            info.append(String.format("  - Max Threads per Block: %d\n", prop.maxThreadsPerBlock));

            return info.toString();
        } catch (Exception e) {
            return "Error getting GPU info: " + e.getMessage();
        }
    }

    /**
     * Returns comprehensive compatibility information for diagnostics.
     *
     * <p>
     * Includes driver version, compute capability, and compatibility status.
     * Useful for debugging and support tickets.
     *
     * @return multi-line compatibility report
     */
    public static String getCompatibilityReport() {
        StringBuilder report = new StringBuilder();
        report.append("=== JVectorCUDA Compatibility Report ===\n\n");

        report.append("Requirements:\n");
        report.append(String.format("  - Min CUDA Driver: %s\n", getMinDriverVersion()));
        report.append(String.format("  - Min Compute Capability: %s\n", getMinComputeCapability()));
        report.append("\n");

        report.append("System:\n");
        report.append(String.format("  - CUDA Available: %s\n", isAvailable()));
        report.append(String.format("  - CUDA Driver: %s\n", getDriverVersion()));

        if (isAvailable()) {
            try {
                cudaDeviceProp prop = new cudaDeviceProp();
                JCuda.cudaGetDeviceProperties(prop, 0);
                report.append(String.format("  - GPU: %s\n", prop.getName()));
                report.append(String.format("  - Compute Capability: %d.%d\n", prop.major, prop.minor));
                report.append(String.format("  - VRAM: %d MB\n", prop.totalGlobalMem / (1024 * 1024)));
            } catch (Exception e) {
                report.append(String.format("  - Error: %s\n", e.getMessage()));
            }
        }

        report.append("\n");
        report.append("PTX Compatibility:\n");
        report.append("  - PTX is forward-compatible (compiled with 11.8, runs on 12.x+)\n");
        report.append("  - JIT compilation adapts to your GPU at runtime\n");

        return report.toString();
    }
}
