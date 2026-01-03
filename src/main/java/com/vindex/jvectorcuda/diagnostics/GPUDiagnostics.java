package com.vindex.jvectorcuda.diagnostics;

import com.vindex.jvectorcuda.CudaDetector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * GPU diagnostics and capability detection.
 * Provides detailed information about GPU availability, capabilities,
 * and recommendations for optimal configuration.
 */
public class GPUDiagnostics {

    private static final Logger logger = LoggerFactory.getLogger(GPUDiagnostics.class);

    /**
     * Diagnostic report with GPU information and recommendations.
     */
    public static class DiagnosticReport {
        private final boolean gpuAvailable;
        private final String cudaDriverVersion;
        private final String gpuModel;
        private final String computeCapability;
        private final long availableVRAM;
        private final long totalVRAM;
        private final int cudaDeviceCount;
        private final String jcudaVersion;
        private final List<String> recommendations;
        private final List<String> warnings;
        private final List<String> errors;

        private DiagnosticReport(Builder builder) {
            this.gpuAvailable = builder.gpuAvailable;
            this.cudaDriverVersion = builder.cudaDriverVersion;
            this.gpuModel = builder.gpuModel;
            this.computeCapability = builder.computeCapability;
            this.availableVRAM = builder.availableVRAM;
            this.totalVRAM = builder.totalVRAM;
            this.cudaDeviceCount = builder.cudaDeviceCount;
            this.jcudaVersion = builder.jcudaVersion;
            this.recommendations = new ArrayList<>(builder.recommendations);
            this.warnings = new ArrayList<>(builder.warnings);
            this.errors = new ArrayList<>(builder.errors);
        }

        public static Builder builder() {
            return new Builder();
        }

        public boolean isGpuAvailable() {
            return gpuAvailable;
        }

        public String getCudaDriverVersion() {
            return cudaDriverVersion;
        }

        public String getGpuModel() {
            return gpuModel;
        }

        public String getComputeCapability() {
            return computeCapability;
        }

        public long getAvailableVRAM() {
            return availableVRAM;
        }

        public long getTotalVRAM() {
            return totalVRAM;
        }

        public int getCudaDeviceCount() {
            return cudaDeviceCount;
        }

        public String getJcudaVersion() {
            return jcudaVersion;
        }

        public List<String> getRecommendations() {
            return new ArrayList<>(recommendations);
        }

        public List<String> getWarnings() {
            return new ArrayList<>(warnings);
        }

        public List<String> getErrors() {
            return new ArrayList<>(errors);
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("=== GPU Diagnostics Report ===\n\n");

            sb.append("GPU Status: ").append(gpuAvailable ? "AVAILABLE ✓" : "NOT AVAILABLE ✗").append("\n");

            if (gpuAvailable) {
                sb.append("\n--- Hardware Information ---\n");
                sb.append("GPU Model: ").append(gpuModel).append("\n");
                sb.append("Compute Capability: ").append(computeCapability).append("\n");
                sb.append("CUDA Devices: ").append(cudaDeviceCount).append("\n");
                if (totalVRAM > 0) {
                    sb.append("VRAM: ").append(formatBytes(availableVRAM)).append(" / ")
                      .append(formatBytes(totalVRAM)).append("\n");
                }
                sb.append("CUDA Driver Version: ").append(cudaDriverVersion).append("\n");
            }

            sb.append("\n--- Software Information ---\n");
            sb.append("JCuda Version: ").append(jcudaVersion).append("\n");
            sb.append("Java Version: ").append(System.getProperty("java.version")).append("\n");
            sb.append("OS: ").append(System.getProperty("os.name")).append(" ")
              .append(System.getProperty("os.arch")).append("\n");

            if (!errors.isEmpty()) {
                sb.append("\n--- Errors ---\n");
                for (String error : errors) {
                    sb.append("❌ ").append(error).append("\n");
                }
            }

            if (!warnings.isEmpty()) {
                sb.append("\n--- Warnings ---\n");
                for (String warning : warnings) {
                    sb.append("⚠️  ").append(warning).append("\n");
                }
            }

            if (!recommendations.isEmpty()) {
                sb.append("\n--- Recommendations ---\n");
                for (String rec : recommendations) {
                    sb.append("💡 ").append(rec).append("\n");
                }
            }

            return sb.toString();
        }

        private String formatBytes(long bytes) {
            if (bytes < 1024) {
                return bytes + " B";
            } else if (bytes < 1024 * 1024) {
                return String.format("%.2f KB", bytes / 1024.0);
            } else if (bytes < 1024 * 1024 * 1024) {
                return String.format("%.2f MB", bytes / (1024.0 * 1024.0));
            } else {
                return String.format("%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
            }
        }

        public static class Builder {
            private boolean gpuAvailable = false;
            private String cudaDriverVersion = "Unknown";
            private String gpuModel = "Unknown";
            private String computeCapability = "Unknown";
            private long availableVRAM = 0;
            private long totalVRAM = 0;
            private int cudaDeviceCount = 0;
            private String jcudaVersion = "Unknown";
            private List<String> recommendations = new ArrayList<>();
            private List<String> warnings = new ArrayList<>();
            private List<String> errors = new ArrayList<>();

            public Builder gpuAvailable(boolean gpuAvailable) {
                this.gpuAvailable = gpuAvailable;
                return this;
            }

            public Builder cudaDriverVersion(String cudaDriverVersion) {
                this.cudaDriverVersion = cudaDriverVersion;
                return this;
            }

            public Builder gpuModel(String gpuModel) {
                this.gpuModel = gpuModel;
                return this;
            }

            public Builder computeCapability(String computeCapability) {
                this.computeCapability = computeCapability;
                return this;
            }

            public Builder availableVRAM(long availableVRAM) {
                this.availableVRAM = availableVRAM;
                return this;
            }

            public Builder totalVRAM(long totalVRAM) {
                this.totalVRAM = totalVRAM;
                return this;
            }

            public Builder cudaDeviceCount(int cudaDeviceCount) {
                this.cudaDeviceCount = cudaDeviceCount;
                return this;
            }

            public Builder jcudaVersion(String jcudaVersion) {
                this.jcudaVersion = jcudaVersion;
                return this;
            }

            public Builder addRecommendation(String recommendation) {
                this.recommendations.add(recommendation);
                return this;
            }

            public Builder addWarning(String warning) {
                this.warnings.add(warning);
                return this;
            }

            public Builder addError(String error) {
                this.errors.add(error);
                return this;
            }

            public DiagnosticReport build() {
                return new DiagnosticReport(this);
            }
        }
    }

    /**
     * Runs comprehensive GPU diagnostics.
     *
     * @return Diagnostic report with GPU information
     */
    public static DiagnosticReport runDiagnostics() {
        DiagnosticReport.Builder builder = DiagnosticReport.builder();

        // Check GPU availability
        boolean gpuAvailable = CudaDetector.isAvailable();
        builder.gpuAvailable(gpuAvailable);

        if (gpuAvailable) {
            // Get GPU info
            String gpuInfo = CudaDetector.getGpuInfo();
            builder.gpuModel(extractGpuModel(gpuInfo));
            builder.computeCapability(extractComputeCapability(gpuInfo));
            builder.cudaDeviceCount(1);

            // Add recommendations based on GPU capabilities
            builder.addRecommendation("GPU acceleration is available. Use GPUVectorIndex or HybridVectorIndex for best performance.");
            builder.addRecommendation("For large datasets, consider using HybridVectorIndex for automatic CPU fallback.");
        } else {
            // GPU not available
            builder.addError("CUDA GPU not detected. JVectorCUDA will fall back to CPU-only mode.");
            builder.addRecommendation("Ensure NVIDIA GPU drivers are installed: https://www.nvidia.com/Download/index.aspx");
            builder.addRecommendation("Verify CUDA is installed: https://developer.nvidia.com/cuda-downloads");
            builder.addRecommendation("Use CPUVectorIndex for CPU-only operation.");
        }

        // JCuda version
        builder.jcudaVersion(getJCudaVersion());

        // Java version checks
        String javaVersion = System.getProperty("java.version");
        if (javaVersion.startsWith("17") || javaVersion.startsWith("21")) {
            // Good versions
        } else {
            builder.addWarning("Recommended Java versions are 17 or 21. Current version: " + javaVersion);
        }

        return builder.build();
    }

    /**
     * Extracts GPU model from info string.
     */
    private static String extractGpuModel(String gpuInfo) {
        if (gpuInfo == null || gpuInfo.isEmpty()) {
            return "Unknown GPU";
        }
        // Format: "GPU: [name], Compute: X.Y, Memory: Z MB"
        if (gpuInfo.startsWith("GPU: ")) {
            int commaIndex = gpuInfo.indexOf(',');
            if (commaIndex > 5) {
                return gpuInfo.substring(5, commaIndex).trim();
            }
        }
        return gpuInfo;
    }

    /**
     * Extracts compute capability from info string.
     */
    private static String extractComputeCapability(String gpuInfo) {
        if (gpuInfo == null || !gpuInfo.contains("Compute:")) {
            return "Unknown";
        }
        int start = gpuInfo.indexOf("Compute:") + 8;
        int end = gpuInfo.indexOf(',', start);
        if (end == -1) {
            end = gpuInfo.length();
        }
        return gpuInfo.substring(start, end).trim();
    }

    /**
     * Gets JCuda version.
     */
    private static String getJCudaVersion() {
        try {
            // Try to get JCuda version from package
            Package jcudaPackage = Class.forName("jcuda.runtime.JCuda").getPackage();
            if (jcudaPackage != null && jcudaPackage.getImplementationVersion() != null) {
                return jcudaPackage.getImplementationVersion();
            }
        } catch (Exception e) {
            logger.debug("Could not determine JCuda version", e);
        }
        return "Unknown (12.0.0 expected)";
    }

    /**
     * Prints diagnostics report to console.
     */
    public static void printDiagnostics() {
        DiagnosticReport report = runDiagnostics();
        System.out.println(report.toString());
    }
}
