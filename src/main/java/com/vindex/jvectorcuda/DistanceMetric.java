package com.vindex.jvectorcuda;

// Distance metrics for vector similarity. Smaller = more similar.
public enum DistanceMetric {
    
    EUCLIDEAN("euclideanDistance", "euclidean_distance.ptx"),
    COSINE("cosineSimilarity", "cosine_similarity.ptx"),
    INNER_PRODUCT("innerProduct", "inner_product.ptx");
    
    private final String kernelName;
    private final String ptxFile;
    
    DistanceMetric(String kernelName, String ptxFile) {
        this.kernelName = kernelName;
        this.ptxFile = ptxFile;
    }
    
    public String getKernelName() {
        return kernelName;
    }
    
    public String getPtxFile() {
        return ptxFile;
    }
}
