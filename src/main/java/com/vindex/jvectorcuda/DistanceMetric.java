package com.vindex.jvectorcuda;

/**
 * Supported distance metrics for vector similarity search.
 * 
 * <p>Each metric defines how "distance" (dissimilarity) between vectors is computed.
 * Smaller distances indicate more similar vectors.
 * 
 * <h2>Metric Selection Guide:</h2>
 * <ul>
 *   <li><b>EUCLIDEAN:</b> L2 distance. Best for dense, unnormalized vectors.
 *       Common in image embeddings and traditional ML.</li>
 *   <li><b>COSINE:</b> 1 - cosine_similarity. Best when magnitude doesn't matter,
 *       only direction. Common in NLP (TF-IDF, sentence embeddings).</li>
 *   <li><b>INNER_PRODUCT:</b> Negative dot product. Best for pre-normalized vectors
 *       (e.g., sentence-transformers). Equivalent to cosine but faster.</li>
 * </ul>
 * 
 * @author JVectorCUDA (AI-assisted, Human-verified)
 * @since 1.0.0
 */
public enum DistanceMetric {
    
    /**
     * Euclidean (L2) distance.
     * <p>Formula: √Σ(a_i - b_i)²
     * <p>Range: [0, ∞)
     * <p>Use case: Dense embeddings, image features, traditional clustering.
     */
    EUCLIDEAN("euclideanDistance", "euclidean_distance.ptx"),
    
    /**
     * Cosine distance (1 - cosine_similarity).
     * <p>Formula: 1 - (A·B)/(||A||·||B||)
     * <p>Range: [0, 2] where 0=identical, 1=orthogonal, 2=opposite
     * <p>Use case: Text embeddings, when vector magnitude varies.
     */
    COSINE("cosineSimilarity", "cosine_similarity.ptx"),
    
    /**
     * Negative inner product (dot product).
     * <p>Formula: -Σ(a_i · b_i)
     * <p>Range: (-∞, ∞)
     * <p>Use case: Pre-normalized vectors (e.g., sentence-transformers).
     *    Equivalent to cosine distance when vectors are unit length.
     */
    INNER_PRODUCT("innerProduct", "inner_product.ptx");
    
    private final String kernelName;
    private final String ptxFile;
    
    DistanceMetric(String kernelName, String ptxFile) {
        this.kernelName = kernelName;
        this.ptxFile = ptxFile;
    }
    
    /**
     * Returns the CUDA kernel function name.
     * @return kernel function name (e.g., "euclideanDistance")
     */
    public String getKernelName() {
        return kernelName;
    }
    
    /**
     * Returns the PTX file containing this kernel.
     * @return PTX filename (e.g., "euclidean_distance.ptx")
     */
    public String getPtxFile() {
        return ptxFile;
    }
}
