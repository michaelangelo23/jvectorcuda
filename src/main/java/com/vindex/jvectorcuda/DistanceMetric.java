package com.vindex.jvectorcuda;

/**
 * Distance metrics for vector similarity comparison.
 *
 * <p>All metrics are normalized so that <b>smaller values indicate higher similarity</b>.
 * This allows consistent sorting regardless of metric.
 *
 * <h2>Metric Comparison</h2>
 * <table border="1">
 *   <tr><th>Metric</th><th>Formula</th><th>Range</th><th>Best For</th></tr>
 *   <tr>
 *     <td>EUCLIDEAN</td>
 *     <td>√Σ(aᵢ - bᵢ)²</td>
 *     <td>[0, ∞)</td>
 *     <td>General purpose, unnormalized vectors</td>
 *   </tr>
 *   <tr>
 *     <td>COSINE</td>
 *     <td>1 - (a·b)/(|a||b|)</td>
 *     <td>[0, 2]</td>
 *     <td>Text embeddings, normalized vectors</td>
 *   </tr>
 *   <tr>
 *     <td>INNER_PRODUCT</td>
 *     <td>-a·b</td>
 *     <td>(-∞, ∞)</td>
 *     <td>Maximum inner product search (MIPS)</td>
 *   </tr>
 * </table>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * // Create index with specific metric
 * VectorIndex index = VectorIndexFactory.gpu(384, DistanceMetric.COSINE);
 *
 * // Or use default (Euclidean)
 * VectorIndex defaultIndex = VectorIndexFactory.auto(768);
 * }</pre>
 *
 * <h2>Choosing a Metric</h2>
 * <ul>
 *   <li><b>EUCLIDEAN:</b> Use for general-purpose vector similarity when magnitude matters</li>
 *   <li><b>COSINE:</b> Use for text/document embeddings where direction matters more than magnitude</li>
 *   <li><b>INNER_PRODUCT:</b> Use for recommendation systems or when vectors are already normalized</li>
 * </ul>
 *
 * @see VectorIndexFactory#auto(int, DistanceMetric)
 * @see VectorIndexFactory#gpu(int, DistanceMetric)
 * @since 1.0.0
 */
public enum DistanceMetric {

    /**
     * Euclidean (L2) distance.
     *
     * <p>Formula: √Σ(aᵢ - bᵢ)²
     * <p>Range: [0, ∞) where 0 = identical vectors
     */
    EUCLIDEAN("euclideanDistance", "euclidean_distance.ptx"),

    /**
     * Cosine distance (1 - cosine similarity).
     *
     * <p>Formula: 1 - (a·b)/(|a||b|)
     * <p>Range: [0, 2] where 0 = identical direction, 2 = opposite direction
     */
    COSINE("cosineSimilarity", "cosine_similarity.ptx"),

    /**
     * Negative inner product (for maximum inner product search).
     *
     * <p>Formula: -a·b (negated so smaller = more similar)
     * <p>Range: (-∞, ∞) where more negative = higher original similarity
     */
    INNER_PRODUCT("innerProduct", "inner_product.ptx");

    private final String kernelName;
    private final String ptxFile;

    DistanceMetric(String kernelName, String ptxFile) {
        this.kernelName = kernelName;
        this.ptxFile = ptxFile;
    }

    /**
     * Returns the CUDA kernel function name for this metric.
     *
     * @return the kernel function name in the PTX file
     */
    public String getKernelName() {
        return kernelName;
    }

    /**
     * Returns the PTX file name containing the CUDA kernel.
     *
     * @return the PTX filename (without path)
     */
    public String getPtxFile() {
        return ptxFile;
    }
}
