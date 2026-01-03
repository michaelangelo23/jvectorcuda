package com.vindex.jvectorcuda.benchmarks;

import com.vindex.jvectorcuda.VectorIndex;
import com.vindex.jvectorcuda.VectorIndexFactory;
import com.vindex.jvectorcuda.exception.GpuOutOfMemoryException;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class GpuOomTest {

    @Test
    public void testGracefulOomHandling() {
        // Attempt to allocate impossible amount of memory (e.g. 10 billion floats =
        // 40GB)
        // This should trigger OOM on most GPUs
        int dimensions = 1024;

        System.out.println("Attempting to allocate ~40GB VRAM to trigger OOM...");

        try (VectorIndex index = VectorIndexFactory.gpu(dimensions)) {
            // Create a large batch of vectors
            int batchSize = 100_000;
            float[][] batch = new float[batchSize][dimensions];

            // Fill with dummy data
            for (int i = 0; i < batchSize; i++) {
                batch[i][0] = 1.0f;
            }

            // Keep adding until OOM
            // 40GB target / (100k * 1024 * 4 bytes) ~= 100 iterations
            for (int i = 0; i < 200; i++) {
                System.out.println("Adding batch " + i + "...");
                index.add(batch);
            }

            fail("Expected GpuOutOfMemoryException was not thrown");
        } catch (GpuOutOfMemoryException e) {
            System.out.println("Caught expected OOM exception: " + e.getMessage());
            assertNotNull(e.getMessage());
            assertTrue(e.getMessage().contains("GPU Out of Memory"));
        } catch (java.lang.OutOfMemoryError e) {
            // Catch Java Heap OOM if we accidentally blow up JVM RAM instead of VRAM
            System.out.println("Caught Java Heap OOM (Test Setup Issue): " + e.getMessage());
        } catch (Exception e) {
            fail("Caught unexpected exception type: " + e.getClass().getName());
        }
    }
}
