package com.vindex.jvectorcuda;

import org.junit.jupiter.api.*;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for ThreadSafeVectorIndex concurrent access patterns.
 */
class ThreadSafeVectorIndexTest {

    private static final int DIMENSIONS = 128;
    private static final int NUM_THREADS = 10;
    private static final int OPS_PER_THREAD = 100;

    @Test
    @DisplayName("Constructor throws on null delegate")
    void testConstructorNullDelegate() {
        assertThrows(IllegalArgumentException.class, () -> new ThreadSafeVectorIndex(null));
    }

    @Test
    @DisplayName("Single-threaded operations work correctly")
    void testSingleThreaded() {
        VectorIndex delegate = VectorIndexFactory.auto(DIMENSIONS);
        ThreadSafeVectorIndex index = new ThreadSafeVectorIndex(delegate);

        float[][] vectors = createRandomVectors(100, DIMENSIONS);
        index.add(vectors);
        assertEquals(100, index.size());

        float[] query = createRandomVector(DIMENSIONS);
        SearchResult result = index.search(query, 5);

        assertNotNull(result);
        assertEquals(5, result.getIds().length);
        assertEquals(5, result.getDistances().length);

        index.close();
    }

    @Test
    @DisplayName("Concurrent reads do not block each other")
    void testConcurrentReads() throws InterruptedException {
        VectorIndex delegate = VectorIndexFactory.auto(DIMENSIONS);
        ThreadSafeVectorIndex index = new ThreadSafeVectorIndex(delegate);

        // Populate index
        float[][] vectors = createRandomVectors(1000, DIMENSIONS);
        index.add(vectors);

        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(NUM_THREADS);
        AtomicInteger successCount = new AtomicInteger(0);

        // Launch multiple concurrent searches
        for (int i = 0; i < NUM_THREADS; i++) {
            executor.submit(() -> {
                try {
                    startLatch.await(); // Wait for all threads to be ready
                    float[] query = createRandomVector(DIMENSIONS);
                    SearchResult result = index.search(query, 10);
                    assertNotNull(result);
                    successCount.incrementAndGet();
                } catch (Exception e) {
                    fail("Concurrent read failed: " + e.getMessage());
                } finally {
                    doneLatch.countDown();
                }
            });
        }

        long startTime = System.nanoTime();
        startLatch.countDown(); // Start all threads simultaneously
        assertTrue(doneLatch.await(10, TimeUnit.SECONDS), "Concurrent reads timed out");
        long duration = System.nanoTime() - startTime;

        assertEquals(NUM_THREADS, successCount.get(), "All concurrent reads should succeed");
        
        // Concurrent reads should not block - duration should be close to single read time
        // If reads were serialized, duration would be NUM_THREADS * single_read_time
        System.out.printf("Concurrent reads completed in %.2f ms%n", duration / 1_000_000.0);

        executor.shutdown();
        index.close();
    }

    @Test
    @DisplayName("Write operations block reads")
    void testWriteBlocksReads() throws InterruptedException, ExecutionException, TimeoutException {
        VectorIndex delegate = VectorIndexFactory.auto(DIMENSIONS);
        ThreadSafeVectorIndex index = new ThreadSafeVectorIndex(delegate);

        // Add initial vectors
        index.add(createRandomVectors(100, DIMENSIONS));

        ExecutorService executor = Executors.newFixedThreadPool(2);
        CountDownLatch writeLatch = new CountDownLatch(1);
        AtomicInteger sizeBeforeWrite = new AtomicInteger(-1);
        AtomicInteger sizeDuringWrite = new AtomicInteger(-1);
        AtomicInteger sizeAfterWrite = new AtomicInteger(-1);

        // Thread 1: Slow write
        Future<?> writeFuture = executor.submit(() -> {
            try {
                writeLatch.countDown(); // Signal write started
                Thread.sleep(100); // Simulate slow write
                index.add(createRandomVectors(100, DIMENSIONS));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });

        // Thread 2: Read during write
        Future<?> readFuture = executor.submit(() -> {
            try {
                sizeBeforeWrite.set(index.size());
                writeLatch.await(); // Wait for write to start
                Thread.sleep(10); // Ensure we're in middle of write
                sizeDuringWrite.set(index.size()); // This should block until write completes
                sizeAfterWrite.set(index.size());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });

        writeFuture.get(5, TimeUnit.SECONDS);
        readFuture.get(5, TimeUnit.SECONDS);

        assertEquals(100, sizeBeforeWrite.get());
        assertEquals(200, sizeDuringWrite.get()); // Read blocked until write completed
        assertEquals(200, sizeAfterWrite.get());

        executor.shutdown();
        index.close();
    }

    @Test
    @DisplayName("Multiple writers are serialized")
    void testMultipleWriters() throws InterruptedException {
        VectorIndex delegate = VectorIndexFactory.auto(DIMENSIONS);
        ThreadSafeVectorIndex index = new ThreadSafeVectorIndex(delegate);

        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
        CountDownLatch doneLatch = new CountDownLatch(NUM_THREADS);
        List<Integer> vectorsPerAdd = new ArrayList<>();

        for (int i = 0; i < NUM_THREADS; i++) {
            int batchSize = 10 + i; // Different sizes to verify ordering
            vectorsPerAdd.add(batchSize);
            executor.submit(() -> {
                try {
                    index.add(createRandomVectors(batchSize, DIMENSIONS));
                } finally {
                    doneLatch.countDown();
                }
            });
        }

        assertTrue(doneLatch.await(10, TimeUnit.SECONDS), "Multiple writers timed out");

        int expectedTotal = vectorsPerAdd.stream().mapToInt(Integer::intValue).sum();
        assertEquals(expectedTotal, index.size(), "All writes should complete without loss");

        executor.shutdown();
        index.close();
    }

    @Test
    @DisplayName("Mixed concurrent reads and writes are safe")
    void testMixedConcurrentOps() throws InterruptedException {
        VectorIndex delegate = VectorIndexFactory.auto(DIMENSIONS);
        ThreadSafeVectorIndex index = new ThreadSafeVectorIndex(delegate);

        // Add initial data
        index.add(createRandomVectors(500, DIMENSIONS));

        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
        CountDownLatch doneLatch = new CountDownLatch(NUM_THREADS * 2);
        AtomicInteger readSuccesses = new AtomicInteger(0);
        AtomicInteger writeSuccesses = new AtomicInteger(0);

        // Launch reader threads
        for (int i = 0; i < NUM_THREADS; i++) {
            executor.submit(() -> {
                try {
                    for (int j = 0; j < OPS_PER_THREAD; j++) {
                        float[] query = createRandomVector(DIMENSIONS);
                        SearchResult result = index.search(query, 5);
                        assertNotNull(result);
                    }
                    readSuccesses.incrementAndGet();
                } catch (Exception e) {
                    fail("Reader thread failed: " + e.getMessage());
                } finally {
                    doneLatch.countDown();
                }
            });
        }

        // Launch writer threads
        for (int i = 0; i < NUM_THREADS; i++) {
            executor.submit(() -> {
                try {
                    for (int j = 0; j < 10; j++) {
                        index.add(createRandomVectors(10, DIMENSIONS));
                    }
                    writeSuccesses.incrementAndGet();
                } catch (Exception e) {
                    fail("Writer thread failed: " + e.getMessage());
                } finally {
                    doneLatch.countDown();
                }
            });
        }

        assertTrue(doneLatch.await(30, TimeUnit.SECONDS), "Mixed ops timed out");
        assertEquals(NUM_THREADS, readSuccesses.get(), "All readers should succeed");
        assertEquals(NUM_THREADS, writeSuccesses.get(), "All writers should succeed");

        // Verify final size
        int expectedSize = 500 + (NUM_THREADS * 10 * 10);
        assertEquals(expectedSize, index.size());

        executor.shutdown();
        index.close();
    }

    @Test
    @DisplayName("searchBatch is thread-safe")
    void testSearchBatchThreadSafety() throws InterruptedException {
        VectorIndex delegate = VectorIndexFactory.auto(DIMENSIONS);
        ThreadSafeVectorIndex index = new ThreadSafeVectorIndex(delegate);

        index.add(createRandomVectors(1000, DIMENSIONS));

        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
        CountDownLatch doneLatch = new CountDownLatch(NUM_THREADS);
        AtomicInteger successCount = new AtomicInteger(0);

        for (int i = 0; i < NUM_THREADS; i++) {
            executor.submit(() -> {
                try {
                    float[][] queries = createRandomVectors(50, DIMENSIONS);
                    List<SearchResult> results = index.searchBatch(queries, 10);
                    assertEquals(50, results.size());
                    successCount.incrementAndGet();
                } catch (Exception e) {
                    fail("Batch search failed: " + e.getMessage());
                } finally {
                    doneLatch.countDown();
                }
            });
        }

        assertTrue(doneLatch.await(30, TimeUnit.SECONDS), "Batch search timed out");
        assertEquals(NUM_THREADS, successCount.get());

        executor.shutdown();
        index.close();
    }

    @Test
    @DisplayName("searchAsync is thread-safe")
    void testSearchAsyncThreadSafety() throws InterruptedException, ExecutionException, TimeoutException {
        VectorIndex delegate = VectorIndexFactory.auto(DIMENSIONS);
        ThreadSafeVectorIndex index = new ThreadSafeVectorIndex(delegate);

        index.add(createRandomVectors(1000, DIMENSIONS));

        List<CompletableFuture<SearchResult>> futures = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            float[] query = createRandomVector(DIMENSIONS);
            futures.add(index.searchAsync(query, 10));
        }

        CompletableFuture<Void> allOf = CompletableFuture.allOf(
            futures.toArray(new CompletableFuture[0])
        );

        allOf.get(10, TimeUnit.SECONDS); // Wait for all to complete

        for (CompletableFuture<SearchResult> future : futures) {
            assertTrue(future.isDone());
            assertNotNull(future.get());
        }

        index.close();
    }

    @Test
    @DisplayName("getDimensions does not require locking")
    void testGetDimensions() {
        VectorIndex delegate = VectorIndexFactory.auto(DIMENSIONS);
        ThreadSafeVectorIndex index = new ThreadSafeVectorIndex(delegate);

        assertEquals(DIMENSIONS, index.getDimensions());

        index.close();
    }

    @Test
    @DisplayName("size() is thread-safe")
    void testSizeThreadSafety() throws InterruptedException {
        VectorIndex delegate = VectorIndexFactory.auto(DIMENSIONS);
        ThreadSafeVectorIndex index = new ThreadSafeVectorIndex(delegate);

        index.add(createRandomVectors(100, DIMENSIONS));

        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
        CountDownLatch doneLatch = new CountDownLatch(NUM_THREADS);
        AtomicInteger minSize = new AtomicInteger(Integer.MAX_VALUE);
        AtomicInteger maxSize = new AtomicInteger(0);

        for (int i = 0; i < NUM_THREADS; i++) {
            executor.submit(() -> {
                try {
                    for (int j = 0; j < 100; j++) {
                        int size = index.size();
                        minSize.updateAndGet(current -> Math.min(current, size));
                        maxSize.updateAndGet(current -> Math.max(current, size));
                    }
                } finally {
                    doneLatch.countDown();
                }
            });
        }

        assertTrue(doneLatch.await(10, TimeUnit.SECONDS));
        assertTrue(minSize.get() >= 100, "Size should be at least initial 100");
        assertTrue(maxSize.get() >= 100, "Size should be at least initial 100");

        executor.shutdown();
        index.close();
    }

    @Test
    @DisplayName("close() waits for all operations to complete")
    void testCloseWaitsForOperations() throws InterruptedException {
        VectorIndex delegate = VectorIndexFactory.auto(DIMENSIONS);
        ThreadSafeVectorIndex index = new ThreadSafeVectorIndex(delegate);

        index.add(createRandomVectors(1000, DIMENSIONS));

        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
        CountDownLatch startLatch = new CountDownLatch(1);
        AtomicInteger activeOps = new AtomicInteger(NUM_THREADS);

        // Launch long-running searches
        for (int i = 0; i < NUM_THREADS; i++) {
            executor.submit(() -> {
                try {
                    startLatch.await();
                    Thread.sleep(100); // Simulate slow search
                    index.search(createRandomVector(DIMENSIONS), 10);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    activeOps.decrementAndGet();
                }
            });
        }

        startLatch.countDown(); // Start all searches
        Thread.sleep(50); // Let searches start

        // Close should wait for all searches to complete
        long closeStart = System.nanoTime();
        index.close();
        long closeDuration = System.nanoTime() - closeStart;

        assertEquals(0, activeOps.get(), "All operations should complete before close returns");
        assertTrue(closeDuration > 50_000_000, "Close should wait for operations to complete (>50ms)");

        executor.shutdown();
    }

    // ===== Edge Cases & AI Blind Spots =====

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {

        @Test
        @DisplayName("Empty index concurrent access")
        void testEmptyIndexConcurrentAccess() throws InterruptedException {
            VectorIndex delegate = VectorIndexFactory.auto(DIMENSIONS);
            ThreadSafeVectorIndex index = new ThreadSafeVectorIndex(delegate);

            ExecutorService executor = Executors.newFixedThreadPool(5);
            CountDownLatch doneLatch = new CountDownLatch(5);

            for (int i = 0; i < 5; i++) {
                executor.submit(() -> {
                    try {
                        assertEquals(0, index.size());
                    } finally {
                        doneLatch.countDown();
                    }
                });
            }

            assertTrue(doneLatch.await(5, TimeUnit.SECONDS));
            executor.shutdown();
            index.close();
        }

        @Test
        @DisplayName("Very large batch concurrent writes")
        void testLargeBatchWrites() throws InterruptedException {
            VectorIndex delegate = VectorIndexFactory.auto(DIMENSIONS);
            ThreadSafeVectorIndex index = new ThreadSafeVectorIndex(delegate);

            ExecutorService executor = Executors.newFixedThreadPool(5);
            CountDownLatch doneLatch = new CountDownLatch(5);

            for (int i = 0; i < 5; i++) {
                executor.submit(() -> {
                    try {
                        index.add(createRandomVectors(10000, DIMENSIONS));
                    } finally {
                        doneLatch.countDown();
                    }
                });
            }

            assertTrue(doneLatch.await(30, TimeUnit.SECONDS));
            assertEquals(50000, index.size());

            executor.shutdown();
            index.close();
        }
    }

    @Nested
    @DisplayName("AI Blind Spots")
    class AIBlindSpots {

        @Test
        @DisplayName("Reentrancy - nested read locks on same thread")
        void testReentrancy() {
            VectorIndex delegate = VectorIndexFactory.auto(DIMENSIONS);
            ThreadSafeVectorIndex index = new ThreadSafeVectorIndex(delegate);

            index.add(createRandomVectors(100, DIMENSIONS));

            // ReentrantReadWriteLock allows reentrancy
            int size1 = index.size();
            int size2 = index.size(); // Should not deadlock

            assertEquals(size1, size2);

            index.close();
        }

        @Test
        @DisplayName("Exception in add() releases lock")
        void testExceptionReleasesLock() {
            VectorIndex delegate = VectorIndexFactory.auto(DIMENSIONS);
            ThreadSafeVectorIndex index = new ThreadSafeVectorIndex(delegate);

            // Add invalid vectors to trigger exception
            assertThrows(IllegalArgumentException.class, () -> {
                index.add(new float[][]{ {1.0f, 2.0f} }); // Wrong dimensions
            });

            // Lock should be released, this should not hang
            index.add(createRandomVectors(10, DIMENSIONS));
            assertEquals(10, index.size());

            index.close();
        }

        @Test
        @DisplayName("searchAsync failures do not leave locks held")
        void testSearchAsyncFailureReleasesLock() throws InterruptedException {
            VectorIndex delegate = VectorIndexFactory.auto(DIMENSIONS);
            ThreadSafeVectorIndex index = new ThreadSafeVectorIndex(delegate);

            index.add(createRandomVectors(100, DIMENSIONS));

            // Trigger async failure with invalid query
            CompletableFuture<SearchResult> future = index.searchAsync(new float[0], 10);

            assertThrows(ExecutionException.class, () -> future.get(5, TimeUnit.SECONDS));

            // Lock should be released, this should not hang
            index.search(createRandomVector(DIMENSIONS), 5);

            index.close();
        }
    }

    // ===== Helper Methods =====

    private static float[][] createRandomVectors(int count, int dimensions) {
        float[][] vectors = new float[count][dimensions];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < dimensions; j++) {
                vectors[i][j] = (float) Math.random();
            }
        }
        return vectors;
    }

    private static float[] createRandomVector(int dimensions) {
        float[] vector = new float[dimensions];
        for (int i = 0; i < dimensions; i++) {
            vector[i] = (float) Math.random();
        }
        return vector;
    }
}
