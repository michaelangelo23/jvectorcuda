package com.vindex.jvectorcuda;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Thread-safe wrapper around any VectorIndex implementation.
 * 
 * Uses a ReadWriteLock to allow concurrent reads (search operations)
 * while ensuring exclusive access for writes (add operations).
 * 
 * <p>Usage example:</p>
 * <pre>{@code
 * VectorIndex baseIndex = VectorIndexFactory.auto(384);
 * try (VectorIndex threadSafeIndex = new ThreadSafeVectorIndex(baseIndex)) {
 *     // Safe to use from multiple threads
 *     CompletableFuture.allOf(
 *         CompletableFuture.runAsync(() -> threadSafeIndex.add(vectors1)),
 *         CompletableFuture.runAsync(() -> threadSafeIndex.search(query, 10))
 *     ).join();
 * }
 * }</pre>
 * 
 * <p><b>Performance Characteristics:</b></p>
 * <ul>
 *   <li>Multiple threads can search concurrently (read lock)</li>
 *   <li>Add operations block all searches (write lock)</li>
 *   <li>Minimal overhead for single-threaded use (~10ns lock acquisition)</li>
 * </ul>
 * 
 * <p><b>Thread Safety Guarantees:</b></p>
 * <ul>
 *   <li>All operations are atomic and properly ordered</li>
 *   <li>No data races or lost updates</li>
 *   <li>Changes from add() visible to all subsequent search() calls</li>
 * </ul>
 */
public class ThreadSafeVectorIndex implements VectorIndex {
    
    private final VectorIndex delegate;
    private final ReadWriteLock rwLock;
    
    /**
     * Creates a thread-safe wrapper around an existing VectorIndex.
     * 
     * @param delegate the underlying (non-thread-safe) VectorIndex implementation
     * @throws IllegalArgumentException if delegate is null
     */
    public ThreadSafeVectorIndex(VectorIndex delegate) {
        if (delegate == null) {
            throw new IllegalArgumentException("Delegate VectorIndex cannot be null");
        }
        this.delegate = delegate;
        this.rwLock = new ReentrantReadWriteLock();
    }
    
    /**
     * Adds vectors to the index with exclusive write access.
     * Blocks all concurrent searches and other add operations.
     * 
     * @param vectors 2D array of shape [numVectors][dimensions]
     * @throws IllegalArgumentException if vectors are null, empty, or have mismatched dimensions
     */
    @Override
    public void add(float[][] vectors) {
        rwLock.writeLock().lock();
        try {
            delegate.add(vectors);
        } finally {
            rwLock.writeLock().unlock();
        }
    }
    
    /**
     * Searches for k nearest neighbors with shared read access.
     * Multiple threads can search concurrently without blocking each other.
     * 
     * @param query vector of shape [dimensions]
     * @param k number of nearest neighbors to return
     * @return SearchResult containing IDs and distances of k nearest neighbors
     * @throws IllegalArgumentException if query is null, empty, or has wrong dimensions
     */
    @Override
    public SearchResult search(float[] query, int k) {
        rwLock.readLock().lock();
        try {
            return delegate.search(query, k);
        } finally {
            rwLock.readLock().unlock();
        }
    }
    
    /**
     * Batch search for multiple queries with shared read access.
     * Multiple threads can perform batch searches concurrently.
     * 
     * @param queries 2D array of shape [numQueries][dimensions]
     * @param k number of nearest neighbors per query
     * @return List of SearchResults, one per query
     * @throws IllegalArgumentException if queries are null, empty, or have wrong dimensions
     */
    @Override
    public List<SearchResult> searchBatch(float[][] queries, int k) {
        rwLock.readLock().lock();
        try {
            return delegate.searchBatch(queries, k);
        } finally {
            rwLock.readLock().unlock();
        }
    }
    
    /**
     * Async search using CompletableFuture with read lock.
     * The search executes on a separate thread from the common ForkJoinPool.
     * 
     * @param query vector of shape [dimensions]
     * @param k number of nearest neighbors to return
     * @return CompletableFuture that will complete with search results
     */
    @Override
    public CompletableFuture<SearchResult> searchAsync(float[] query, int k) {
        // Use read lock for searchAsync to allow concurrent read-only GPU/index access,
        // consistent with the thread-safety guarantees of other search operations.
        return CompletableFuture.supplyAsync(() -> {
            rwLock.readLock().lock();
            try {
                return delegate.search(query, k);
            } finally {
                rwLock.readLock().unlock();
            }
        });
    }
    
    /**
     * Returns the dimensionality of vectors in this index.
     * This is a read-only property and doesn't require locking.
     * 
     * @return number of dimensions
     */
    @Override
    public int getDimensions() {
        // getDimensions is immutable after construction, no lock needed
        return delegate.getDimensions();
    }
    
    /**
     * Returns the current number of vectors in the index.
     * Acquires read lock to ensure consistent view.
     * 
     * @return number of vectors
     */
    @Override
    public int size() {
        rwLock.readLock().lock();
        try {
            return delegate.size();
        } finally {
            rwLock.readLock().unlock();
        }
    }
    
    /**
     * Closes the underlying VectorIndex and releases all resources.
     * This method acquires write lock to ensure no operations are in progress.
     */
    @Override
    public void close() {
        rwLock.writeLock().lock();
        try {
            delegate.close();
        } finally {
            rwLock.writeLock().unlock();
        }
    }
}
