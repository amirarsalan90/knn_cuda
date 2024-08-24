#include "knn_cuda.h"
#include <cmath>
#include <cstdio> // Include cstdio for printf support

// Single-point CUDA kernel for calculating Euclidean distances
__global__ void calculate_distances(const float* X_train, const float* X_test,
                                    float* distances, int n_train,
                                    int n_features) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n_train) {
        float dist = 0.0f;
        for (int i = 0; i < n_features; i++) {
            float diff = X_train[idx * n_features + i] - X_test[i];
            dist += diff * diff;
        }
        distances[idx] = sqrtf(dist);
    }
}

// Batch CUDA kernel for calculating Euclidean distances
__global__ void calculate_distances_batch(const float* X_train,
                                          const float* X_test, float* distances,
                                          int n_train, int n_test,
                                          int n_features) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n_test * n_train) {
        int test_idx = idx / n_train;
        int train_idx = idx % n_train;

        float dist = 0.0f;
        for (int i = 0; i < n_features; i++) {
            float diff = X_train[train_idx * n_features + i] -
                         X_test[test_idx * n_features + i];
            dist += diff * diff;
        }
        distances[test_idx * n_train + train_idx] = sqrtf(dist);
    }
}

// Wrapper function for single-point CUDA kernel
void launch_cuda_knn(const float* d_X_train, const float* d_X_test,
                     float* d_distances, int n_train, int n_features,
                     int blockSize) {
    int numBlocks = (n_train + blockSize - 1) / blockSize;
    calculate_distances<<<numBlocks, blockSize>>>(
        d_X_train, d_X_test, d_distances, n_train, n_features);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

// Wrapper function for batch CUDA kernel
void launch_cuda_knn_batch(const float* d_X_train, const float* d_X_test,
                           float* d_distances, int n_train, int n_test,
                           int n_features, int blockSize) {
    int numBlocks = (n_test * n_train + blockSize - 1) / blockSize;
    calculate_distances_batch<<<numBlocks, blockSize>>>(
        d_X_train, d_X_test, d_distances, n_train, n_test, n_features);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}
