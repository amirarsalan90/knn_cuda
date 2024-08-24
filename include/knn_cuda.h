#ifndef KNN_CUDA_H
#define KNN_CUDA_H

#include <cuda_runtime.h>

void launch_cuda_knn(const float* d_X_train, const float* d_X_test,
                     float* d_distances, int n_train, int n_features,
                     int blockSize);

void launch_cuda_knn_batch(const float* d_X_train, const float* d_X_test,
                           float* d_distances, int n_train, int n_test,
                           int n_features, int blockSize);

#endif // KNN_CUDA_H
