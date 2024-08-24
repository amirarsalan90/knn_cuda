#include "knn_classifier.h"
#include "knn_cuda.h" 
#include <algorithm>
#include <chrono> 
#include <cmath>
#include <cuda_runtime.h>
#include <iostream> 
#include <map>

template <typename T> KNNClassifier<T>::KNNClassifier(int k) : k(k) {
}

template <typename T>
void KNNClassifier<T>::fit(const std::vector<std::vector<T>>& X_train,
                           const std::vector<int>& y_train) {
    using namespace std::chrono;

    this->X_train = X_train;
    this->y_train = y_train;

    int n_train = X_train.size();
    int n_features = X_train[0].size();

    auto start_flatten = high_resolution_clock::now();

    X_train_flattened.resize(n_train * n_features);
    for (int i = 0; i < n_train; ++i) {
        for (int j = 0; j < n_features; ++j) {
            X_train_flattened[i * n_features + j] = X_train[i][j];
        }
    }

    auto end_flatten = high_resolution_clock::now();
    auto duration_flatten =
        duration_cast<milliseconds>(end_flatten - start_flatten).count();
    std::cout << "Time taken for flattening data in fit(): " << duration_flatten
              << " ms" << std::endl;
}

template <typename T>
std::vector<int>
KNNClassifier<T>::predict(const std::vector<std::vector<T>>& X_test) {
    if (X_test.size() == 1) {
        return {predict_single_cuda(X_test[0])};
    } else {
        return predict_batch_cuda(X_test);
    }
}

template <typename T>
int KNNClassifier<T>::predict_single_cuda(const std::vector<T>& x_test) {
    using namespace std::chrono;

    int n_train = X_train.size();
    int n_features = X_train[0].size();

    auto start_mem = high_resolution_clock::now();

    float *d_X_train, *d_X_test, *d_distances;
    cudaMalloc(&d_X_train, n_train * n_features * sizeof(float));
    cudaMalloc(&d_X_test, n_features * sizeof(float));
    cudaMalloc(&d_distances, n_train * sizeof(float));

    cudaMemcpy(d_X_train, X_train_flattened.data(),
               n_train * n_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X_test, x_test.data(), n_features * sizeof(float),
               cudaMemcpyHostToDevice);

    auto end_mem = high_resolution_clock::now();
    auto duration_mem =
        duration_cast<milliseconds>(end_mem - start_mem).count();
    std::cout << "Time taken for memory allocation and data transfer: "
              << duration_mem << " ms" << std::endl;

    auto start_kernel = high_resolution_clock::now();

    int blockSize = 1024;
    launch_cuda_knn(d_X_train, d_X_test, d_distances, n_train, n_features,
                    blockSize);

    auto end_kernel = high_resolution_clock::now();
    auto duration_kernel =
        duration_cast<milliseconds>(end_kernel - start_kernel).count();
    std::cout << "Time taken for kernel execution: " << duration_kernel << " ms"
              << std::endl;

    auto start_post = high_resolution_clock::now();

    std::vector<float> distances(n_train);
    cudaMemcpy(distances.data(), d_distances, n_train * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_X_train);
    cudaFree(d_X_test);
    cudaFree(d_distances);

    std::vector<std::pair<float, int>> dist_label_pairs;
    for (int i = 0; i < n_train; ++i) {
        dist_label_pairs.push_back({distances[i], y_train[i]});
    }
    std::sort(dist_label_pairs.begin(), dist_label_pairs.end());

    std::vector<int> neighbor_labels(k);
    for (int i = 0; i < k; ++i) {
        neighbor_labels[i] = dist_label_pairs[i].second;
    }

    auto end_post = high_resolution_clock::now();
    auto duration_post =
        duration_cast<milliseconds>(end_post - start_post).count();
    std::cout << "Time taken for post-processing: " << duration_post << " ms"
              << std::endl;

    return most_frequent_label(neighbor_labels);
}

template <typename T>
std::vector<int> KNNClassifier<T>::predict_batch_cuda(
    const std::vector<std::vector<T>>& X_test) {
    using namespace std::chrono;

    int n_test = X_test.size();
    int n_train = X_train.size();
    int n_features = X_train[0].size();

    auto start_mem = high_resolution_clock::now();

    float *d_X_train, *d_X_test, *d_distances;
    cudaMalloc(&d_X_train, n_train * n_features * sizeof(float));
    cudaMalloc(&d_X_test, n_test * n_features * sizeof(float));
    cudaMalloc(&d_distances, n_test * n_train * sizeof(float));

    std::vector<float> X_test_flattened(n_test * n_features);
    for (int i = 0; i < n_test; ++i) {
        for (int j = 0; j < n_features; ++j) {
            X_test_flattened[i * n_features + j] = X_test[i][j];
        }
    }

    cudaMemcpy(d_X_train, X_train_flattened.data(),
               n_train * n_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X_test, X_test_flattened.data(),
               n_test * n_features * sizeof(float), cudaMemcpyHostToDevice);

    auto end_mem = high_resolution_clock::now();
    auto duration_mem =
        duration_cast<milliseconds>(end_mem - start_mem).count();
    std::cout << "Time taken for memory allocation and data transfer: "
              << duration_mem << " ms" << std::endl;

    auto start_kernel = high_resolution_clock::now();

    int blockSize = 1024;
    launch_cuda_knn_batch(d_X_train, d_X_test, d_distances, n_train, n_test,
                          n_features, blockSize);

    auto end_kernel = high_resolution_clock::now();
    auto duration_kernel =
        duration_cast<milliseconds>(end_kernel - start_kernel).count();
    std::cout << "Time taken for kernel execution: " << duration_kernel << " ms"
              << std::endl;

    auto start_post = high_resolution_clock::now();

    std::vector<float> distances(n_test * n_train);
    cudaMemcpy(distances.data(), d_distances, n_test * n_train * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_X_train);
    cudaFree(d_X_test);
    cudaFree(d_distances);

    std::vector<int> predictions(n_test);

    for (int t = 0; t < n_test; ++t) {
        std::vector<std::pair<float, int>> dist_label_pairs;
        for (int i = 0; i < n_train; ++i) {
            dist_label_pairs.push_back(
                {distances[t * n_train + i], y_train[i]});
        }
        std::sort(dist_label_pairs.begin(), dist_label_pairs.end());

        std::vector<int> neighbor_labels(k);
        for (int i = 0; i < k; ++i) {
            neighbor_labels[i] = dist_label_pairs[i].second;
        }

        predictions[t] = most_frequent_label(neighbor_labels);
    }

    auto end_post = high_resolution_clock::now();
    auto duration_post =
        duration_cast<milliseconds>(end_post - start_post).count();
    std::cout << "Time taken for post-processing: " << duration_post << " ms"
              << std::endl;

    return predictions;
}

template <typename T>
int KNNClassifier<T>::most_frequent_label(
    const std::vector<int>& labels) const {
    std::map<int, int> label_count;
    for (const int& label : labels) {
        label_count[label]++;
    }

    int most_frequent = -1;
    int max_count = 0;
    for (const auto& pair : label_count) {
        if (pair.second > max_count) {
            most_frequent = pair.first;
            max_count = pair.second;
        }
    }
    return most_frequent;
}

template class KNNClassifier<float>;
template class KNNClassifier<double>;
template class KNNClassifier<int>;
