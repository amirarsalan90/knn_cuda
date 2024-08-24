#ifndef KNN_CLASSIFIER_H
#define KNN_CLASSIFIER_H

#include <vector>

template <typename T> class KNNClassifier {
public:
    KNNClassifier(int k);
    void fit(const std::vector<std::vector<T>>& X_train,
             const std::vector<int>& y_train);
    std::vector<int> predict(const std::vector<std::vector<T>>& X_test);

private:
    int k;
    std::vector<std::vector<T>> X_train;
    std::vector<int> y_train;
    std::vector<float> X_train_flattened;

    T euclidean_distance(const std::vector<T>& a,
                         const std::vector<T>& b) const;
    int predict_single(const std::vector<T>& x);
    int most_frequent_label(const std::vector<int>& labels) const;

    // Declare the functions for CUDA
    int predict_single_cuda(const std::vector<T>& x_test);
    std::vector<int>
    predict_batch_cuda(const std::vector<std::vector<T>>& X_test);
};

#endif // KNN_CLASSIFIER_H
