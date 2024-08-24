#include "csv_utils.h" // The file where your CSV utility functions are
#include "knn_classifier.h"
#include <iostream>
#include <string>
#include <vector>

int main() {
    // Load training and test data
    std::vector<std::vector<float>> X_train =
        read_csv("../python_code/"
                 "train_features.csv");
    std::vector<int> y_train = read_labels(
        "../python_code/train_labels.csv");
    std::vector<std::vector<float>> X_test = read_csv(
        "../python_code/test_features.csv");
    std::vector<int> y_test = read_labels(
        "../python_code/test_labels.csv");

    // Initialize KNN classifier with K = 5
    KNNClassifier<float> knn(5);
    knn.fit(X_train, y_train);

    // Predict on the test set
    std::vector<int> predictions = knn.predict(X_test);

    // Save predictions to a CSV file
    save_predictions(predictions, "predictions.csv");

    return 0;
}
