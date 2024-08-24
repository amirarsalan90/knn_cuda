#include "csv_utils.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// Function to read a CSV file into a vector of vectors
std::vector<std::vector<float>> read_csv(const std::string& filename) {
    std::vector<std::vector<float>> data;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stof(value));
        }
        data.push_back(row);
    }
    return data;
}

// Function to read labels (integer values) from a CSV file
std::vector<int> read_labels(const std::string& filename) {
    std::vector<int> labels;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        labels.push_back(std::stoi(line));
    }
    return labels;
}

// Function to save predictions to a CSV file
void save_predictions(const std::vector<int>& predictions,
                      const std::string& filename) {
    std::ofstream file(filename);
    for (const int pred : predictions) {
        file << pred << std::endl;
    }
}
