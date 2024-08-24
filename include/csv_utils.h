#ifndef CSV_UTILS_H
#define CSV_UTILS_H

#include <string>
#include <vector>

std::vector<std::vector<float>> read_csv(const std::string& filename);
std::vector<int> read_labels(const std::string& filename);
void save_predictions(const std::vector<int>& predictions,
                      const std::string& filename);

#endif // CSV_UTILS_H
