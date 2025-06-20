#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

namespace ml
{

  double mean_squared_error(const std::vector<double> &y_true, const std::vector<double> &y_pred);
  double accuracy(const std::vector<double> &y_true, const std::vector<double> &y_pred);
  double hinge_loss(const std::vector<double> &y_true, const std::vector<double> &y_pred);
  double inertia(const std::vector<std::vector<double>> &X, const std::vector<std::vector<double>> &centroids, const std::vector<int> &labels);

}
