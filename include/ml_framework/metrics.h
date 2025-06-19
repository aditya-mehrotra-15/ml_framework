#pragma once
#include <vector>
#include "matrix.h"
namespace ml
{
  double mean_squared_error(const std::vector<double> &y_true, const std::vector<double> &y_pred);
  double accuracy(const std::vector<double> &y_true, const std::vector<double> &y_pred);
}
