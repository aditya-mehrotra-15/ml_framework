#pragma once
#include "matrix.h"
#include <vector>

namespace ml
{
  class DBSCAN
  {
    double eps;
    int minpts;
    double dist(const std::vector<double> &a, const std::vector<double> &b) const;

  public:
    DBSCAN(double eps = 0.5, int minpts = 5);
    std::vector<int> fit_predict(const Matrix &X);
  };
}
