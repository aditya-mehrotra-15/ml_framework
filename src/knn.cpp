#include "ml_framework/knn.h"
#include <algorithm>
#include <cmath>

namespace ml
{
  KNN::KNN(int k_) : k(k_) {}
  void KNN::fit(const Matrix &X, const std::vector<double> &y)
  {
    Xtrain = X;
    ytrain = y;
  }
  std::vector<double> KNN::predict(const Matrix &X)
  {
    size_t n = X.size();
    std::vector<double> out(n);
    for (size_t i = 0; i < n; ++i)
    {
      std::vector<std::pair<double, double>> dist;
      for (size_t j = 0; j < Xtrain.size(); ++j)
      {
        double d = 0;
        for (size_t f = 0; f < X[0].size(); ++f)
          d += (X[i][f] - Xtrain[j][f]) * (X[i][f] - Xtrain[j][f]);
        dist.emplace_back(std::sqrt(d), ytrain[j]);
      }
      std::nth_element(dist.begin(), dist.begin() + k, dist.end());
      double sum = 0;
      for (int t = 0; t < k; ++t)
        sum += dist[t].second;
      out[i] = sum / k;
    }
    return out;
  }
}
