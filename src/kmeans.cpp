#include "ml_framework/kmeans.h"
#include <random>
#include <limits>
#include <cmath>

namespace ml
{
  KMeans::KMeans(int k_, int max_iters_) : k(k_), max_iters(max_iters_) {}
  void KMeans::fit(const Matrix &X)
  {
    int n = X.size(), m = X[0].size();
    centroids.assign(k, std::vector<double>(m));
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> dist(0, n - 1);
    for (int i = 0; i < k; ++i)
      centroids[i] = X[dist(rng)];
    std::vector<int> labels(n);
    for (int it = 0; it < max_iters; ++it)
    {
      // assign
      for (int i = 0; i < n; ++i)
      {
        double bestd = std::numeric_limits<double>::max();
        int bestc = 0;
        for (int c = 0; c < k; ++c)
        {
          double d = 0;
          for (int j = 0; j < m; ++j)
            d += (X[i][j] - centroids[c][j]) * (X[i][j] - centroids[c][j]);
          if (d < bestd)
          {
            bestd = d;
            bestc = c;
          }
        }
        labels[i] = bestc;
      }
      // update
      Matrix newc(k, std::vector<double>(m, 0.0));
      std::vector<int> counts(k, 0);
      for (int i = 0; i < n; ++i)
      {
        ++counts[labels[i]];
        for (int j = 0; j < m; ++j)
          newc[labels[i]][j] += X[i][j];
      }
      for (int c = 0; c < k; ++c)
        if (counts[c])
          for (int j = 0; j < m; ++j)
            centroids[c][j] = newc[c][j] / counts[c];
    }
  }
  std::vector<int> KMeans::predict(const Matrix &X) const
  {
    int n = X.size(), m = X[0].size();
    std::vector<int> labels(n);
    for (int i = 0; i < n; ++i)
    {
      double bestd = std::numeric_limits<double>::max();
      int bestc = 0;
      for (int c = 0; c < k; ++c)
      {
        double d = 0;
        for (int j = 0; j < m; ++j)
          d += (X[i][j] - centroids[c][j]) * (X[i][j] - centroids[c][j]);
        if (d < bestd)
        {
          bestd = d;
          bestc = c;
        }
      }
      labels[i] = bestc;
    }
    return labels;
  }
}
