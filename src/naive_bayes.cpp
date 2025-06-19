#include "ml_framework/naive_bayes.h"
#include <map>
#include <cmath>

namespace ml
{
#ifndef M_PI
  constexpr double M_PI = 3.14159265358979323846;
#endif
  void NaiveBayes::fit(const Matrix &X, const std::vector<double> &y)
  {
    std::map<double, std::vector<std::vector<double>>> subsets;
    for (size_t i = 0; i < y.size(); ++i)
      subsets[y[i]].push_back(X[i]);
    int nclasses = subsets.size(), m = X[0].size();
    classes.clear();
    priors.clear();
    mean.clear();
    var.clear();
    classes.reserve(nclasses);
    mean.resize(nclasses);
    var.resize(nclasses);
    priors.reserve(nclasses);
    int idx = 0;
    for (auto &kv : subsets)
    {
      classes.push_back(kv.first);
      auto &S = kv.second;
      double prior = double(S.size()) / X.size();
      priors.push_back(prior);
      mean[idx].resize(m, 0.0);
      var[idx].resize(m, 0.0);
      for (int j = 0; j < m; ++j)
      {
        double sum = 0;
        for (auto &r : S)
          sum += r[j];
        mean[idx][j] = sum / S.size();
        double s2 = 0;
        for (auto &r : S)
          s2 += (r[j] - mean[idx][j]) * (r[j] - mean[idx][j]);
        var[idx][j] = s2 / S.size();
      }
      ++idx;
    }
  }
  std::vector<double> NaiveBayes::predict(const Matrix &X)
  {
    size_t n = X.size(), k = classes.size(), m = X[0].size();
    std::vector<double> out(n);
    for (size_t i = 0; i < n; ++i)
    {
      double bestlogp = -1e300, bestc = classes[0];
      for (size_t c = 0; c < k; ++c)
      {
        double logp = std::log(priors[c]);
        for (size_t j = 0; j < m; ++j)
        {
          double x = X[i][j];
          double p = 1.0 / std::sqrt(2 * M_PI * var[c][j]) * std::exp(-(x - mean[c][j]) * (x - mean[c][j]) / (2 * var[c][j]));
          logp += std::log(p + 1e-9);
        }
        if (logp > bestlogp)
        {
          bestlogp = logp;
          bestc = classes[c];
        }
      }
      out[i] = bestc;
    }
    return out;
  }
}
