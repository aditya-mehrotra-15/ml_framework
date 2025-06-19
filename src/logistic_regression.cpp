#include "ml_framework/logistic_regression.h"
#include <cmath>

namespace ml
{
  double LogisticRegression::sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }
  LogisticRegression::LogisticRegression(double lr_, int iters_) : intercept(0), lr(lr_), iters(iters_) {}
  void LogisticRegression::fit(const Matrix &X, const std::vector<double> &y)
  {
    size_t n = X.size(), m = X[0].size();
    coef.assign(m, 0.0);
    intercept = 0.0;
    for (int epoch = 0; epoch < iters; ++epoch)
    {
      std::vector<double> preds(n);
      for (size_t i = 0; i < n; ++i)
      {
        double linear = intercept;
        for (size_t j = 0; j < m; ++j)
          linear += coef[j] * X[i][j];
        preds[i] = sigmoid(linear);
      }
      double grad0 = 0;
      std::vector<double> gradw(m, 0.0);
      for (size_t i = 0; i < n; ++i)
      {
        double err = preds[i] - y[i];
        grad0 += err;
        for (size_t j = 0; j < m; ++j)
          gradw[j] += err * X[i][j];
      }
      for (size_t j = 0; j < m; ++j)
        coef[j] -= lr * gradw[j] / n;
      intercept -= lr * grad0 / n;
    }
  }
  std::vector<double> LogisticRegression::predict(const Matrix &X)
  {
    size_t n = X.size(), m = X[0].size();
    std::vector<double> out(n);
    for (size_t i = 0; i < n; ++i)
    {
      double linear = intercept;
      for (size_t j = 0; j < m; ++j)
        linear += coef[j] * X[i][j];
      out[i] = sigmoid(linear);
    }
    return out;
  }
}
