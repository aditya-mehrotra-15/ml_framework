#include "ml_framework/lasso_regression.h"
#include <cmath>

namespace ml
{
  LassoRegression::LassoRegression(double alpha) : RidgeRegression(alpha) {}
  void LassoRegression::fit(const Matrix &X, const std::vector<double> &y)
  {
    size_t n = X.size(), m = X[0].size();
    coef.assign(m, 0.0);
    intercept = 0.0;
    double lr = 0.01, alpha = 1.0;
    for (int it = 0; it < 1000; ++it)
    {
      std::vector<double> preds = predict(X);
      for (size_t j = 0; j < m; ++j)
      {
        double grad = 0;
        for (size_t i = 0; i < n; ++i)
          grad += (preds[i] - y[i]) * X[i][j];
        grad = grad / n + alpha * (coef[j] > 0 ? 1 : -1);
        coef[j] -= lr * grad;
      }
      double grad0 = 0;
      for (size_t i = 0; i < n; ++i)
        grad0 += preds[i] - y[i];
      intercept -= lr * grad0 / n;
    }
  }
}
