#include "ml_framework/lasso_regression.h"
#include "ml_framework/metrics.h"
#include <cmath>

namespace ml
{
  LassoRegression::LassoRegression(double alpha, double lr, int iters)
      : RidgeRegression(alpha, lr, iters) {}

  void LassoRegression::fit(const Matrix &X, const std::vector<double> &y)
  {
    size_t n = X.size(), m = X[0].size();
    coef.assign(m, 0.0);
    intercept = 0.0;
    double alpha = this->alpha;
    for (int epoch = 0; epoch < this->iters; ++epoch)
    {
      std::vector<double> preds = predict(X);
      for (size_t j = 0; j < m; ++j)
      {
        double grad = 0;
        for (size_t i = 0; i < n; ++i)
          grad += (preds[i] - y[i]) * X[i][j];
        grad = grad / n + alpha * (coef[j] > 0 ? 1 : -1);
        coef[j] -= this->lr * grad;
      }
      double grad0 = 0;
      for (size_t i = 0; i < n; ++i)
        grad0 += preds[i] - y[i];
      intercept -= this->lr * grad0 / n;

      if (progress_callback_ && (epoch % progress_interval_ == 0))
      {
        double mse = mean_squared_error(y, preds);
        progress_callback_(epoch, "MSE", mse);
      }
    }
  }
}
