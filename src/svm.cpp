#include "ml_framework/svm.h"
#include "ml_framework/metrics.h"

namespace ml
{

  SVM::SVM(double lr_, double C_, int iters_) : lr(lr_), C(C_), iters(iters_), intercept(0) {}

  void SVM::fit(const Matrix &X, const std::vector<double> &y)
  {
    size_t n = X.size(), m = X[0].size();
    coef.assign(m, 0.0);
    intercept = 0.0;
    std::vector<double> Yn(n);
    for (size_t i = 0; i < n; ++i)
      Yn[i] = y[i] > 0.5 ? 1.0 : -1.0;

    for (int it = 0; it < iters; ++it)
    {
      for (size_t i = 0; i < n; ++i)
      {
        double dot = intercept;
        for (size_t j = 0; j < m; ++j)
          dot += coef[j] * X[i][j];
        if (Yn[i] * dot < 1)
        {
          for (size_t j = 0; j < m; ++j)
            coef[j] += lr * (Yn[i] * X[i][j] - 2.0 / iters * coef[j]);
          intercept += lr * Yn[i];
        }
        else
        {
          for (size_t j = 0; j < m; ++j)
            coef[j] += lr * (-2.0 / iters * coef[j]);
        }
      }
      if (progress_callback_ && (it % progress_interval_ == 0))
      {
        auto preds = predict(X);
        double hloss = hinge_loss(y, preds);
        progress_callback_(it, "HingeLoss", hloss);
      }
    }
  }

  std::vector<double> SVM::predict(const Matrix &X)
  {
    size_t n = X.size(), m = X[0].size();
    std::vector<double> out(n);
    for (size_t i = 0; i < n; ++i)
    {
      double score = intercept;
      for (size_t j = 0; j < m; ++j)
        score += coef[j] * X[i][j];
      out[i] = score;
    }
    return out;
  }

}
