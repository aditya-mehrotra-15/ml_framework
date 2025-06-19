#pragma once
#include "model.h"
#include <vector>

namespace ml
{
  class SVM : public Model
  {
    std::vector<double> coef;
    double intercept, lr, C;
    int iters;

  public:
    SVM(double lr = 0.01, double C = 1.0, int iters = 1000);
    void fit(const Matrix &X, const std::vector<double> &y) override;
    std::vector<double> predict(const Matrix &X) override;
  };
}
