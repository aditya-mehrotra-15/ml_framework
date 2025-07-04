#pragma once
#include "linear_regression.h"

namespace ml
{
  class PolynomialRegression : public LinearRegression
  {
    int degree;
    Matrix transform(const Matrix &X);

  public:
    PolynomialRegression(int degree = 2, double lr = 0.01, int iters = 1000);
    void fit(const Matrix &X, const std::vector<double> &y) override;
    std::vector<double> predict(const Matrix &X) override;
  };
}
