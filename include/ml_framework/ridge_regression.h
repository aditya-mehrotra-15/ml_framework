#pragma once
#include "linear_regression.h"
namespace ml

{
  class RidgeRegression : public LinearRegression
  {
    double alpha;

  public:
    RidgeRegression(double alpha = 1.0);
    void fit(const Matrix &X, const std::vector<double> &y) override;
  };
}
