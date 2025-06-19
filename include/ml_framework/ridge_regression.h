#pragma once
#include "linear_regression.h"

namespace ml
{
  class RidgeRegression : public LinearRegression
  {
  protected:  
    double alpha;

  public:
    RidgeRegression(double alpha = 1.0, double lr = 0.01, int iters = 1000);
    void fit(const Matrix &X, const std::vector<double> &y) override;
  };
}
