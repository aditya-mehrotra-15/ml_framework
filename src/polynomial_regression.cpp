#include "ml_framework/polynomial_regression.h"
#include <cmath>

namespace ml
{
  PolynomialRegression::PolynomialRegression(int deg) : degree(deg) {}

  Matrix PolynomialRegression::transform(const Matrix &X)
  {
    Matrix Xp;
    for (const auto &row : X)
    {
      std::vector<double> r = {1.0};
      for (int d = 1; d <= degree; ++d)
        for (double v : row)
          r.push_back(std::pow(v, d));
      Xp.push_back(r);
    }
    return Xp;
  }

  void PolynomialRegression::fit(const Matrix &X, const std::vector<double> &y)
  {
    LinearRegression::fit(transform(X), y);
  }

  std::vector<double> PolynomialRegression::predict(const Matrix &X)
  {
    return LinearRegression::predict(transform(X));
  }
}
