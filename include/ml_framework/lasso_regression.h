#pragma once
#include "ridge_regression.h"

namespace ml {
class LassoRegression : public RidgeRegression {
public:
    LassoRegression(double alpha = 1.0);
    void fit(const Matrix& X, const std::vector<double>& y) override;
};
}
