#pragma once
#include "model.h"
#include <vector>

namespace ml
{
    class LinearRegression : public Model
    {
    protected:
        std::vector<double> coef;
        double intercept;

    public:
        LinearRegression();
        void fit(const Matrix &X, const std::vector<double> &y) override;
        std::vector<double> predict(const Matrix &X) override;
    };
}
