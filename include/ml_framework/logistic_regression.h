#pragma once
#include "model.h"
#include <vector>

namespace ml
{
    class LogisticRegression : public Model
    {
        std::vector<double> coef;
        double intercept, lr;
        int iters;
        static double sigmoid(double z);

    public:
        LogisticRegression(double lr = 0.1, int iters = 1000);
        void fit(const Matrix &X, const std::vector<double> &y) override;
        std::vector<double> predict(const Matrix &X) override;
    };
}
