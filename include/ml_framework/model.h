#pragma once
#include "matrix.h"
#include <vector>
namespace ml
{
    struct Model
    {
        virtual void fit(const Matrix &X, const std::vector<double> &y) = 0;
        virtual std::vector<double> predict(const Matrix &X) = 0;
        virtual ~Model() = default;
    };
}
