#pragma once
#include "matrix.h"
#include <vector>
#include <cmath>

namespace ml
{

    // fit on training data
    inline void fit_transform_standardize(Matrix &X, std::vector<double> &mean, std::vector<double> &stdev)
    {
        if (X.empty())
            return;
        int m = X[0].size(), n = X.size();
        mean.assign(m, 0.0);
        stdev.assign(m, 0.0);
        // Compute mean
        for (int j = 0; j < m; ++j)
            for (int i = 0; i < n; ++i)
                mean[j] += X[i][j];
        for (int j = 0; j < m; ++j)
            mean[j] /= n;
        // Compute stddev
        for (int j = 0; j < m; ++j)
            for (int i = 0; i < n; ++i)
                stdev[j] += std::pow(X[i][j] - mean[j], 2);
        for (int j = 0; j < m; ++j)
            stdev[j] = std::sqrt(stdev[j] / n);
        // Scale
        for (auto &row : X)
            for (int j = 0; j < m; ++j)
                row[j] = (row[j] - mean[j]) / (stdev[j] ? stdev[j] : 1.0);
    }

    // scale test data withh train data params.
    inline void transform_standardize(Matrix &X, const std::vector<double> &mean, const std::vector<double> &stdev)
    {
        if (X.empty())
            return;
        int m = X[0].size();
        for (auto &row : X)
            for (int j = 0; j < m; ++j)
                row[j] = (row[j] - mean[j]) / (stdev[j] ? stdev[j] : 1.0);
    }

}
