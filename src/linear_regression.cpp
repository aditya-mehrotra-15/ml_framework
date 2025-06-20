#include "ml_framework/linear_regression.h"
#include "ml_framework/metrics.h"

namespace ml
{

    LinearRegression::LinearRegression(double lr_, int iters_) : intercept(0), lr(lr_), iters(iters_) {}

    void LinearRegression::fit(const Matrix &X, const std::vector<double> &y)
    {
        size_t n = X.size(), m = X[0].size();
        coef.assign(m, 0.0);
        intercept = 0.0;

        for (int epoch = 0; epoch < iters; ++epoch)
        {
            std::vector<double> preds = predict(X);
            for (size_t j = 0; j < m; ++j)
            {
                double grad = 0;
                for (size_t i = 0; i < n; ++i)
                    grad += (preds[i] - y[i]) * X[i][j];
                coef[j] -= lr * grad / n;
            }
            double grad0 = 0;
            for (size_t i = 0; i < n; ++i)
                grad0 += preds[i] - y[i];
            intercept -= lr * grad0 / n;

            if (progress_callback_ && (epoch % progress_interval_ == 0))
            {
                double mse = mean_squared_error(y, preds);
                progress_callback_(epoch, "MSE", mse);
            }
        }
    }

    std::vector<double> LinearRegression::predict(const Matrix &X)
    {
        std::vector<double> out(X.size(), intercept);
        for (size_t i = 0; i < X.size(); ++i)
            for (size_t j = 0; j < X[0].size(); ++j)
                out[i] += coef[j] * X[i][j];
        return out;
    }

}
