#include "ml_framework/metrics.h"

namespace ml
{

    double mean_squared_error(const std::vector<double> &y_true, const std::vector<double> &y_pred)
    {
        double sum = 0;
        for (size_t i = 0; i < y_true.size(); ++i)
            sum += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
        return sum / y_true.size();
    }

    double accuracy(const std::vector<double> &y_true, const std::vector<double> &y_pred)
    {
        size_t correct = 0;
        for (size_t i = 0; i < y_true.size(); ++i)
            if ((y_pred[i] > 0.5 ? 1.0 : 0.0) == y_true[i])
                ++correct;
        return double(correct) / y_true.size();
    }

    double hinge_loss(const std::vector<double> &y_true, const std::vector<double> &y_pred)
    {
        double loss = 0.0;
        for (size_t i = 0; i < y_true.size(); ++i)
        {
            double y = y_true[i] > 0.5 ? 1.0 : -1.0;
            loss += std::max(0.0, 1 - y * y_pred[i]);
        }
        return loss / y_true.size();
    }

    double inertia(const std::vector<std::vector<double>> &X, const std::vector<std::vector<double>> &centroids, const std::vector<int> &labels)
    {
        double sum = 0.0;
        for (size_t i = 0; i < X.size(); ++i)
        {
            double d = 0.0;
            int c = labels[i];
            for (size_t j = 0; j < X[i].size(); ++j)
                d += (X[i][j] - centroids[c][j]) * (X[i][j] - centroids[c][j]);
            sum += d;
        }
        return sum;
    }

}
