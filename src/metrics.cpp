#include "ml_framework/metrics.h"
#include <cmath>

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
}
