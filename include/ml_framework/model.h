#pragma once
#include "matrix.h"
#include <vector>
#include <functional>
#include <string>

namespace ml
{

    // Callback type: (epoch, metric_name, metric_value)
    using ProgressCallback = std::function<void(int, const std::string &, double)>;

    struct Model
    {
        virtual void fit(const Matrix &X, const std::vector<double> &y) = 0;
        virtual std::vector<double> predict(const Matrix &X) = 0;
        virtual ~Model() = default;

        // Set a callback to report progress every 'interval' epochs/iterations
        void set_progress_callback(ProgressCallback callback, int interval = 100)
        {
            progress_callback_ = callback;
            progress_interval_ = interval;
        }

    protected:
        ProgressCallback progress_callback_ = nullptr;
        int progress_interval_ = 100;
    };

}
