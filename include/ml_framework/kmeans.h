#pragma once
#include "matrix.h"
#include <vector>
#include <functional>
#include <string>

namespace ml
{
    using ProgressCallback = std::function<void(int, const std::string &, double)>;

    class KMeans
    {
        Matrix centroids;
        int k, max_iters;
        ProgressCallback progress_callback_ = nullptr;
        int progress_interval_ = 100;

    public:
        KMeans(int k = 3, int max_iters = 100);
        void fit(const Matrix &X);
        std::vector<int> predict(const Matrix &X) const;
        const Matrix &get_centroids() const { return centroids; }
        void set_progress_callback(ProgressCallback callback, int interval = 100)
        {
            progress_callback_ = callback;
            progress_interval_ = interval;
        }
    };
}
