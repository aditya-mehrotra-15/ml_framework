#pragma once
#include "matrix.h"
#include <vector>

namespace ml
{
    class KMeans
    {
        Matrix centroids;
        int k, max_iters;

    public:
        KMeans(int k = 3, int max_iters = 100);
        void fit(const Matrix &X);
        std::vector<int> predict(const Matrix &X) const;
        const Matrix &get_centroids() const { return centroids; }
    };
}
