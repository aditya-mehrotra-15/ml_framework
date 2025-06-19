#pragma once
#include "model.h"
#include <vector>

namespace ml
{
  class KNN : public Model
  {
    int k;
    Matrix Xtrain;
    std::vector<double> ytrain;

  public:
    KNN(int k = 3);
    void fit(const Matrix &X, const std::vector<double> &y) override;
    std::vector<double> predict(const Matrix &X) override;
  };
}
