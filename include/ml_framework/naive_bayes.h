#pragma once
#include "model.h"
#include <vector>

namespace ml
{
  class NaiveBayes : public Model
  {
    std::vector<double> classes, priors;
    std::vector<std::vector<double>> mean, var;

  public:
    void fit(const Matrix &X, const std::vector<double> &y) override;
    std::vector<double> predict(const Matrix &X) override;
  };
}
