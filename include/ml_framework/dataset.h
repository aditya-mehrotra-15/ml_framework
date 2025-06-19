#pragma once
#include "matrix.h"
#include <vector>
#include <string>

namespace ml
{
  struct Dataset
  {
    Matrix X;
    std::vector<double> y;
    static Dataset load_csv(const std::string &filepath, bool header = false);
  };
}
