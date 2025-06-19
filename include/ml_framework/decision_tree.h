#pragma once
#include "model.h"
#include <memory>

namespace ml
{
  struct TreeNode
  {
    bool leaf = false;
    int feature = 0;
    double threshold = 0, value = 0;
    std::unique_ptr<TreeNode> left, right;
  };
  class DecisionTree : public Model
  {
    std::unique_ptr<TreeNode> root;
    int max_depth, min_samples_split;
    std::unique_ptr<TreeNode> build(const Matrix &X, const std::vector<double> &y, int depth);
    double predict_one(const std::vector<double> &x, const TreeNode *node) const;

  public:
    DecisionTree(int max_depth = 5, int min_samples_split = 2);
    void fit(const Matrix &X, const std::vector<double> &y) override;
    std::vector<double> predict(const Matrix &X) override;
  };
}
