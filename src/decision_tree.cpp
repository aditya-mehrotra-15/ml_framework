#include "ml_framework/decision_tree.h"
#include <numeric>
#include <algorithm>
#include <limits>

namespace ml
{
  DecisionTree::DecisionTree(int md, int ms) : max_depth(md), min_samples_split(ms) {}
  void DecisionTree::fit(const Matrix &X, const std::vector<double> &y)
  {
    root = build(X, y, 0);
  }
  std::unique_ptr<TreeNode> DecisionTree::build(const Matrix &X, const std::vector<double> &y, int depth)
  {
    int n = X.size(), m = X[0].size();
    auto node = std::make_unique<TreeNode>();
    bool pure = std::all_of(y.begin(), y.end(), [&](double v)
                            { return v == y[0]; });
    if (depth >= max_depth || n < min_samples_split || pure)
    {
      node->leaf = true;
      node->value = std::accumulate(y.begin(), y.end(), 0.0) / n;
      return node;
    }
    double best_gain = 0, best_t = 0, parent_mse = 0;
    int best_f = 0;
    double mean = std::accumulate(y.begin(), y.end(), 0.0) / n;
    for (double v : y)
      parent_mse += (v - mean) * (v - mean);
    parent_mse /= n;
    for (int f = 0; f < m; ++f)
    {
      std::vector<double> vals(n);
      for (int i = 0; i < n; ++i)
        vals[i] = X[i][f];
      std::sort(vals.begin(), vals.end());
      for (int i = 1; i < n; ++i)
      {
        double t = (vals[i - 1] + vals[i]) / 2;
        std::vector<double> yl, yr;
        for (int j = 0; j < n; ++j)
          (X[j][f] < t ? yl : yr).push_back(y[j]);
        if (yl.empty() || yr.empty())
          continue;
        auto mse = [](const std::vector<double> &ys)
        {
          double mean = std::accumulate(ys.begin(), ys.end(), 0.0) / ys.size();
          double s = 0;
          for (double v : ys)
            s += (v - mean) * (v - mean);
          return s / ys.size();
        };
        double gain = parent_mse - yl.size() * mse(yl) / n - yr.size() * mse(yr) / n;
        if (gain > best_gain)
        {
          best_gain = gain;
          best_f = f;
          best_t = t;
        }
      }
    }
    if (best_gain == 0)
    {
      node->leaf = true;
      node->value = mean;
      return node;
    }
    node->leaf = false;
    node->feature = best_f;
    node->threshold = best_t;
    Matrix XL, XR;
    std::vector<double> yl, yr;
    for (int i = 0; i < n; ++i)
      if (X[i][best_f] < best_t)
      {
        XL.push_back(X[i]);
        yl.push_back(y[i]);
      }
      else
      {
        XR.push_back(X[i]);
        yr.push_back(y[i]);
      }
    node->left = build(XL, yl, depth + 1);
    node->right = build(XR, yr, depth + 1);
    return node;
  }
  double DecisionTree::predict_one(const std::vector<double> &x, const TreeNode *node) const
  {
    if (node->leaf)
      return node->value;
    if (x[node->feature] < node->threshold)
      return predict_one(x, node->left.get());
    return predict_one(x, node->right.get());
  }
  std::vector<double> DecisionTree::predict(const Matrix &X)
  {
    std::vector<double> out(X.size());
    for (size_t i = 0; i < X.size(); ++i)
      out[i] = predict_one(X[i], root.get());
    return out;
  }
}
