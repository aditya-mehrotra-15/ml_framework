#include "ml_framework/dbscan.h"
#include <queue>
#include <cmath>

namespace ml
{
  DBSCAN::DBSCAN(double eps_, int minpts_) : eps(eps_), minpts(minpts_) {}
  double DBSCAN::dist(const std::vector<double> &a, const std::vector<double> &b) const
  {
    double d = 0;
    for (size_t i = 0; i < a.size(); ++i)
      d += (a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(d);
  }
  std::vector<int> DBSCAN::fit_predict(const Matrix &X)
  {
    int n = X.size();
    std::vector<int> labels(n, -1);
    int clusterid = 0;
    for (int i = 0; i < n; ++i)
    {
      if (labels[i] != -1)
        continue;
      std::vector<int> neigh;
      for (int j = 0; j < n; ++j)
        if (dist(X[i], X[j]) < eps)
          neigh.push_back(j);
      if ((int)neigh.size() < minpts)
        continue;
      labels[i] = clusterid;
      std::queue<int> q;
      for (int j : neigh)
        q.push(j);
      while (!q.empty())
      {
        int u = q.front();
        q.pop();
        if (labels[u] == -1)
          labels[u] = clusterid;
        if (labels[u] != clusterid)
          continue;
        std::vector<int> neigh2;
        for (int v = 0; v < n; ++v)
          if (dist(X[u], X[v]) < eps)
            neigh2.push_back(v);
        if ((int)neigh2.size() >= minpts)
          for (int v : neigh2)
            if (labels[v] == -1)
              q.push(v);
      }
      ++clusterid;
    }
    return labels;
  }
}
