#include "ml_framework/dataset.h"
#include "ml_framework/kmeans.h"
#include "ml_framework/dbscan.h"
#include "ml_framework/scaler.h"
#include <iostream>
#include <vector>

int main()
{
    // data numerical features only no target col
    auto data = ml::Dataset::load_csv("clustering_data.csv", false);
    if (data.X.empty())
    {
        std::cerr << "Failed to load data\n";
        return 1;
    }

    // --- Always scale features ---
    std::vector<double> mean, stdev;
    ml::fit_transform_standardize(data.X, mean, stdev);

    // 1. KMeans Clustering
    int num_clusters = 3;
    ml::KMeans kmeans(num_clusters, 100);
    kmeans.fit(data.X);
    auto kmeans_labels = kmeans.predict(data.X);

    std::cout << "KMeans clustering results:\n";
    for (size_t i = 0; i < kmeans_labels.size(); ++i)
        std::cout << "Sample " << i << " assigned to cluster " << kmeans_labels[i] << "\n";
    std::cout << "Centroids:\n";
    const auto &centroids = kmeans.get_centroids();
    for (size_t c = 0; c < centroids.size(); ++c)
    {
        std::cout << "Cluster " << c << ": ";
        for (auto v : centroids[c])
            std::cout << v << " ";
        std::cout << "\n";
    }

    // 2. DBSCAN Clustering
    double eps = 0.5;
    int minpts = 5;
    ml::DBSCAN dbscan(eps, minpts);
    auto dbscan_labels = dbscan.fit_predict(data.X);

    std::cout << "\nDBSCAN clustering results:\n";
    for (size_t i = 0; i < dbscan_labels.size(); ++i)
    {
        if (dbscan_labels[i] == -1)
            std::cout << "Sample " << i << " is noise\n";
        else
            std::cout << "Sample " << i << " assigned to cluster " << dbscan_labels[i] << "\n";
    }

    return 0;
}
