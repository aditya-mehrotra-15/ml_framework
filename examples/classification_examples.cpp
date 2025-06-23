#include "ml_framework/dataset.h"
#include "ml_framework/logistic_regression.h"
#include "ml_framework/decision_tree.h"
#include "ml_framework/knn.h"
#include "ml_framework/svm.h"
#include "ml_framework/naive_bayes.h"
#include "ml_framework/metrics.h"
#include "ml_framework/scaler.h"
#include <iostream>
#include <cmath>
#include <vector>

int main() {
    // classification dataset (numeric features, categorical target)
    // last col is target
    auto data = ml::Dataset::load_csv("toy_binary_dataset.csv", false);
    if (data.X.empty()) {
        std::cerr << "Failed to load data\n";
        return 1;
    }

    // Split into 80% train, 20% test
    size_t ntrain = static_cast<size_t>(data.X.size() * 0.8);
    ml::Matrix Xtrain(data.X.begin(), data.X.begin() + ntrain);
    ml::Matrix Xtest(data.X.begin() + ntrain, data.X.end());
    std::vector<double> ytrain(data.y.begin(), data.y.begin() + ntrain);
    std::vector<double> ytest(data.y.begin() + ntrain, data.y.end());

    std::vector<double> mean, stdev;
    ml::fit_transform_standardize(Xtrain, mean, stdev);
    ml::transform_standardize(Xtest, mean, stdev);
    
    //see regression examples
    auto progress_callback = [](int epoch, const std::string &metric, double value)
    {
        std::cout << "Epoch " << epoch << ": " << metric << " = " << value << "\n";
    };

    // Logistic Regression
    ml::LogisticRegression logreg(
        0.05, //lr
        2000  //iters
    );
    logreg.set_progress_callback(progress_callback, 100);
    logreg.fit(Xtrain, ytrain);
    auto logreg_probs = logreg.predict(Xtest);
    std::cout << "Logistic Regression Accuracy: " << ml::accuracy(ytest, logreg_probs) << "\n";

    // Decision Tree
    ml::DecisionTree tree(
        3, // max depth
        3  //min samples to split
    );
    tree.fit(Xtrain, ytrain);
    auto tree_preds = tree.predict(Xtest);
    std::cout << "Decision Tree Accuracy: " << ml::accuracy(ytest, tree_preds) << "\n";

    // KNN
    ml::KNN knn(5); // k = 5
    knn.fit(Xtrain, ytrain);
    auto knn_preds = knn.predict(Xtest);
    std::cout << "KNN Accuracy: " << ml::accuracy(ytest, knn_preds) << "\n";

    // SVM
    ml::SVM svm(0.01, 1.0, 1000);
    svm.set_progress_callback(progress_callback, 100);
    svm.fit(Xtrain, ytrain);
    auto svm_preds = svm.predict(Xtest);
    std::cout << "SVM Accuracy: " << ml::accuracy(ytest, svm_preds) << "\n";

    // Naive Bayes
    ml::NaiveBayes nb;
    nb.fit(Xtrain, ytrain);
    auto nb_preds = nb.predict(Xtest);
    std::cout << "Naive Bayes Accuracy: " << ml::accuracy(ytest, nb_preds) << "\n";

    return 0;
}
