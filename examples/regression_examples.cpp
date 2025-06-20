#include "ml_framework/dataset.h"
#include "ml_framework/linear_regression.h"
#include "ml_framework/polynomial_regression.h"
#include "ml_framework/ridge_regression.h"
#include "ml_framework/lasso_regression.h"
#include "ml_framework/metrics.h"
#include "ml_framework/scaler.h"
#include <iostream>
#include <vector>

int main()
{
    // regression dataset (numeric features, continuous target)
    // last col is target
    auto data = ml::Dataset::load_csv("regression_data1.csv", false);
    if (data.X.empty())
    {
        std::cerr << "Failed to load data\n";
        return 1;
    }

    // Train/test split (80/20)
    size_t ntrain = static_cast<size_t>(data.X.size() * 0.8);
    ml::Matrix Xtrain(data.X.begin(), data.X.begin() + ntrain);
    ml::Matrix Xtest(data.X.begin() + ntrain, data.X.end());
    std::vector<double> ytrain(data.y.begin(), data.y.begin() + ntrain);
    std::vector<double> ytest(data.y.begin() + ntrain, data.y.end());

    // Feature scaling (compulsary to prevent overflows)
    std::vector<double> mean, stdev;
    ml::fit_transform_standardize(Xtrain, mean, stdev);
    ml::transform_standardize(Xtest, mean, stdev);

    //How you want to display progress during training (model.h dekh)
    auto progress_callback = [](int epoch, const std::string &metric, double value)
    {
        std::cout << "Epoch " << epoch << ": " << metric << " = " << value << "\n";
    };

    // 1. Linear Regression
    ml::LinearRegression lr(
        0.005, // lr (default: 0.01)
        5000   // iters (default: 1000)
    );
    lr.set_progress_callback(progress_callback, 100); // to view progress during training (here, every 100 epochs)
    lr.fit(Xtrain, ytrain);
    auto lr_preds = lr.predict(Xtest);
    std::cout << "Linear Regression MSE: " << ml::mean_squared_error(ytest, lr_preds) << "\n";

    // 2. Polynomial Regressio
    ml::PolynomialRegression poly(
        2, // degree (default: 2)
        0.005,
        3000);
    poly.fit(Xtrain, ytrain);
    auto poly_preds = poly.predict(Xtest);
    std::cout << "Polynomial Regression MSE: " << ml::mean_squared_error(ytest, poly_preds) << "\n";

    // 3. Ridge Regression
    ml::RidgeRegression ridge(
        0.1, // alpha (L2 regularization strength)
        0.005,
        3000);
    ridge.fit(Xtrain, ytrain);
    auto ridge_preds = ridge.predict(Xtest);
    std::cout << "Ridge Regression MSE: " << ml::mean_squared_error(ytest, ridge_preds) << "\n";

    // 4. Lasso Regression
    ml::LassoRegression lasso(
        0.5, // alpha (L1 regularization strength)
        0.005,
        3000);
    lasso.fit(Xtrain, ytrain);
    auto lasso_preds = lasso.predict(Xtest);
    std::cout << "Lasso Regression MSE: " << ml::mean_squared_error(ytest, lasso_preds) << "\n";

    return 0;
}
