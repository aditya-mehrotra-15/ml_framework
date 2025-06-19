#include "ml_framework/dataset.h"
#include "ml_framework/linear_regression.h"
#include "ml_framework/polynomial_regression.h"
#include "ml_framework/ridge_regression.h"
#include "ml_framework/lasso_regression.h"
#include "ml_framework/metrics.h"
#include "ml_framework/scaler.h"
#include <iostream>
#include <cmath>
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

    
    std::vector<double> mean, stdev;
    ml::fit_transform_standardize(Xtrain, mean, stdev);  // Fit and scale train
    ml::transform_standardize(Xtest, mean, stdev);       // Scale test with train params
    

    // 1. Linear Regression 
    ml::LinearRegression lr;
    lr.fit(Xtrain, ytrain);
    auto lr_preds = lr.predict(Xtest);
    std::cout << "Linear Regression MSE: "
              << ml::mean_squared_error(ytest, lr_preds) << "\n";

    // 2. Polynomial Regression (degree=2)
    ml::PolynomialRegression poly(2);
    poly.fit(Xtrain, ytrain);
    auto poly_preds = poly.predict(Xtest);
    std::cout << "Polynomial Regression MSE: "
              << ml::mean_squared_error(ytest, poly_preds) << "\n";

    // 3. Ridge Regression L2
    ml::RidgeRegression ridge(1.0);
    ridge.fit(Xtrain, ytrain);
    auto ridge_preds = ridge.predict(Xtest);
    std::cout << "Ridge Regression MSE: "
              << ml::mean_squared_error(ytest, ridge_preds) << "\n";

    // 4. Lasso Regression L1
    ml::LassoRegression lasso(0.5);
    lasso.fit(Xtrain, ytrain);
    auto lasso_preds = lasso.predict(Xtest);
    std::cout << "Lasso Regression MSE: "
              << ml::mean_squared_error(ytest, lasso_preds) << "\n";

    return 0;
}
