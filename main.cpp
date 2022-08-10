#include <iostream>
#include <Eigen/Dense>
#include <random>
#include "LinearRegression.h"

Eigen::MatrixXd simulateX(const int n, const int p, bool addIntercept = true)
{

  Eigen::MatrixXd X = Eigen::MatrixXd::Random(n,p);

  if (addIntercept) {
    // add column of ones to first column
    X.col(0).setOnes();
  }

  return X;

}

Eigen::VectorXd simulateY(const Eigen::MatrixXd& X, const Eigen::VectorXd& beta, const int n, const double stddev = 1)
{

  Eigen::VectorXd y = X * beta;

  // add noise from Gaussian distribution
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> d{0, stddev};

  for (int i = 0; i < n; ++i) {
    y[i] += d(gen);
  }

  return y;
}

int main()
{

  constexpr int n = 100;
  constexpr int p = 6;

  const Eigen::VectorXd syntheticBeta =  Eigen::VectorXd::Random(p);
  const Eigen::MatrixXd X = simulateX(n, p);
  const Eigen::VectorXd y = simulateY(X, syntheticBeta, n);

  LinearRegression mod = LinearRegression(n, p);
  mod.fit(y, X);
  Eigen::VectorXd y_new = mod.predict(X);
  double modMSE = mod.MSE(y, y_new);
  std::cout << syntheticBeta << std::endl;

  std::cout << "================================" << std::endl;

  std::cout << mod.getCoefficients() << std::endl;

  std::cout << "================================" << std::endl;

  std::cout << modMSE << std::endl;

  return 0;
}
