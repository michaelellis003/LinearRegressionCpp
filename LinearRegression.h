//
// Created by Michael Ellis on 8/1/22.
//

#ifndef LINEARREGRESSIONCPP__LINEARREGRESSION_H_
#define LINEARREGRESSIONCPP__LINEARREGRESSION_H_

#include <Eigen/Dense>
class LinearRegression {
 private:
  int m_n {0};                                                        // sample size
  int m_p {1};                                                        // number of features/predictors including intercept
  Eigen::VectorXd m_coefficients = Eigen::VectorXd::Zero(m_p);

 public:

  explicit LinearRegression(int n = 0, int p = 1);

  void fit(const Eigen::VectorXd& y, const Eigen::MatrixXd& X);

  Eigen::VectorXd predict(const Eigen::MatrixXd& X_new);

  double MSE(const Eigen::VectorXd &y, const Eigen::VectorXd& y_pred);

  int getSampleSize() { return m_n; };
  int getNumCoefficents() { return m_p; };
  Eigen::VectorXd getCoefficients() { return m_coefficients; };

};

#endif //LINEARREGRESSIONCPP__LINEARREGRESSION_H_
