//
// Created by Michael Ellis on 8/1/22.
//

#include "LinearRegression.h"

LinearRegression::LinearRegression(int n, int p) {
  m_n = n;
  m_p = p;
  m_coefficients = Eigen::VectorXd::Zero(m_p);
}

void LinearRegression::fit(const Eigen::VectorXd& y, const Eigen::MatrixXd& X) {

  m_coefficients = X.colPivHouseholderQr().solve(y);

}

Eigen::VectorXd LinearRegression::predict(const Eigen::MatrixXd &X_new) {
  return X_new * m_coefficients;
}
double LinearRegression::MSE(const Eigen::VectorXd &y, const Eigen::VectorXd &y_pred) {
  Eigen::VectorXd y_error = y - y_pred;
  double sse = y_error.transpose() * y_error;
  return sse / m_n;
}
