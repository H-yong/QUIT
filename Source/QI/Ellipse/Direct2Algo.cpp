/*
 *  Direct2Algo.cpp
 *
 *  Copyright (c) 2017 Tobias Wood.
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 */

#include "QI/Ellipse/Direct2Algo.h"
#include "QI/Ellipse/EllipseHelpers.h"
#include "QI/Fit.h"
#include "ceres/ceres.h"

namespace QI {

struct Direct2Cost {
public:
    const Eigen::ArrayXcd &data;
    const double TR;
    const Eigen::ArrayXd &phi;
    const bool debug;

    template<typename T>
    bool operator() (T const* const* p, T* resids) const {
        typedef Eigen::Array<T, Eigen::Dynamic, 1> ArrayXT;

        const T &G = p[0][0];
        const T &a_1 = p[0][1];
        const T &b_1 = p[0][2];
        const T &f0_1 = p[0][3];

        const T &f_2 = p[0][4];
        const T &a_2 = p[0][5];
        const T &b_2 = p[0][6];
        const T &f0_2 = p[0][7] + f0_1; // Fit this as an offset
        const T &f_1 = 1.0 - f_2;
        const T &psi0 = p[0][8];
        
        // Convert the SSFP Ellipse parameters into a magnetization
        const ArrayXT m1 = EllipseToSignal(G * f_1, a_1, b_1, f0_1, psi0, TR, phi);
        const ArrayXT m2 = EllipseToSignal(G * f_2, a_2, b_2, f0_2, psi0, TR, phi);
        const ArrayXT m = m1 + m2;
        Eigen::Map<ArrayXT> r(resids, data.size()*2);
        r.head(data.size()) = m.head(data.size()) - data.real();
        r.tail(data.size()) = m.tail(data.size()) - data.imag();
        // if (debug) {
        //     std::cout << "*** DIRECT 2 COST ***\n"
        //               << "IE: f " << f_1 << " a " << a_1 << " b_1 " << b_1 << " f0 " << f0_1 << "\n"
        //               << "M:  f " << f_2 << " a " << a_2 << " b_1 " << b_2 << " f0 " << f0_2 << "\n"
        //               << "G: " << G << " psi0: " << psi0 << std::endl;
        // }
        return true;
    }
};

Eigen::ArrayXd Direct2Algo::apply_internal(const Eigen::ArrayXcf &indata, const double TR, const Eigen::ArrayXd &phi, const bool debug, float &residual) const {
    Eigen::ArrayXcd data = indata.cast<std::complex<double>>();
    const double scale = data.abs().maxCoeff();
    data /= scale;

    std::complex<double> c_mean = data.mean();
    // Get as estimate of f0 and psi0
    // Do a linear regression of phi against unwrapped phase diff, intercept is theta0
    Eigen::VectorXd Y = QI::Unwrap((data / c_mean - std::complex<double>(1.0, 0.0)).arg());
    Eigen::MatrixXd X(Y.rows(), 2);
    X.col(0) = phi - M_PI;
    X.col(1).setOnes();
    const Eigen::VectorXd b = QI::LeastSquares(X, Y);
    const double theta0_est = arg((data[0] / c_mean) - std::complex<double>(1.0, 0.0));//b[1];
    const double psi0_est   = arg(c_mean / std::polar(1.0, theta0_est/2));
    if (debug) {
        std::cerr << "X\n" << X.transpose() << "\nY\n" << Y.transpose() << std::endl;
        std::cerr << "b " << b.transpose() << " theta0_est " << theta0_est << " psi0_est " << psi0_est << std::endl;
        std::cerr << "arg(c_mean) " << std::arg(c_mean) << std::endl;
        std::cerr << "old theta0_est " << arg((data[0] / c_mean) - std::complex<double>(1.0, 0.0)) << std::endl;
    }
    
    const double f0_1_est = theta0_est / (2.0 * M_PI * TR);
    Eigen::Array<double, 9, 1> p;
    p << abs(c_mean), 0.9, 0.5, f0_1_est, 0.1, 0.7, 0.8, -10., psi0_est;
    ceres::Problem problem;
    auto *cost = new ceres::DynamicAutoDiffCostFunction<Direct2Cost>(new Direct2Cost{data, TR, phi, debug});
    cost->AddParameterBlock(9);
    cost->SetNumResiduals(data.size()*2);
    problem.AddResidualBlock(cost, NULL, p.data());
    const double not_zero = 1.0e-6;
    const double not_one  = 1.0 - not_zero;
    const double max_a = exp(-TR / 4.3); // Set a sensible maximum on T2
    problem.SetParameterLowerBound(p.data(), 0, not_zero); problem.SetParameterUpperBound(p.data(), 0, not_one);
    problem.SetParameterLowerBound(p.data(), 1, 0.85); problem.SetParameterUpperBound(p.data(), 1, max_a);
    problem.SetParameterLowerBound(p.data(), 2, 0.4); problem.SetParameterUpperBound(p.data(), 2, 0.6);
    problem.SetParameterLowerBound(p.data(), 3, -1/TR + 1.0e-6); problem.SetParameterUpperBound(p.data(), 3,  1/TR - 1.0e-6);
    problem.SetParameterLowerBound(p.data(), 4, not_zero); problem.SetParameterUpperBound(p.data(), 4, 0.2);
    problem.SetParameterLowerBound(p.data(), 5, 0.5); problem.SetParameterUpperBound(p.data(), 5, 0.85);
    problem.SetParameterLowerBound(p.data(), 6, 0.65); problem.SetParameterUpperBound(p.data(), 6, 0.9);
    problem.SetParameterLowerBound(p.data(), 7, -50.); problem.SetParameterUpperBound(p.data(), 7,  50.);
    problem.SetParameterLowerBound(p.data(), 8, -M_PI); problem.SetParameterUpperBound(p.data(), 8,  M_PI);
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.max_num_iterations = 50;
    options.function_tolerance = 1e-5;
    options.gradient_tolerance = 1e-6;
    options.parameter_tolerance = 1e-4;
    if (!debug) options.logging_type = ceres::SILENT;
    if (debug) {
        std::cout << "STARTING P: " << p.transpose() << std::endl;
    }
    ceres::Solve(options, &problem, &summary);
    if (debug || !summary.IsSolutionUsable()) {
        //std::cout << summary.FullReport() << std::endl;
        std::cout << "FINAL P: " << p.transpose() << std::endl;
    }
    residual = summary.final_cost;
    p[0] *= scale;
    return p;
};

} // End namespace QI