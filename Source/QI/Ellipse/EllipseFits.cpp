/*
 *  Ellipse.cpp
 *
 *  Copyright (c) 2016 Tobias Wood.
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 */

#include "QI/Ellipse/EllipseFits.h"
#include "QI/Ellipse/EllipseHelpers.h"
#include "ceres/ceres.h"

namespace QI {

typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

Eigen::MatrixXd HyperS(const Eigen::ArrayXd &x, const Eigen::ArrayXd &y) {
    Eigen::Matrix<double, Eigen::Dynamic, 6> D(x.rows(), 6);
    D.col(0) = x*x;
    D.col(1) = 2*x*y;
    D.col(2) = y*y;
    D.col(3) = 2*x;
    D.col(4) = 2*y;
    D.col(5).setConstant(1);
    return D.transpose() * D;
}

Eigen::MatrixXd FitzC() {
    Matrix6d C = Matrix6d::Zero();
    // Fitgibbon et al
    C(0,2) = -2; C(1,1) = 1; C(2,0) = -2;
    return C;
}

Eigen::MatrixXd HyperC(const Eigen::ArrayXd &x, const Eigen::ArrayXd &y) {
    Matrix6d C = Matrix6d::Zero();
    // Hyper Ellipse
    const double N = x.cols();
    const double xc = x.sum() / N;
    const double yc = y.sum() / N;
    const double sx = x.square().sum() / N;
    const double sy = y.square().sum() / N;
    const double xy = (x * y).sum() / N; 
    C << 6*sx, 6*xy, sx+sy, 6*xc, 2*yc, 1,
         6*xy, 4*(sx+sy), 6*xy, 4*yc, 4*xc, 0,
         sx + sy, 6*xy, 6*sy, 2*xc, 6*yc, 1,
         6*xc, 4*yc, 2*xc, 4, 0, 0,
         2*yc, 4*xc, 6*yc, 0, 4, 0,
         1, 0, 1, 0, 0, 0;
    return C;
}

Array5d HyperEllipse(const Eigen::ArrayXcf &input, const double TR, const Eigen::ArrayXd &phi) {
    Eigen::ArrayXcd data = input.cast<std::complex<double>>();
    const double scale = data.abs().maxCoeff();
    Eigen::ArrayXd x = data.real() / scale;
    Eigen::ArrayXd y = data.imag() / scale;
    
    Eigen::MatrixXd S = HyperS(x, y);
    Matrix6d C = HyperC(x, y);
    
    // Note S and C are swapped so we can use GES
    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix6d> solver(C, S);
    Vector6d Z;
    if (fabs(solver.eigenvalues()[5]) > fabs(solver.eigenvalues()[0]))
        Z = solver.eigenvectors().col(5);
    else
        Z = solver.eigenvectors().col(0);

    const double dsc=(Z[1]*Z[1]-Z[0]*Z[2]);
    const double xc = (Z[2]*Z[3]-Z[1]*Z[4])/dsc;
    const double yc = (Z[0]*Z[4]-Z[1]*Z[3])/dsc;
    const double theta_te = atan2(yc,xc);
    const double num = 2*(Z[0]*(Z[4]*Z[4])+Z[2]*(Z[3]*Z[3])+Z[5]*(Z[1]*Z[1])-2*Z[1]*Z[3]*Z[4]-Z[0]*Z[2]*Z[5]);
    double A = sqrt(num/(dsc*(sqrt((Z[0]-Z[2])*(Z[0]-Z[2]) + 4*Z[1]*Z[1])-(Z[0]+Z[2]))));
    double B = sqrt(num/(dsc*(-sqrt((Z[0]-Z[2])*(Z[0]-Z[2]) + 4*Z[1]*Z[1])-(Z[0]+Z[2]))));
    if (A > B) {
        std::swap(A, B);
    }
    double G, a, b;
    double c = sqrt(xc*xc+yc*yc);
    SemiaxesToHoff(A, B, c, G, a, b);

    /* Calculate theta_tr, i.e. drop RF phase, eddy currents etc. */
    /* First, center, rotate back to vertical and get 't' parameter */
    const Eigen::ArrayXcd vert = data / std::polar(scale, theta_te);
    const Eigen::ArrayXd ct = (vert.real() - c) / A;
    const Eigen::VectorXd rhs = (ct - b) / (b*ct - 1);
    Eigen::MatrixXd lhs(rhs.rows(), 2);
    lhs.col(0) = cos(phi);
    lhs.col(1) = sin(phi);
    const Eigen::VectorXd K = (lhs.transpose() * lhs).partialPivLu().solve(lhs.transpose() * rhs);
    const double theta_0 = atan2(K[1], K[0]);
    const double psi_0 = std::arg(std::polar(1.0, theta_te) / std::polar(1.0, theta_0/2));
    Array5d outputs;
    outputs << G * scale, a, b, theta_0  / (2*M_PI*TR), psi_0;
    return outputs;
}

struct EllipseCost {
public:
    const Eigen::ArrayXcd &data;
    const double TR;
    const Eigen::ArrayXd &phi;

    bool operator() (double const* const* p, double* resids) const {

        const double &G = p[0][0];
        const double &a = p[0][1];
        const double &b = p[0][2];
        const double &f0 = p[0][3];
        const double &psi0 = p[0][4];
        // Convert the SSFP Ellipse parameters into a magnetization
        const double theta0 = 2*M_PI*f0*TR;
        const Eigen::ArrayXd theta = theta0 - phi;
        const double psi = theta0/2 + psi0;
        Eigen::ArrayXcd et(theta.size());
        et.real() = cos(theta);
        et.imag() = sin(theta);
        const Eigen::ArrayXcd m = G*std::polar(1.0, psi)*(1 - a*et) /
                                  (1 - b*cos(theta));
        Eigen::Map<Eigen::ArrayXd> r(resids, data.size());
        r = (m - data).abs();
        /*std::cout << "*** COST ***" << std::endl;
        std::cout << "G " << G << " a " << a << " b " << b << " f0 " << f0 << " phi0 " << phi0 << std::endl;
        std::cout << "psi " << psi << " theta " << theta.transpose() << std::endl;
        std::cout << "m " << m.transpose() << std::endl;
        std::cout << "d " << data.transpose() << std::endl;
        std::cout << "r " << r.transpose() << std::endl;*/
        return true;
    }
};

Array5d DirectEllipse(const Eigen::ArrayXcf &indata, const double TR, const Eigen::ArrayXd &phi) {
    Eigen::ArrayXcd data = indata.cast<std::complex<double>>();
    const double scale = data.abs().maxCoeff();
    data /= scale;

    std::complex<double> c_mean = data.mean();

    auto *cost = new ceres::DynamicNumericDiffCostFunction<EllipseCost>(new EllipseCost{data, TR, phi});
    cost->AddParameterBlock(5);
    cost->SetNumResiduals(data.size());

    // Get as estimate of f0 and psi0
    // Assume first point is 180 phase increment and take the phase difference
    const double theta0_est = arg((data[0] / c_mean) - std::complex<double>(1.0, 0.0));
    const double psi0_est   = arg(c_mean / std::polar(1.0, theta0_est/2));

    Array5d p; p << abs(c_mean), 0.95, 0.75, theta0_est / (2.0 * M_PI * TR), psi0_est;
    // std::cout << "Start p: " << p.transpose() << std::endl;
    ceres::Problem problem;
    problem.AddResidualBlock(cost, NULL, p.data());
    const double not_zero = std::nextafter(0.0, 1.0);
    const double not_one  = std::nextafter(1.0, 0.0);
    problem.SetParameterLowerBound(p.data(), 0, not_zero); problem.SetParameterUpperBound(p.data(), 0, not_one);
    problem.SetParameterLowerBound(p.data(), 1, not_zero); problem.SetParameterUpperBound(p.data(), 1, not_one);
    problem.SetParameterLowerBound(p.data(), 2, not_zero); problem.SetParameterUpperBound(p.data(), 2, not_one);
    problem.SetParameterLowerBound(p.data(), 3, -1 / TR); problem.SetParameterUpperBound(p.data(), 3,  1 / TR);
    problem.SetParameterLowerBound(p.data(), 4, -M_PI); problem.SetParameterUpperBound(p.data(), 4,  M_PI);
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.max_num_iterations = 50;
    options.function_tolerance = 1e-5;
    options.gradient_tolerance = 1e-6;
    options.parameter_tolerance = 1e-4;
    options.logging_type = ceres::SILENT;
    ceres::Solve(options, &problem, &summary);
    if (!summary.IsSolutionUsable()) {
        std::cout << summary.FullReport() << std::endl;
    }
    // std::cout << "End p: " << p.transpose() << std::endl;
    p[0] *= scale;
    return p;
};

} // End namespace QI