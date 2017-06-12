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

#include "QI/Ellipse/Ellipse.h"
#include "QI/Ellipse/EllipseHelpers.h"
#include "QI/Banding.h"
#include "QI/GoldenSection.h"

namespace QI {

Eigen::MatrixXd HyperEllipse::buildS(const Eigen::ArrayXd &x, const Eigen::ArrayXd &y) const {
    Eigen::Matrix<double, Eigen::Dynamic, 6> D(x.rows(), 6);
    D.col(0) = x*x;
    D.col(1) = 2*x*y;
    D.col(2) = y*y;
    D.col(3) = 2*x;
    D.col(4) = 2*y;
    D.col(5).setConstant(1);
    return D.transpose() * D;
}

Eigen::MatrixXd HyperEllipse::fitzC() const {
    Matrix6d C = Matrix6d::Zero();
    // Fitgibbon et al
    C(0,2) = -2; C(1,1) = 1; C(2,0) = -2;
    return C;
}

Eigen::MatrixXd HyperEllipse::hyperC(const Eigen::ArrayXd &x, const Eigen::ArrayXd &y) const {
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

std::array<float, 7> HyperEllipse::applyFlip(const Eigen::Map<const Eigen::ArrayXcf, 0, Eigen::InnerStride<>> &indata,
                                             const double TR, const double flip) const {
    Eigen::ArrayXcd data = indata.cast<std::complex<double>>();
    const double scale = data.abs().maxCoeff();
    Eigen::ArrayXd x = data.real() / scale;
    Eigen::ArrayXd y = data.imag() / scale;
    
    Eigen::MatrixXd S = buildS(x, y);
    Matrix6d C = hyperC(x, y);
    
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
    double a, b;
    double c = sqrt(xc*xc+yc*yc);
    SemiaxesToHoff(A, B, c, a, b);

    /* Calculate theta_tr, i.e. drop RF phase, eddy currents etc. */
    /* First, center, rotate back to vertical and get 't' parameter */
    const Eigen::ArrayXcd vert = data / std::polar(1., theta_te);
    const Eigen::ArrayXd ct = (vert.real() - c) / A;
    const Eigen::VectorXd rhs = (ct - b) / (b*ct - 1);
    Eigen::MatrixXd lhs(rhs.rows(), 2);
    lhs.col(0) = cos(m_sequence->allPhi());
    lhs.col(1) = sin(m_sequence->allPhi());
    const Eigen::VectorXd K = (lhs.transpose() * lhs).partialPivLu().solve(lhs.transpose() * rhs);
    const double theta_0 = atan2(K[1], K[0]);
    std::array<float, NumOutputs> outputs;
    EllipseToMRI(a, b, c*scale, theta_0, TR, flip,
                 outputs[0], outputs[1], outputs[2], outputs[3], m_debug);
    outputs[4] = a;
    outputs[5] = b;
    outputs[6] = std::arg(std::polar(1.,theta_te) / std::polar(1.,theta_0/2));
    return outputs;
}

void ConstrainedEllipse::fit(const Eigen::ArrayXd &x, const Eigen::ArrayXd &y, const Eigen::Vector2d p, const Eigen::Vector2d q, 
             Eigen::Vector2d gammaBound, Eigen::Matrix2d &A, Eigen::Vector2d &x_c, double &g) const {
    const double epsFmin = 1e-8, epsRootsImag = 1e-6, epsRootsMin = 1e-3; // Threshold Values
    Eigen::Matrix3d B; B << 0,  0, 2,
                            0, -1, 0,
                            2,  0, 0;                                     // Constraint Matrix
    Eigen::PartialPivLU<Eigen::Matrix3d> BLU(B);
    const size_t n = x.rows();
    Eigen::MatrixXd Z = Eigen::MatrixXd::Identity(n,n) - Eigen::MatrixXd::Ones(n,n)/n; // Centering Matrix
    Eigen::MatrixXd D0(n, 3); D0.col(0) = (x-p[0]).square(); D0.col(1) = (x-p[0])*(y-p[1]); D0.col(2) = (y-p[1]).square();
    Eigen::MatrixXd D1(n, 3); D1.col(0) = 2*q[0]*(x-p[0]); D1.col(1) = q[0]*(y-p[1])+q[1]*(x-p[0]); D1.col(2) = 2*q[1]*(y-p[1]);
    Eigen::Matrix3d C0 = D0.transpose()*Z*D0;
    Eigen::Matrix3d C1 = -D0.transpose()*Z*D1 - D1.transpose()*Z*D0;
    Eigen::Matrix3d C2 = D1.transpose()*Z*D1;
    Eigen::Matrix3d BC0 = BLU.solve(C0);
    Eigen::Matrix3d BC1 = BLU.solve(C1);
    Eigen::Matrix3d BC2 = BLU.solve(C2);

    std::function<double(double)> getLambdaOfGamma = [&] (double gamma) {
        Eigen::Matrix3d M = BC0 + gamma*BC1 + (gamma*gamma)*BC2;
        Eigen::EigenSolver<Eigen::Matrix3d> eig(M);
        double lambda = fabs(eig.eigenvalues().real().maxCoeff());
        return lambda;
    };

    bool isGlobalMinimumFound = false;
    int its = 0;
    while (!isGlobalMinimumFound) {
        // GetLamdaOfGamma
        double gamma = GoldenSectionSearch(getLambdaOfGamma, gammaBound[0], gammaBound[1], epsFmin);
        double lambda = getLambdaOfGamma(gamma);
        //cout << "gamma " << gamma << " lambda " << lambda << endl;
        Eigen::Matrix3d P; P << 1, 0, q[1]*q[1],
                                0, 1, -2*q[0]*q[1],
                                0, 0, q[0]*q[0];
        Eigen::Matrix3d C2t = P.transpose()*C2*P;
        Eigen::Matrix3d C1t = P.transpose()*C1*P;
        Eigen::Matrix3d C0t = P.transpose()*C0*P;
        Eigen::Matrix3d Bt = P.transpose()*B*P;
        C0t = C0t - lambda*Bt;
        Eigen::Matrix2d C2b = C2t.topLeftCorner<2,2>()-C1t.block<2,1>(0,2)*C1t.block<1,2>(2,0)/C0t.coeff(2,2);
        Eigen::Matrix2d C1b = C1t.topLeftCorner<2,2>()-C1t.block<2,1>(0,2)*C0t.block<1,2>(2,0)/C0t.coeff(2,2) - C0t.block<2,1>(0,2)*C1t.block<1,2>(2,0)/C0t.coeff(2,2);
        Eigen::Matrix2d C0b = C0t.topLeftCorner<2,2>()-C0t.block<2,1>(0,2)*C0t.block<1,2>(2,0)/C0t.coeff(2,2);
        Eigen::Matrix4d rootsA; rootsA << Eigen::Matrix2d::Zero(), Eigen::Matrix2d::Identity(),
                                            C0b, C1b;
        Eigen::Matrix4d rootsB; rootsB << Eigen::Matrix2d::Identity(), Eigen::Matrix2d::Zero(),
                                            Eigen::Matrix2d::Zero(), -C2b;
        Eigen::GeneralizedEigenSolver<Eigen::Matrix4d> roots(rootsA, rootsB);
        Eigen::Vector4cd rootsVec = roots.eigenvalues();
        //cout << "rootsVec " << rootsVec.transpose() << endl;
        auto realRootsIdx = (rootsVec.imag().array().abs() < epsRootsImag) && ((rootsVec.array() - gamma).abs() > epsRootsMin);
        //cout << "realRootsIdx " << realRootsIdx.transpose() << endl;
        std::vector<double> realRoots; for (int i = 0; i < 4; i++) { if (realRootsIdx[i]) realRoots.push_back(real(rootsVec[i])); }
        std::sort(realRoots.begin(), realRoots.end());
        // cout << realRoots.size() << " : "; for (int i = 0; i < realRoots.size(); i++) { cout << realRoots[i] << " "; } cout << endl;
        if (realRoots.size() == 0) {
            isGlobalMinimumFound = true;
        } else if (realRoots.size() == 1) {
            // 1 real root. Global minimum is outside the given search range
            if (realRoots[0] < gammaBound[0]) {
                gammaBound[1] = gammaBound[0]; gammaBound[0] = realRoots[0];
            } else {
                gammaBound[0] = gammaBound[1]; gammaBound[1] = realRoots[0];
            }
        } else if (realRoots.size() == 2) {
            if (((realRoots[0] < gammaBound[0]) && (realRoots[1] > gammaBound[1])) || // 2 roots outside bounds
                ((realRoots[0] > gammaBound[0]) && (realRoots[1] < gammaBound[1]))) { // 2 roots inside bounds
                gammaBound[0] = realRoots[0]; gammaBound[1] = realRoots[1];
            } else if (realRoots[0] < gammaBound[0]) { // 1 root below bounds
                gammaBound[1] = realRoots[1];
            } else {
                gammaBound[0] = realRoots[0];
            }
            if (gamma > gammaBound[0] && gamma < gammaBound[1]) {
                // This shouldn't happen, but might due to noise. In which case claim it is the global minimum
                isGlobalMinimumFound = true;
            }
        } else { // 3 roots
            bool allOutside = true;
            std::vector<float> newBounds;
            for (int i = 0; i < 3; i++) {
                if (realRoots[i] > gammaBound[0] && realRoots[i] < gammaBound[1]) {
                    allOutside = false;
                    newBounds.push_back(realRoots[i]);
                }
            }
            if (allOutside) {
                isGlobalMinimumFound = true;
            } else {
                gammaBound[0] = std::min(newBounds[0], newBounds[1]);
                gammaBound[1] = std::max(newBounds[0], newBounds[1]);
            }
        }
        if (gammaBound[0] == gammaBound[1])
            isGlobalMinimumFound = true;
        //cout << "new gammaBound " << gammaBound.transpose() << endl;
        Eigen::Matrix3d M = BC0 + gamma*BC1 + gamma*gamma*BC2;
        Eigen::EigenSolver<Eigen::Matrix3d> eig(M);
        int eigIdx; eig.eigenvalues().real().maxCoeff(&eigIdx);
        //cout << "M eigenvalues " << eig.eigenvalues().transpose() << " idx " << eigIdx << endl;
        Eigen::Vector3d theta = eig.eigenvectors().col(eigIdx).real();
        // First part is a hacked sign function
        //cout << "theta " << theta.transpose() << endl;
        theta = ((theta[0] > 0) - (theta[0] < 0))*theta/sqrt(theta.transpose()*B*theta); // Normalization
        //cout << "theta n" << theta.transpose() << endl;
        double h_opt = (-1.0/n * Eigen::VectorXd::Ones(n).transpose()) * (D0 - gamma*D1)*theta;
        A << theta[0], theta[1]/2,
                theta[1]/2, theta[2];
        x_c = p + gamma*q;
        g = h_opt - gamma*gamma*q.transpose()*A*q;
        its++;
        if (its > 10)
            break;
    }
}

std::array<float, 7> ConstrainedEllipse::applyFlip(const Eigen::Map<const Eigen::ArrayXcf, 0, Eigen::InnerStride<>> &indata,
                                            const double TR, const double flip) const {
    Eigen::ArrayXcd data = indata.cast<std::complex<double>>();
    const double scale = data.abs().maxCoeff();
    data /= scale;
    Eigen::ArrayXcd a(data.rows() / 2);
    Eigen::ArrayXcd b(data.rows() / 2);
    QI::SplitBlocks(data, a, b, m_reorderBlock);
    const std::complex<double> gs = QI::GeometricSolution(a, b, QI::RegEnum::Line);
    Eigen::ArrayXd x = data.real();
    Eigen::ArrayXd y = data.imag();
    Eigen::Vector2d p = Eigen::Vector2d::Zero();
    Eigen::Vector2d q; q << real(gs), imag(gs);
    Eigen::Vector2d gammaBound; gammaBound << 0.25, 1.5;
    Eigen::Matrix2d A; Eigen::Vector2d xc; double g;
    fit(x, y, p, q, gammaBound, A, xc, g);
    Vector6d Z;
    {
        double a = A.coeffRef(0,0);
        double b = A.coeffRef(1,0)+A.coeffRef(0,1);
        double c = A.coeffRef(1,1);
        Z << a, b/2, c, -(2*a*xc[0] + b*xc[1])/2, -(2*c*xc[1] + b*xc[0])/2,
                a*xc[0]*xc[0] + b*xc[0]*xc[1] + c*xc[1]*xc[1] + g;
    }
    const double dsc=(Z[1]*Z[1]-Z[0]*Z[2]);
    const double th = atan2(xc[1],xc[0]);
    const double num = 2*(Z[0]*(Z[4]*Z[4])+Z[2]*(Z[3]*Z[3])+Z[5]*(Z[1]*Z[1])-2*Z[1]*Z[3]*Z[4]-Z[0]*Z[2]*Z[5]);
    double sA = sqrt(num/(dsc*(sqrt((Z[0]-Z[2])*(Z[0]-Z[2]) + 4*Z[1]*Z[1])-(Z[0]+Z[2]))));
    double sB = sqrt(num/(dsc*(-sqrt((Z[0]-Z[2])*(Z[0]-Z[2]) + 4*Z[1]*Z[1])-(Z[0]+Z[2]))));
    if (sA > sB) {
        std::swap(sA, sB);
    }
    double ea, eb;
    double c = sqrt(xc[0]*xc[0]+xc[1]*xc[1]);
    SemiaxesToHoff(sA, sB, c, ea, eb);
    std::array<float, NumOutputs> outputs;
    EllipseToMRI(ea, eb, c*scale, th, TR, flip,
                 outputs[0], outputs[1], outputs[2], outputs[3], m_debug);
    outputs[4] = ea;
    outputs[5] = eb;
    return outputs;
}

} // End namespace QI