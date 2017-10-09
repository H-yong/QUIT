/*
 *  Direct.h
 *
 *  Copyright (c) 2016, 2017 Tobias Wood.
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 */

#ifndef QI_ELLIPSE_DIRECT_H
#define QI_ELLIPSE_DIRECT_H

#include <Eigen/Dense>

namespace QI {

Eigen::Array<double, 5, 1> DirectEllipse(const Eigen::ArrayXcf &input, const double TR, const Eigen::ArrayXd &phi, const bool debug, float &residual);

} // End namespace QI

#endif // QI_ELLIPSE_DIRECT_H