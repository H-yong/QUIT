/*
 *  Hyper.h
 *
 *  Copyright (c) 2017 Tobias Wood.
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 */

#ifndef QI_ELLIPSE_HYPER_H
#define QI_ELLIPSE_HYPER_H

#include <Eigen/Dense>

namespace QI {

Eigen::Array<double, 5, 1> HyperEllipse(const Eigen::ArrayXcf &input, const double TR, const Eigen::ArrayXd &phi);

} // End namespace QI

#endif // QI_ELLIPSE_HYPER_H