/*
 *  EllipseAlgo.cpp
 *
 *  Copyright (c) 2016, 2017 Tobias Wood.
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 */

#include "QI/Ellipse/EllipseAlgo.h"
/*#include "QI/Ellipse/EllipseHelpers.h"
#include "QI/Banding.h"
#include "QI/GoldenSection.h"*/

namespace QI {

EllipseAlgo::EllipseAlgo(EllipseMethods m, std::shared_ptr<QI::SSFPEcho> &seq, bool debug) :
    m_method(m), m_sequence(seq), m_debug(debug)
{
    m_zero = TOutput(m_sequence->flip().rows());
    m_zero.Fill(0.);
}

bool EllipseAlgo::apply(const std::vector<TInput> &inputs,
                        const std::vector<TConst> &consts,
                        std::vector<TOutput> &outputs, TConst &residual,
                        TInput &resids, TIters &its) const
{
    const int np = m_sequence->phase_incs().rows();
    for (int f = 0; f < m_sequence->flip().rows(); f++) {
        Eigen::ArrayXcf data(np);
        for (int i = 0; i < np; i++) {
            data[i] = inputs[0][f*np + i];
        }
        if (m_debug) {
            std::cout << "Flip: " << m_sequence->flip()[f] << " Data: " << data.transpose() << std::endl;
        }
        Eigen::Array<double, 5, 1> tempOutputs;
        switch (m_method) {
        case EllipseMethods::Hyper: tempOutputs = HyperEllipse(data, m_sequence->TR(), m_sequence->phase_incs()); break;
        case EllipseMethods::Direct: tempOutputs = DirectEllipse(data, m_sequence->TR(), m_sequence->phase_incs(), m_debug); break;
        }
        for (int o = 0; o < NumOutputs; o++) {
            outputs[o][f] = tempOutputs[o];
        }
    }
    return true;
}

} // End namespace QI