/*
 *  Ellipse.h
 *
 *  Copyright (c) 2016 Tobias Wood.
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 */

#ifndef QI_ELLIPSE_H
#define QI_ELLIPSE_H

#include <memory>
#include <complex>
#include <array>
#include <vector>
#include <string>

#include "Eigen/Dense"

#include "QI/Macro.h"
#include "QI/Types.h"
#include "QI/Util.h"
#include "QI/Sequences/SteadyStateSequence.h"
#include "QI/Ellipse/EllipseFits.h"

namespace QI {

class EllipseAlgo : public QI::ApplyVectorXFVectorF::Algorithm {
public:
    const static size_t NumOutputs = 5;
    typedef const Eigen::Map<const Eigen::ArrayXcf, 0, Eigen::InnerStride<>> map_t;
protected:
    EllipseMethods m_method;
    bool m_phaseFirst = false, m_debug = false;
    std::shared_ptr<QI::SSFPEcho> m_sequence = nullptr;
    TOutput m_zero;
public:

    EllipseAlgo(EllipseMethods m, std::shared_ptr<QI::SSFPEcho> &seq, bool debug, bool phase) :
        m_method(m), m_sequence(seq), m_debug(debug), m_phaseFirst(phase)
    {
        m_zero = TOutput(m_sequence->flip().rows());
        m_zero.Fill(0.);
    }

    size_t numInputs() const override { return 1; }
    size_t numConsts() const override { return 0; }
    size_t numOutputs() const override { return NumOutputs; }
    size_t dataSize() const override { return m_sequence->size(); }
    size_t outputSize(const int i) const override { return m_sequence->flip().rows(); }
    void setReorderPhase(const bool p) { m_phaseFirst = p; }
    virtual std::vector<float> defaultConsts() const override {
        std::vector<float> def(1, 1.0f); // B1
        return def;
    }
    virtual const TOutput &zero(const size_t i) const override { return m_zero; }
    const std::vector<std::string> & names() const {
        static std::vector<std::string> _names = {"G", "a", "b", "f0", "phi_rf"};
        return _names;
    }
    virtual bool apply(const std::vector<TInput> &inputs, const std::vector<TConst> &consts,
                       std::vector<TOutput> &outputs, TConst &residual,
                       TInput &resids, TIters &its) const override
    {
        size_t phase_stride = m_sequence->flip().rows();
        size_t flip_stride = 1;
        if (m_phaseFirst)
            std::swap(phase_stride, flip_stride);
        for (int f = 0; f < m_sequence->flip().rows(); f++) {
            map_t vf(inputs[0].GetDataPointer() + f*flip_stride, m_sequence->phase_incs().rows(), Eigen::InnerStride<>(phase_stride));
            if (m_debug) {
                //std::cout << "Flip: " << m_sequence->flip() << " B1: " << B1 << " B1*flip: " << B1*m_sequence->flip() << std::endl;
            }
            Array5d tempOutputs;
            switch (m_method) {
                case EllipseMethods::Hyper: tempOutputs = HyperEllipse(vf, m_sequence->TR(), m_sequence->phase_incs()); break;
                case EllipseMethods::Direct: tempOutputs = DirectEllipse(vf, m_sequence->TR(), m_sequence->phase_incs(), m_debug); break;
            }
            for (int o = 0; o < NumOutputs; o++) {
                outputs[o][f] = tempOutputs[o];
            }
        }
        return true;
    }
};

} // End namespace QI

#endif // QI_ELLIPSE_H