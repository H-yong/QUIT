/*
 *  EllipseAlgo.h
 *
 *  Copyright (c) 2016, 2017 Tobias Wood.
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 */

#ifndef QI_ELLIPSE_ALGO_H
#define QI_ELLIPSE_ALGO_H

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
#include "Filters/ApplyAlgorithmFilter.h"
#include "QI/Ellipse/Direct.h"
#include "QI/Ellipse/Hyper.h"

namespace QI {

enum class EllipseMethods { Hyper, Direct };

class EllipseAlgo : public QI::ApplyVectorXFVectorF::Algorithm {
public:
    const static size_t NumOutputs = 5;
    typedef const Eigen::Map<const Eigen::ArrayXcf, 0, Eigen::InnerStride<>> map_t;
protected:
    EllipseMethods m_method;
    bool m_debug = false;
    std::shared_ptr<QI::SSFPEcho> m_sequence = nullptr;
    TOutput m_zero;
public:

    EllipseAlgo(EllipseMethods m, std::shared_ptr<QI::SSFPEcho> &seq, bool debug);

    size_t numInputs() const override { return 1; }
    size_t numConsts() const override { return 0; }
    size_t numOutputs() const override { return NumOutputs; }
    size_t dataSize() const override { return m_sequence->size(); }
    size_t outputSize(const int i) const override { return m_sequence->flip().rows(); }
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
                       TInput &resids, TIters &its) const override;
};

} // End namespace QI

#endif // QI_ELLIPSE_ALGO_H