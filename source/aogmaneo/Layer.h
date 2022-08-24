// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Encoder.h"
#include "Predictor.h"

namespace aon {
// A layer with encoder and predictor
class Layer {
private:
    Encoder enc;
    Predictor pred;

public:
    // Defaults
    Layer()
    {}

    // Create with random initialization
    void initRandom(
        const Int3 &hiddenSize,
        const Array<Encoder::VisibleLayerDesc> &visibleLayerDescs,
        bool hasFeedBack
    );

    void stepUp(
        const Array<const IntBuffer*> &inputCIs
    );

    void stepDown(
        const IntBuffer* feedBackCIs
    );

    // Serialization
    int size() const; // Returns size in bytes
    int stateSize() const; // Returns size of state in bytes

    void write(
        StreamWriter &writer
    ) const;

    void read(
        StreamReader &reader
    );

    void writeState(
        StreamWriter &writer
    ) const;

    void readState(
        StreamReader &reader
    );

    Encoder &getEnc() {
        return enc;
    }

    const Encoder &getEnc() const {
        return enc;
    }

    Predictor &getPred() {
        return pred;
    }

    const Predictor &getPred() const {
        return pred;
    }
};
} // Namespace aon

