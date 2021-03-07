// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "SparseCoder.h"

namespace aon {
// A prediction layer (predicts x_(t+1))
class Predictor {
private:
    Int3 hiddenSize; // Size of the output/hidden/prediction

    FloatBuffer hiddenActivations;

    IntBuffer hiddenCIs; // Hidden state

    FloatBuffer weights;

    // --- Kernels ---

    void forward(
        const Int2 &columnPos
    );

    void learn(
        const Int2 &columnPos,
        const IntBuffer* hiddenTargetCIs
    );

public:
    SparseCoder sc;

    float beta; // Learning rate

    // Defaults
    Predictor()
    :
    beta(0.5f)
    {}

    // Create with random initialization
    void initRandom(
        const Int3 &hiddenSize,
        int intermediateSize,
        const Array<SparseCoder::VisibleLayerDesc> &visibleLayerDescs
    );

    // Activate the predictor (predict values)
    void step(
        const Array<const IntBuffer*> &inputCIs, // Hidden/output/prediction size
        const IntBuffer* hiddenTargetCIs,
        bool learnEnabled
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

    // Get the hidden activations (predictions)
    const IntBuffer &getHiddenCIs() const {
        return hiddenCIs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }
};
} // Namespace aon
