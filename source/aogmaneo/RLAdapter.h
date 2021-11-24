// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Helpers.h"

namespace aon {
// Adapts a program-driven hierarchy to reinforcement learning
class RLAdapter {
private:
    Int3 hiddenSize;

    IntBuffer progCIs;

    FloatBuffer weights;

    void forward(
        const Int2 &columnPos,
        const IntBuffer* hiddenCIs,
        float reward,
        bool learnEnabled
    );

public:
    float lr; // Learning rate

    // Defaults
    RLAdapter()
    :
    lr(0.01f)
    {}

    // Create with random initialization
    void initRandom(
        const Int3 &hiddenSize // Hidden/output/prediction size
    );

    void step(
        float reward,
        const IntBuffer* hiddenCIs,
        bool learnEnabled
    );

    const IntBuffer &getProgCIs() const {
        return progCIs;
    }

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

    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }
};
}
