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
// Adapts a prog-driven hierarchy to reinforcement learning
class StateAdapter {
public:
    struct HistorySample {
        IntBuffer hiddenCIs;
    };

private:
    Int3 hiddenSize;
    int radius;

    IntBuffer progCIs;

    FloatBuffer weights;

    CircleBuffer<HistorySample> history;
    int historySize;

    void forward(
        const Int2 &columnPos,
        const IntBuffer* goalCIs,
        const IntBuffer* hiddenCIs
    );

    void learn(
        const Int2 &columnPos,
        int t1,
        int t2
    );

public:
    float lr; // Learning rate rate
    float discount;
    int historyIters;

    // Defaults
    StateAdapter()
    :
    lr(0.1f),
    discount(0.9f),
    historyIters(16)
    {}

    // Create with random initialization
    void initRandom(
        const Int3 &hiddenSize, // Hidden/output/prediction size
        int radius,
        int historyCapacity
    );

    void step(
        const IntBuffer* goalCIs,
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
