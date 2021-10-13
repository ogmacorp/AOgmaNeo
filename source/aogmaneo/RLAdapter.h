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
// Adapts a goal-driven hierarchy to reinforcement learning
class RLAdapter {
public:
    struct Sample {
        IntBuffer hiddenCIs;

        float reward;
    };

private:
    Int3 hiddenSize;

    IntBuffer goalCIs;

    int numSamples;
    CircleBuffer<Sample> samples;

    void select(
        const Int2 &columnPos
    );

public:
    float lr; // Learning rate
    int minOverlap;

    // Defaults
    RLAdapter()
    :
    lr(0.5f),
    minOverlap(16)
    {}

    void init(
        const Int3 &hiddenSize,
        int maxSamples
    );

    void step(
        const IntBuffer* hiddenCIs,
        float reward,
        bool learnEnabled
    );

    const IntBuffer &getGoalCIs() const {
        return goalCIs;
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
