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
private:
    Int3 hiddenSize;
    int radius;

    IntBuffer goalCIs;
    IntBuffer goalCIsPrev;
    IntBuffer hiddenCIsPrev;

    FloatBuffer hiddenValues;

    FloatBuffer weights;
    FloatBuffer traces;

    void forward(
        const Int2 &columnPos,
        const IntBuffer* hiddenCIs,
        float reward,
        bool learnEnabled
    );

public:
    float lr; // Learning rate
    float discount;
    float traceDecay;

    // Defaults
    RLAdapter()
    :
    lr(0.001f),
    discount(0.99f),
    traceDecay(0.98f)
    {}

    // Create with random initialization
    void initRandom(
        const Int3 &hiddenSize, // Hidden/output/prediction size
        int radius
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

    int getRadius() const {
        return radius;
    }
};
}
