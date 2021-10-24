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
    int numGoals;

    IntBuffer goalCIs;

    FloatBuffer protos;
    FloatBuffer values;
    FloatBuffer traces;

    FloatBuffer rates;

public:
    float glr; // Goal learning rate
    float vlr; // Value learning rate
    float falloff;
    float discount;
    float traceDecay;

    // Defaults
    RLAdapter()
    :
    glr(0.1f),
    vlr(0.01f),
    falloff(0.01f),
    discount(0.99f),
    traceDecay(0.01f)
    {}

    // Create with random initialization
    void initRandom(
        const Int3 &hiddenSize, // Hidden/output/prediction size
        int numGoals
    );

    void step(
        const IntBuffer* hiddenCIs,
        float reward,
        bool learnEnabled,
        bool stateUpdate
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
