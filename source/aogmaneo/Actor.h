// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Helpers.h"

namespace aon {
// A reinforcement learning layer
class Actor {
public:
    // Visible layer descriptor
    struct VisibleLayerDesc {
        Int3 size; // Visible/input size

        int radius; // Radius onto input

        // Defaults
        VisibleLayerDesc()
        :
        size(4, 4, 16),
        radius(2)
        {}
    };

    // Visible layer
    struct VisibleLayer {
        FloatBuffer valueWeights; // Value function weights
        FloatBuffer actionWeights; // Action function weights
        
        // Prev versions
        FloatBuffer valueWeightsPrev;
        FloatBuffer actionWeightsPrev;
    };

    // History sample for delayed updates
    struct HistorySample {
        IntBuffer inputCIs;
        IntBuffer hiddenTargetCIsPrev;

        float reward;
    };

private:
    Int3 hiddenSize; // Hidden/output/action size

    // Current history size - fixed after initialization. Determines length of wait before updating
    int historySize;

    FloatBuffer hiddenActs; // Temporary buffer

    IntBuffer hiddenCIs; // Hidden states

    CircleBuffer<HistorySample> historySamples; // History buffer, fixed length

    // Visible layers and descriptors
    VisibleLayer vl;
    VisibleLayerDesc vld;

    // --- Kernels ---

    void forward(
        const Int2 &columnPos,
        const IntBuffer* nextCIs,
        const IntBuffer* inputCIs,
        unsigned int* state
    );

    void learn(
        const Int2 &columnPos,
        int t,
        float r,
        float d,
        bool mimic
    );

public:
    float vlr; // Value learning rate
    float alr; // Action learning rate
    float discount; // Discount factor
    float temperature; // Exploration amount
    int minSteps; // Minimum steps before sample can be used
    int historyIters; // Number of iterations over samples

    // Defaults
    Actor()
    :
    vlr(0.01f),
    alr(0.01f),
    discount(0.99f),
    temperature(1.0f),
    minSteps(16),
    historyIters(16)
    {}

    // Initialized randomly
    void initRandom(
        const Int3 &hiddenSize,
        int historyCapacity,
        const VisibleLayerDesc &vld
    );

    // Step (get actions and update)
    void step(
        const IntBuffer* nextCIs,
        const IntBuffer* inputCIs,
        const IntBuffer* hiddenTargetCIsPrev,
        float reward,
        bool learnEnabled,
        bool mimic
    );

    void clearState();

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

    // Get a visible layer
    const VisibleLayer &getVisibleLayer() const {
        return vl;
    }

    // Get a visible layer descriptor
    const VisibleLayerDesc &getVisibleLayerDesc() const {
        return vld;
    }

    // Get hidden state/output/actions
    const IntBuffer &getHiddenCIs() const {
        return hiddenCIs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }

    int getHistoryCapacity() const {
        return historySamples.size();
    }

    int getHistorySize() const {
        return historySize;
    }
};
} // namespace aon
