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
// A prediction layer (predicts x_(t+1))
class Decoder {
public:
    // Visible layer descriptor
    struct VisibleLayerDesc {
        Int3 size; // Size of input
        int gSizeZ; // Size of input for goals (z only)

        int radius; // Radius onto input

        // Defaults
        VisibleLayerDesc()
        :
        size(4, 4, 16),
        gSizeZ(16),
        radius(2)
        {}
    };

    // Visible layer
    struct VisibleLayer {
        FloatBuffer iWeights; // Input
        FloatBuffer fbWeights; // Feedback
        FloatBuffer iRates;
        FloatBuffer fbRates;

        FloatBuffer iGates;
        FloatBuffer fbGates;
    };

    struct HistorySample {
        IntBuffer inputCIs;
        IntBuffer actualCIs;
        IntBuffer hiddenTargetCIs;
    };

private:
    Int3 hiddenSize; // Size of the output/hidden/prediction
    Byte hasFeedBack;

    FloatBuffer hiddenActivations;
    IntBuffer hiddenCIs; // Hidden state

    // Visible layers and descs
    VisibleLayer visibleLayer;
    VisibleLayerDesc visibleLayerDesc;

    int historySize;
    CircleBuffer<HistorySample> history;

    // --- Kernels ---

    void forward(
        const Int2 &columnPos,
        const IntBuffer* goalCIs,
        const IntBuffer* inputCIs,
        const IntBuffer* feedBackCIs
    );

    void backward(
        const Int2 &columnPos,
        int t1,
        int t2
    );

    void learn(
        const Int2 &columnPos,
        int t1,
        int t2,
        float modulation
    );

public:
    float lr; // Learning rate
    float decay;
    float discount; // Discount factor
    int historyIters;
    int maxSteps;

    // Defaults
    Decoder()
    :
    lr(0.2f),
    decay(0.002f),
    discount(0.9f),
    historyIters(16),
    maxSteps(8)
    {}

    // Create with random initialization
    void initRandom(
        const Int3 &hiddenSize, // Hidden/output/prediction size
        int historyCapacity,
        const VisibleLayerDesc &visibleLayerDesc,
        bool hasFeedBack
    );

    // Activate the predictor (predict values)
    void step(
        const IntBuffer* goalCIs,
        const IntBuffer* actualCIs,
        const IntBuffer* inputCIs,
        const IntBuffer* feedBackCIs,
        const IntBuffer* hiddenTargetCIs,
        bool learnEnabled,
        bool stateUpdate
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

    // Get a visible layer
    VisibleLayer &getVisibleLayer() {
        return visibleLayer;
    }

    // Get a visible layer
    const VisibleLayer &getVisibleLayer() const {
        return visibleLayer;
    }

    // Get a visible layer descriptor
    const VisibleLayerDesc &getVisibleLayerDesc() const {
        return visibleLayerDesc;
    }

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
