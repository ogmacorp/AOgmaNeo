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
        FloatBuffer weights;
    };

    struct HistorySample {
        IntBuffer inputCIs;
        IntBuffer hiddenTargetCIsPrev;
    };

private:
    Int3 hiddenSize; // Size of the output/hidden/prediction
    int numDendrites;

    FloatBuffer hiddenActivations;

    IntBuffer hiddenCIs; // Hidden state

    // Visible layers and descs
    VisibleLayer vl;
    VisibleLayerDesc vld;

    int historySize;
    CircleBuffer<HistorySample> history;

    // --- Kernels ---

    void forward(
        const Int2 &columnPos,
        const IntBuffer* progCIs,
        const IntBuffer* inputCIs
    );

    void learn(
        const Int2 &columnPos,
        const IntBuffer* hiddenTargetCIsPrev,
        const IntBuffer* inputCIs,
        const IntBuffer* inputCIsPrev,
        float strength
    );

public:
    float lr; // Learning rate
    float boost; // Weaker secondary learning rate
    float discount; // Prediction falloff

    // Defaults
    Decoder()
    :
    lr(0.01f),
    boost(0.001f),
    discount(0.95f)
    {}

    // Create with random initialization
    void initRandom(
        const Int3 &hiddenSize,
        int numDendrites,
        int historyCapacity,
        const VisibleLayerDesc &vld
    );

    // Activate the predictor (predict values)
    void activate(
        const IntBuffer* progCIs,
        const IntBuffer* inputCIs
    );

    // Learning predictions (update weights)
    void learn(
        const IntBuffer* hiddenTargetCIsPrev,
        const IntBuffer* inputCIs
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
        return vl;
    }

    // Get a visible layer
    const VisibleLayer &getVisibleLayer() const {
        return vl;
    }

    // Get a visible layer descriptor
    const VisibleLayerDesc &getVisibleLayerDesc() const {
        return vld;
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
