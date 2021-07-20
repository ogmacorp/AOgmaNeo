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
        SByteBuffer weights;

        ByteBuffer inputCIsPrev; // Previous timestep (prev) input states
    };

private:
    Int3 hiddenSize; // Size of the output/hidden/prediction

    FloatBuffer hiddenActivations;

    ByteBuffer hiddenCIs; // Hidden state

    // Visible layers and descs
    Array<VisibleLayer> visibleLayers;
    Array<VisibleLayerDesc> visibleLayerDescs;

    // --- Kernels ---

    void forward(
        const Int2 &columnPos,
        const Array<const ByteBuffer*> &inputCIs
    );

    void learn(
        const Int2 &columnPos,
        const ByteBuffer* hiddenTargetCIs
    );

public:
    float lr; // Learning rate
    float scale;

    // Defaults
    Decoder()
    :
    lr(0.2f),
    scale(8.0f)
    {}

    // Create with random initialization
    void initRandom(
        const Int3 &hiddenSize, // Hidden/output/prediction size
        const Array<VisibleLayerDesc> &visibleLayerDescs
    );

    // Activate the predictor (predict values)
    void activate(
        const Array<const ByteBuffer*> &inputCIs
    );

    // Learning predictions (update weights)
    void learn(
        const ByteBuffer* hiddenTargetCIs
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

    // Get number of visible layers
    int getNumVisibleLayers() const {
        return visibleLayers.size();
    }

    // Get a visible layer
    VisibleLayer &getVisibleLayer(
        int i // Index of visible layer
    ) {
        return visibleLayers[i];
    }

    // Get a visible layer
    const VisibleLayer &getVisibleLayer(
        int i // Index of visible layer
    ) const {
        return visibleLayers[i];
    }

    // Get a visible layer descriptor
    const VisibleLayerDesc &getVisibleLayerDesc(
        int i // Index of visible layer
    ) const {
        return visibleLayerDescs[i];
    }

    // Get the hidden activations (predictions)
    const ByteBuffer &getHiddenCIs() const {
        return hiddenCIs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }
};
} // Namespace aon
