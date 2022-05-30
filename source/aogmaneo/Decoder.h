// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Encoder.h"

namespace aon {
// A prediction layer (predicts x_(t+1))
class Decoder {
private:
    Int3 hiddenSize; // Size of the hidden/hidden/prediction

    IntBuffer hiddenCIs;
    IntBuffer hiddenCIsPrev;

    Encoder support;

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
    float lr; // Learning rate

    Decoder()
    :
    lr(0.1f)
    {}

    // Create with random initialization
    void initRandom(
        const Int3 &hiddenSize, // Hidden/hidden/prediction size
        int supportSize,
        const Array<Encoder::VisibleLayerDesc> &visibleLayerDescs
    );

    // Activate the predictor (predict values) and optionally learn
    void step(
        const Array<const IntBuffer*> &inputCIs,
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

    // Get number of visible layers
    int getNumVisibleLayers() const {
        return support.getNumVisibleLayers();
    }

    // Get a visible layer
    Encoder::VisibleLayer &getVisibleLayer(
        int i // Index of visible layer
    ) {
        return support.getVisibleLayer(i);
    }

    // Get a visible layer
    const Encoder::VisibleLayer &getVisibleLayer(
        int i // Index of visible layer
    ) const {
        return support.getVisibleLayer(i);
    }

    // Get a visible layer descriptor
    const Encoder::VisibleLayerDesc &getVisibleLayerDesc(
        int i // Index of visible layer
    ) const {
        return support.getVisibleLayerDesc(i);
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
