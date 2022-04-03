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
// Sparse coder
class Encoder {
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
        FloatBuffer rates;

        IntBuffer inputCIsPrev;

        FloatBuffer reconstruction;

        float importance;

        VisibleLayer()
        :
        importance(1.0f)
        {}
    };

private:
    Int3 hiddenSize; // Size of hidden/output layer

    FloatBuffer hiddenAccums;

    IntBuffer hiddenCIs;

    FloatBuffer hiddenGates;

    // Visible layers and associated descriptors
    Array<VisibleLayer> visibleLayers;
    Array<VisibleLayerDesc> visibleLayerDescs;
    
    // --- Kernels ---
    
    void forward(
        const Int2 &columnPos,
        const Array<const IntBuffer*> &inputCIs,
        const FloatBuffer* hiddenErrors,
        int it,
        bool learnEnabled
    );

    void backward(
        const Int2 &columnPos,
        const IntBuffer* inputCIs,
        int vli
    );

    void learn(
        const Int2 &columnPos,
        const IntBuffer* inputCIs,
        int vli
    );

    void reconstruct(
        const Int2 &columnPos,
        const IntBuffer* hiddenCIs,
        IntBuffer* reconCIs,
        int vli
    );

public:
    int explainIters;
    float lr;
    float decay;

    Encoder()
    :
    explainIters(8),
    lr(0.01f),
    decay(0.002f)
    {}

    // Create a sparse coding layer with random initialization
    void initRandom(
        const Int3 &hiddenSize, // Hidden/output size
        const Array<VisibleLayerDesc> &visibleLayerDescs // Descriptors for visible layers
    );

    void step(
        const Array<const IntBuffer*> &inputCIs, // Input states
        const FloatBuffer* hiddenErrors,
        bool learnEnabled // Whether to learn
    );

    void reconstruct(
        const IntBuffer* hiddenCIs,
        IntBuffer* reconCIs,
        int vli
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

    // Get the number of visible layers
    int getNumVisibleLayers() const {
        return visibleLayers.size();
    }

    // Get a visible layer
    VisibleLayer &getVisibleLayer(
        int vli // Index of visible layer
    ) {
        return visibleLayers[vli];
    }

    // Get a visible layer
    const VisibleLayer &getVisibleLayer(
        int vli // Index of visible layer
    ) const {
        return visibleLayers[vli];
    }

    // Get a visible layer descriptor
    const VisibleLayerDesc &getVisibleLayerDesc(
        int vli // Index of visible layer
    ) const {
        return visibleLayerDescs[vli];
    }

    // Get the hidden states
    const IntBuffer &getHiddenCIs() const {
        return hiddenCIs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }
};
} // namespace aon

