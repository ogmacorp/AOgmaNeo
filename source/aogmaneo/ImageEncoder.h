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
class ImageEncoder {
public:
    enum Mode {
        commit = 0,
        update = 1,
        ignore = 2
    };

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
        FloatBuffer weights0;
        FloatBuffer weights1;

        FloatBuffer weightsRecon;

        ByteBuffer reconstruction;

        float importance;

        VisibleLayer()
        :
        importance(1.0f)
        {}
    };

private:
    Int3 hiddenSize; // Size of hidden/output layer

    Array<Mode> hiddenModes;

    FloatBuffer hiddenMaxActs;

    IntBuffer hiddenCIs;

    IntBuffer hiddenCommits;

    // Visible layers and associated descriptors
    Array<VisibleLayer> visibleLayers;
    Array<VisibleLayerDesc> visibleLayerDescs;
    
    // --- Kernels ---
    
    void activate(
        const Int2 &columnPos,
        const Array<const ByteBuffer*> &inputs,
        unsigned int* state
    );

    void learn(
        const Int2 &columnPos,
        const Array<const ByteBuffer*> &inputs
    );

    void learnReconstruction(
        const Int2 &columnPos,
        const ByteBuffer* inputs,
        int vli
    );

    void reconstruct(
        const Int2 &columnPos,
        const IntBuffer* reconCIs,
        int vli
    );

public:
    float gap;
    float vigilance;
    float lr; // Learning rate
    float rr; // Recon rate
    int lRadius;

    ImageEncoder()
    :
    gap(0.0001f),
    vigilance(0.9f),
    lr(0.01f),
    rr(0.1f),
    lRadius(0)
    {}

    // Create a sparse coding layer with random initialization
    void initRandom(
        const Int3 &hiddenSize, // Hidden/output size
        const Array<VisibleLayerDesc> &visibleLayerDescs // Descriptors for visible layers
    );

    void step(
        const Array<const ByteBuffer*> &inputs, // Input states
        bool learnEnabled = true, // Whether to learn
        bool learnRecon = true // Learning reconstruction
    );

    void reconstruct(
        const IntBuffer* reconCIs
    );

    const ByteBuffer &getReconstruction(
        int vli
    ) const {
        return visibleLayers[vli].reconstruction;
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

    // Get the number of visible layers
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

    // Get the hidden states
    const IntBuffer &getHiddenCIs() const {
        return hiddenCIs;
    }

    // Get the hidden commits
    const IntBuffer &getHiddenCommits() const {
        return hiddenCommits;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }
};
} // namespace aon
