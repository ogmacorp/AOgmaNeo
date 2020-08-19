// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Helpers.h"

namespace aon {
// Sparse coder
class SparseCoder {
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
        ByteBuffer weights; // Binary weight matrix
    };

private:
    Int3 hiddenSize; // Size of hidden/output layer
    int lRadius; // Lateral radius

    FloatBuffer hiddenStimuli;
    FloatBuffer hiddenActivations;

    ByteBuffer hiddenCs; // Hidden states
    ByteBuffer hiddenCsTemp; // Temporary hidden states
    FloatBuffer hiddenRates; // Learning rates

    // Visible layers and associated descriptors
    Array<VisibleLayer> visibleLayers;
    Array<VisibleLayerDesc> visibleLayerDescs;

    ByteBuffer laterals; // Lateral weights
    
    // --- Kernels ---
    
    void forward(
        const Int2 &pos,
        const Array<const ByteBuffer*> &inputCs
    );

    void inhibit(
        const Int2 &pos
    );

    void learn(
        const Int2 &pos,
        const Array<const ByteBuffer*> &inputCs
    );

public:
    float alpha; // Learning rate decay
    float gamma; // Activation decay
    int explainIters; // Explaining-away iterations
    
    // Defaults
    SparseCoder()
    :
    alpha(0.02f),
    gamma(0.8f),
    explainIters(8)
    {}

    // Create a sparse coding layer with random initialization
    void initRandom(
        const Int3 &hiddenSize, // Hidden/output size
        int lRadius, // Lateral radius
        const Array<VisibleLayerDesc> &visibleLayerDescs // Descriptors for visible layers
    );

    // Activate the sparse coder (perform sparse coding)
    void step(
        const Array<const ByteBuffer*> &inputCs, // Input states
        bool learnEnabled // Whether to learn
    );

    // Serialization
    void write(
        StreamWriter &writer
    ) const;

    void read(
        StreamReader &reader
    );

    // Get the number of visible layers
    int getNumVisibleLayers() const {
        return visibleLayers.size();
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
    const ByteBuffer &getHiddenCs() const {
        return hiddenCs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }
};
} // namespace aon
