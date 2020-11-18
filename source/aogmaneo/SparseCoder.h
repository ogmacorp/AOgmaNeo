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
        size(4, 4, 32),
        radius(2)
        {}
    };

    // Visible layer
    struct VisibleLayer {
        ByteBuffer weights0;
        ByteBuffer weights1; // Complement
        ByteBuffer mask; // Mask away units
    };

private:
    Int3 hiddenSize; // Size of hidden/output layer

    IntBuffer hiddenCommits;
    FloatBuffer hiddenActivations;
    FloatBuffer hiddenMatches;

    IntBuffer hiddenCIs; // Hidden states

    // Visible layers and associated descriptors
    Array<VisibleLayer> visibleLayers;
    Array<VisibleLayerDesc> visibleLayerDescs;
    
    // --- Kernels ---
    
    void forward(
        const Int2 &columnPos,
        const Array<const IntBuffer*> &inputCIs,
        bool learnEnabled
    );

public:
    float alpha; // Activation parameter
    float beta; // Weight learning rate
    float vigilance; // Vigilance parameter

    // Defaults
    SparseCoder()
    :
    alpha(0.01f),
    beta(0.5f),
    vigilance(0.5f)
    {}

    // Create a sparse coding layer with random initialization
    void initRandom(
        const Int3 &hiddenSize, // Hidden/output size
        const Array<VisibleLayerDesc> &visibleLayerDescs // Descriptors for visible layers
    );

    // Activate the sparse coder (perform sparse coding)
    void step(
        const Array<const IntBuffer*> &inputCIs, // Input states
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
    const IntBuffer &getHiddenCIs() const {
        return hiddenCIs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }
};
} // namespace aon
