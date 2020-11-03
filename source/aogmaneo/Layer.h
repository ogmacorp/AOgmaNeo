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
class Layer {
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

    struct VisibleLayer {
        FloatBuffer weights;

        FloatBuffer visibleActivations;
        IntBuffer visibleCIs;
        IntBuffer visibleRandomCIs;
    };

private:
    Int3 hiddenSize; // Size of hidden/output layer

    FloatBuffer hiddenActivations;
    IntBuffer hiddenCIs; // Hidden states
    IntBuffer hiddenRandomCIs; // With Boltzmann distribution

    // Visible layers and associated descriptors
    Array<VisibleLayer> visibleLayers;
    Array<VisibleLayerDesc> visibleLayerDescs;

    // --- Kernels ---
    
    void forward(
        const Int2 &columnPos,
        const Array<const IntBuffer*> &inputCIs,
        unsigned int* state
    );

    void backward(
        const Int2 &columnPos,
        int vli
    );

    void reconBackward(
        const Int2 &columnPos,
        int vli,
        unsigned int* state
    );

    void reconForward(
        const Int2 &columnPos,
        unsigned int* state
    );

    void learn(
        const Int2 &columnPos,
        const Array<const IntBuffer*> &inputCIs
    );

public:
    float alpha; // Learning rate decay
    int gibbsIters; // Gibbs sampling iterations
    
    // Defaults
    Layer()
    :
    alpha(0.03f),
    gibbsIters(3)
    {}

    // Create a sparse coding layer with random initialization
    void initRandom(
        const Int3 &hiddenSize, // Hidden/output size
        const Array<VisibleLayerDesc> &visibleLayerDescs // Descriptors for visible layers
    );

    // Activate the layer (RBM update)
    void activate(
        const Array<const IntBuffer*> &inputCIs, // Input states, pass nullptr to not hold constant
        bool generate // Whether to perform Gibbs sampling and prepare for learning
    );

    // Learn from full configuration, will perform Gibbs sampling
    void learn(
        const Array<const IntBuffer*> &inputCIs
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
