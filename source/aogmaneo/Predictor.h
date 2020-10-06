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
// A prediction layer (predicts x_(t+1))
class Predictor {
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

        IntBuffer inputCsPrev; // Previous timestep (prev) input states
    };

private:
    Int3 hiddenSize; // Size of the output/hidden/prediction

    FloatBuffer hiddenActivations;

    IntBuffer hiddenCs; // Hidden state

    // Visible layers and descs
    Array<VisibleLayer> visibleLayers;
    Array<VisibleLayerDesc> visibleLayerDescs;

    // --- Kernels ---

    void activate(
        const Int2 &pos,
        const Array<const IntBuffer*> &inputCs
    );

    void learn(
        const Int2 &pos,
        const IntBuffer* hiddenTargetCs
    );

    void generateErrors(
        const Int2 &pos,
        const IntBuffer* hiddenTargetCs,
        FloatBuffer* visibleErrors,
        int vli
    );

public:
    float alpha; // Learning rate

    // Defaults
    Predictor()
    :
    alpha(2.0f)
    {}

    // Create with random initialization
    void initRandom(
        const Int3 &hiddenSize, // Hidden/output/prediction size
        const Array<VisibleLayerDesc> &visibleLayerDescs
    );

    // Activate the predictor (predict values)
    void activate(
        const Array<const IntBuffer*> &inputCs
    );

    // Learning predictions (update weights)
    void learn(
        const IntBuffer* hiddenTargetCs
    );

    // Generate rewards from errors
    void generateErrors(
        const IntBuffer* hiddenTargetCs,
        FloatBuffer* visibleErrors,
        int vli
    );

    // Serialization
    void write(
        StreamWriter &writer
    ) const;

    void read(
        StreamReader &reader
    );

    // Get number of visible layers
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

    // Get the hidden activations (predictions)
    const IntBuffer &getHiddenCs() const {
        return hiddenCs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }
};
} // Namespace aon
