// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Helpers.h"
#include <iostream>

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

        FloatBuffer partialActs;

        IntBuffer inputCIs;
        IntBuffer reconCIs;

        bool useInputs;
        bool needsUpdate;

        float importance;

        VisibleLayer()
        :
        useInputs(false),
        needsUpdate(true),
        importance(1.0f)
        {}
    };

private:
    Int3 hiddenSize; // Size of hidden/output layer

    FloatBuffer hiddenMaxActs;
    FloatBuffer hiddenTotals;

    IntBuffer hiddenCIs; // Hidden states

    // Visible layers and associated descriptors
    Array<VisibleLayer> visibleLayers;
    Array<VisibleLayerDesc> visibleLayerDescs;

    // --- Kernels ---
    
    void forward(
        const Int2 &columnPos
    );

    void learn(
        const Int2 &columnPos
    );

    void reconstruct(
        const Int2 &columnPos,
        int vli
    );

public:
    float gap;
    float vigilance;
    float lr;
    int groupRadius;

    // Defaults
    Encoder()
    :
    gap(0.0001f),
    vigilance(0.7f),
    lr(0.1f),
    groupRadius(2)
    {}

    // Create a sparse coding layer with random initialization
    void initRandom(
        const Int3 &hiddenSize, // Hidden/output size
        const Array<VisibleLayerDesc> &visibleLayerDescs // Descriptors for visible layers
    );

    void setInputCIs(
        const IntBuffer* inputCIs,
        int vli
    );

    // Activate the sparse coder (perform sparse coding)
    void activate();

    void learn();

    void reconstruct(
        int vli
    );

    const IntBuffer &getReconCIs(
        int vli
    ) const {
        return visibleLayers[vli].reconCIs;
    }

    void clearState() {
        hiddenCIs.fill(0);
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

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }
};
} // namespace aon
