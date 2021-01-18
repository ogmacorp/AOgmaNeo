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
// A reinforcement learning layer
class Actor {
public:
    // Visible layer descriptor
    struct VisibleLayerDesc {
        Int3 size; // Visible/input size

        int radius; // Radius onto input

        unsigned char recurrent;

        // Defaults
        VisibleLayerDesc()
        :
        size(4, 4, 16),
        radius(2),
        recurrent(false)
        {}
    };

    // Visible layer
    struct VisibleLayer {
        FloatBuffer weights; // Weights
        FloatBuffer traces; // Eligibility traces
    };

private:
    Int3 hiddenSize; // Hidden/output/action size

    IntBuffer hiddenCIs; // Hidden states

    // Visible layers and descriptors
    Array<VisibleLayer> visibleLayers;
    Array<VisibleLayerDesc> visibleLayerDescs;

    // --- Kernels ---

    void activate(
        const Int2 &pos,
        const Array<const IntBuffer*> &inputCIs
    );

    void learn(
        const Int2 &pos,
        const FloatBuffer* hiddenErrors
    );

public:
    float alpha;
    float traceDecay;

    // Defaults
    Actor()
    :
    alpha(0.001f),
    traceDecay(0.95f)
    {}

    // Initialized randomly
    void initRandom(
        const Int3 &hiddenSize,
        const Array<VisibleLayerDesc> &visibleLayerDescs
    );

    void activate(
        const Array<const IntBuffer*> &inputCIs
    );

    void learn(
        const FloatBuffer* hiddenErrors
    );

    void clearTraces();

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
    const VisibleLayer &getVisibleLayer(
        int i // Index of layer
    ) const {
        return visibleLayers[i];
    }

    // Get a visible layer descriptor
    const VisibleLayerDesc &getVisibleLayerDesc(
        int i // Index of layer
    ) const {
        return visibleLayerDescs[i];
    }

    // Get hidden state/output/actions
    const IntBuffer &getHiddenCIs() const {
        return hiddenCIs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }
};
} // namespace aon
