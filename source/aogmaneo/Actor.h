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

        // Defaults
        VisibleLayerDesc()
        :
        size(4, 4, 16),
        radius(2)
        {}
    };

    // Visible layer
    struct VisibleLayer {
        FloatBuffer valueWeights; // Value function weights
        FloatBuffer actionWeights; // Action function weights
    };

    // History sample for delayed updates
    struct HistorySample {
        Array<ByteBuffer> inputCs;
        ByteBuffer hiddenTargetCsPrev;

        float reward;
    };

private:
    Int3 hiddenSize; // Hidden/output/action size

    // Current history size - fixed after initialization. Determines length of wait before updating
    int historySize;
    int supportSize;

    FloatBuffer hiddenProbs;
    FloatBuffer hiddenProbsTemp;
    FloatBuffer hiddenTargetProbs;

    ByteBuffer hiddenCs; // Hidden states

    CircleBuffer<HistorySample> historySamples; // History buffer, fixed length

    // Visible layers and descriptors
    Array<VisibleLayer> visibleLayers;
    Array<VisibleLayerDesc> visibleLayerDescs;

    // --- Kernels ---

    void forward(
        const Int2 &pos,
        const Array<const ByteBuffer*> &inputCs,
        unsigned long* state
    );

    void learn(
        const Int2 &pos,
        const Array<const ByteBuffer*> &inputCs,
        const ByteBuffer* hiddenTargetCsPrev,
        float q,
        float g,
        bool mimic
    );

public:
    float alpha; // Value learning rate
    float beta; // Action learning rate
    float gamma; // Discount factor

    // Defaults
    Actor()
    :
    alpha(0.5f),
    beta(0.01f),
    gamma(0.99f)
    {}

    // Initialized randomly
    void initRandom(
        const Int3 &hiddenSize,
        int historyCapacity,
        int supportSize,
        const Array<VisibleLayerDesc> &visibleLayerDescs
    );

    // Step (get actions and update)
    void step(
        const Array<const ByteBuffer*> &inputCs,
        const ByteBuffer* hiddenTargetCsPrev,
        float reward,
        bool learnEnabled,
        bool mimic
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
    const ByteBuffer &getHiddenCs() const {
        return hiddenCs;
    }

    const FloatBuffer &getHiddenProbs() const {
        return hiddenProbs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }
};
} // namespace aon
