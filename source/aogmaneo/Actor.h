// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Helpers.h"
#include "SparseMatrix.h"

namespace aon {
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
        SparseMatrix valueWeights; // Value function weights
        ByteBuffer actionWeights; // Action function weights
    };

    // History sample for delayed updates
    struct HistorySample {
        Array<ByteBuffer> inputCs;
        ByteBuffer hiddenTargetCsPrev;

        FloatBuffer hiddenValuesPrev;
        
        float reward;
    };

private:
    Int3 hiddenSize; // Hidden/output/action size

    // Current history size - fixed after initialization. Determines length of wait before updating
    int historySize;

    ByteBuffer hiddenCs; // Hidden states

    FloatBuffer hiddenValues; // Hidden value function output buffer

    CircleBuffer<HistorySample> historySamples; // History buffer, fixed length

    // Visible layers and descriptors
    Array<VisibleLayer> visibleLayers;
    Array<VisibleLayerDesc> visibleLayerDescs;

    // --- Kernels ---

    void forward(
        const Int2 &pos,
        const Array<const ByteBuffer*> &inputCs
    );

    void learn(
        const Int2 &pos,
        const Array<const ByteBuffer*> &inputCsPrev,
        const ByteBuffer* hiddenTargetCsPrev,
        const FloatBuffer* hiddenValuesPrev,
        float q,
        float g,
        bool mimic,
        unsigned long* state
    );

public:
    float alpha; // Value learning rate
    float beta; // Action learning rate
    float gamma; // Discount factor
    int minSteps;
    int historyIters;
    float expScale; // Scaling for exp

    // Defaults
    Actor()
    :
    alpha(0.02f),
    beta(0.1f),
    gamma(0.99f),
    minSteps(8),
    historyIters(8),
    expScale(4.0f)
    {}

    // Initialized randomly
    void initRandom(
        const Int3 &hiddenSize,
        int historyCapacity,
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

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }
};
} // namespace aon
