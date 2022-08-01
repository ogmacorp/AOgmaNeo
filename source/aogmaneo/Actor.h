// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
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
        Array<IntBuffer> inputCIs;
        IntBuffer hiddenTargetCIsPrev;

        float reward;
    };

private:
    Int3 hiddenSize; // Hidden/output/action size

    // Current history size - fixed after initialization. Determines length of wait before updating
    int historySize;

    int numValueCellsPerColumn;

    FloatBuffer hiddenActivations; // Temporary buffer

    IntBuffer hiddenCIs; // Hidden states

    FloatBuffer hiddenValues; // Hidden value function output buffer

    CircleBuffer<HistorySample> historySamples; // History buffer, fixed length

    // Visible layers and descriptors
    Array<VisibleLayer> visibleLayers;
    Array<VisibleLayerDesc> visibleLayerDescs;

    // Value CSDRs
    Array<IntBuffer> valueCIs;

    // --- Kernels ---

    void forward(
        const Int2 &columnPos,
        const Array<const IntBuffer*> &inputCIs,
        unsigned int* state
    );

    void learn(
        const Int2 &columnPos,
        int t,
        float q,
        float g,
        bool mimic
    );

public:
    // For value CSDRs
    float magnification;

    // Hyperparameters
    float vlr; // Value learning rate
    float alr; // Action learning rate
    float discount; // Discount factor
    float temperature; // Exploration amount
    int minSteps; // Minimum steps before sample can be used
    int historyIters; // Number of iterations over samples

    // Defaults
    Actor()
    :
    vlr(0.01f),
    alr(0.01f),
    discount(0.99f),
    temperature(1.0f),
    minSteps(16),
    historyIters(16)
    {}

    // Initialized randomly
    void initRandom(
        const Int3 &hiddenSize,
        int numValueScales,
        int numValueCellsPerColumn,
        int historyCapacity,
        const Array<VisibleLayerDesc> &visibleLayerDescs
    );

    // Step (get actions and update)
    void step(
        const Array<const IntBuffer*> &inputCIs,
        const IntBuffer* hiddenTargetCIsPrev,
        float reward,
        bool learnEnabled,
        bool mimic
    );

    const IntBuffer* getValueCIs(
        int j // Scale index
    ) const;

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

    int getHistoryCapacity() const {
        return historySamples.size();
    }

    int getHistorySize() const {
        return historySize;
    }
};
} // namespace aon
