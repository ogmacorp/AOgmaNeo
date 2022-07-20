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
        FloatBuffer weightsEval;
        FloatBuffer weightsLearn;
    };

    // History sample for delayed updates
    struct HistorySample {
        Array<IntBuffer> inputCIs;
        Array<FloatBuffer> inputActs;
        IntBuffer hiddenTargetCIsPrev;

        float reward;
    };

private:
    Int3 hiddenSize; // Hidden/output/action size

    // Current history size - fixed after initialization. Determines length of wait before updating
    int historySize;

    IntBuffer hiddenCIs; // Hidden states

    FloatBuffer hiddenQs;
    FloatBuffer hiddenTDErrors;

    CircleBuffer<HistorySample> historySamples; // History buffer, fixed length

    // Visible layers and descriptors
    Array<VisibleLayer> visibleLayers;
    Array<VisibleLayerDesc> visibleLayerDescs;

    // --- Kernels ---

    void activate(
        const Int2 &columnPos,
        const Array<const IntBuffer*> &inputCIs,
        const Array<const FloatBuffer*> &inputActs,
        const IntBuffer* hiddenTargetCIsPrev,
        float reward,
        bool learnEnabled
    );

    void learn(
        const Int2 &columnPos,
        int t,
        float r,
        float d
    );

    void generateErrors(
        const Int2 &columnPos,
        FloatBuffer* visibleErrors,
        int t,
        int vli
    ); 

public:
    float lr; // Learning rate
    float drift;
    float discount;
    int nSteps;
    int historyIters;

    // Defaults
    Actor()
    :
    lr(0.01f),
    drift(0.01f),
    discount(0.99f),
    nSteps(5),
    historyIters(16)
    {}

    // Initialized randomly
    void initRandom(
        const Int3 &hiddenSize,
        int historyCapacity,
        const Array<VisibleLayerDesc> &visibleLayerDescs
    );

    // Step (get actions and update)
    void step(
        const Array<const IntBuffer*> &inputCIs,
        const Array<const FloatBuffer*> &inputActs,
        const IntBuffer* hiddenTargetCIsPrev,
        float reward,
        bool learnEnabled
    );

    void generateErrors(
        FloatBuffer* visibleErrors,
        int vli
    );

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
