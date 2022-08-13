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
// A prediction layer (predicts x_(t+1))
class Decoder {
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

        IntBuffer inputCIsPrev; // Previous timestep (prev) input states
        FloatBuffer inputActsPrev;
    };

private:
    Int3 hiddenSize; // Size of the output/hidden/prediction

    FloatBuffer hiddenActs;
    FloatBuffer hiddenActsPrev;

    IntBuffer hiddenCIs; // Hidden state

    // Visible layers and descs
    Array<VisibleLayer> visibleLayers;
    Array<VisibleLayerDesc> visibleLayerDescs;

    // --- Kernels ---

    void activate(
        const Int2 &columnPos,
        const Array<const IntBuffer*> &inputCIs,
        const Array<const FloatBuffer*> &inputActs
    );

    void learn(
        const Int2 &columnPos,
        const IntBuffer* hiddenTargetCIs
    );

    void generateErrors(
        const Int2 &columnPos,
        const IntBuffer* hiddenTargetCIs,
        FloatBuffer* visibleErrors,
        int vli
    ); 

public:
    float lr; // Learning rate

    // Defaults
    Decoder()
    :
    lr(1.0f)
    {}

    // Create with random initialization
    void initRandom(
        const Int3 &hiddenSize, // Hidden/output/prediction size
        const Array<VisibleLayerDesc> &visibleLayerDescs
    );

    // Activate the predictor (predict values)
    void activate(
        const Array<const IntBuffer*> &inputCIs,
        const Array<const FloatBuffer*> &inputActs
    );

    // Learning predictions (update weights)
    void learn(
        const IntBuffer* hiddenTargetCIs
    );

    void stepEnd(
        const Array<const IntBuffer*> &inputCIs,
        const Array<const FloatBuffer*> &inputActs
    );

    void generateErrors(
        const IntBuffer* hiddenTargetCIs,
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

    // Get the hidden activations (predictions)
    const IntBuffer &getHiddenCIs() const {
        return hiddenCIs;
    }

    const FloatBuffer &getHiddenActs() const {
        return hiddenActs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }
};
} // Namespace aon
