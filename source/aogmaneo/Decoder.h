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
        FloatBuffer weightsPrev;
    };

private:
    Int3 hiddenSize; // Size of the output/hidden/prediction

    FloatBuffer hiddenErrors;

    IntBuffer hiddenCIs; // Hidden state

    // Visible layers and descs
    VisibleLayer vl;
    VisibleLayerDesc vld;

    // --- Kernels ---

    void activate(
        const Int2 &columnPos,
        const IntBuffer* goalCIs,
        const FloatBuffer* inputActs
    );

    void determineErrors(
        const Int2 &columnPos,
        const IntBuffer* hiddenTargetCIs,
        const IntBuffer* inputCIs,
        const FloatBuffer* inputActsPrev
    );

    void propagateErrors(
        const Int2 &columnPos,
        const IntBuffer* hiddenTargetCIs,
        FloatBuffer* visibleErrors
    ); 

    void learn(
        const Int2 &columnPos,
        const IntBuffer* inputCIs,
        const FloatBuffer* inputActsPrev
    );

public:
    float lr; // Learning rate

    // Defaults
    Decoder()
    :
    lr(0.5f)
    {}

    // Create with random initialization
    void initRandom(
        const Int3 &hiddenSize, // Hidden/output/prediction size
        const VisibleLayerDesc &vld
    );

    // Activate the predictor (predict values)
    void activate(
        const IntBuffer* goalCIs,
        const FloatBuffer* inputActs
    );

    void learn(
        const IntBuffer* inputCIs,
        const FloatBuffer* inputActsPrev
    );

    void generateErrors(
        const IntBuffer* hiddenTargetCIs,
        const IntBuffer* inputCIs,
        const FloatBuffer* inputActsPrev,
        FloatBuffer* visibleErrors
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

    // Get a visible layer
    VisibleLayer &getVisibleLayer() {
        return vl;
    }

    // Get a visible layer
    const VisibleLayer &getVisibleLayer() const {
        return vl;
    }

    // Get a visible layer descriptor
    const VisibleLayerDesc &getVisibleLayerDesc() const {
        return vld;
    }

    // Get the hidden activations (predictions)
    const IntBuffer &getHiddenCIs() const {
        return hiddenCIs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }
};
} // Namespace aon
