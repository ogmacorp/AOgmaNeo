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
// Image coder
class ImageEncoder {
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
        ByteBuffer protos;

        ByteBuffer recons;
    };

private:
    Int3 hiddenSize; // Size of hidden/output layer

    IntBuffer hiddenCIs; // Hidden states

    FloatBuffer hiddenRates;

    // Visible layers and associated descriptors
    Array<VisibleLayer> visibleLayers;
    Array<VisibleLayerDesc> visibleLayerDescs;
    
    // --- Kernels ---
    
    void forward(
        const Int2 &columnPos,
        const Array<const ByteBuffer*> &inputs,
        bool learnEnabled
    );

    void reconstruct(
        const Int2 &columnPos,
        const IntBuffer* reconCIs,
        int vli
    );

public:
    float lr;

    // Defaults
    ImageEncoder()
    :
    lr(0.1f)
    {}

    void initRandom(
        const Int3 &hiddenSize, // Hidden/output size
        const Array<VisibleLayerDesc> &visibleLayerDescs // Descriptors for visible layers
    );

    // Activate the sparse coder (perform sparse coding)
    void step(
        const Array<const ByteBuffer*> &inputs, // Input states
        bool learnEnabled // Whether to learn
    );

    void reconstruct(
        const IntBuffer* reconCIs
    );

    const ByteBuffer &getReconstruction(
        int vli
    ) const {
        return visibleLayers[vli].recons;
    }

    // Serialization
    int size() const; // Returns size in bytes

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
        int vli // Index of visible layer
    ) const {
        return visibleLayers[vli];
    }

    // Get a visible layer descriptor
    const VisibleLayerDesc &getVisibleLayerDesc(
        int vli // Index of visible layer
    ) const {
        return visibleLayerDescs[vli];
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
