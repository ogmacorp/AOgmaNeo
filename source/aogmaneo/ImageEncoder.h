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
// Visible layer descriptor
struct ImageEncoderVisibleLayerDesc {
    Int3 size; // Size of input

    int radius; // Radius onto input

    // Defaults
    ImageEncoderVisibleLayerDesc()
    :
    size(4, 4, 16),
    radius(2)
    {}
};

// Image coder
class ImageEncoder {
public:
    // Visible layer
    struct VisibleLayer {
        ByteBuffer weights; // Byte weight matrix

        ByteBuffer reconstruction;
    };

private:
    Int3 hiddenSize; // Size of hidden/output layer

    ByteBuffer hiddenCs; // Hidden states

    // Visible layers and associated descriptors
    Array<VisibleLayer> visibleLayers;
    Array<ImageEncoderVisibleLayerDesc> visibleLayerDescs;
    
    // --- Kernels ---
    
    void forward(
        const Int2 &pos,
        const Array<const ByteBuffer*> &inputCs,
        bool learnEnabled
    );

    void reconstruct(
        const Int2 &pos,
        const ByteBuffer* reconCs,
        int vli
    );

public:
    // Defaults
    ImageEncoder()
    {}

    // Create a sparse coding layer with random initialization
    void initRandom(
        const Int3 &hiddenSize, // Hidden/output size
        const Array<ImageEncoderVisibleLayerDesc> &visibleLayerDescs // Descriptors for visible layers
    );

    // Activate the sparse coder (perform sparse coding)
    void step(
        const Array<const ByteBuffer*> &inputCs, // Input states
        bool learnEnabled // Whether to learn
    );

    void reconstruct(
        const ByteBuffer* reconCs
    );

    const ByteBuffer &getReconstruction(
        int i
    ) {
        return visibleLayers[i].reconstruction;
    }

    // Get the number of visible layers
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
    const ImageEncoderVisibleLayerDesc &getVisibleLayerDesc(
        int i // Index of visible layer
    ) const {
        return visibleLayerDescs[i];
    }

    // Get the hidden states
    const ByteBuffer &getHiddenCs() const {
        return hiddenCs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }
};
} // namespace aon
