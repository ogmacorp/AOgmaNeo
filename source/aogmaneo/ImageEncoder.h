// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Encoder.h"

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
        FloatBuffer protos;

        FloatBuffer reconstruction;
    };

    struct HigherLayerDesc {
        Int3 hiddenSize;

        int radius;

        HigherLayerDesc()
        :
        hiddenSize(4, 4, 16),
        radius(2)
        {}
    };

private:
    Int3 hiddenSize; // Size of hidden/output layer

    IntBuffer hiddenCIs; // Hidden states

    Array<Encoder> higherLayers;
    Array<IntBuffer> higherLayerReconCIs;

    // Visible layers and associated descriptors
    Array<VisibleLayer> visibleLayers;
    Array<VisibleLayerDesc> visibleLayerDescs;
    
    // --- Kernels ---
    
    void forward(
        const Int2 &columnPos,
        const Array<const FloatBuffer*> &inputCIs,
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
    lr(0.01f)
    {}

    // Create a sparse coding layer with random initialization
    void initRandom(
        const Int3 &hiddenSize, // Hidden/output size
        const Array<VisibleLayerDesc> &visibleLayerDescs // Descriptors for visible layers
    ) {
        initRandom(hiddenSize, visibleLayerDescs, Array<HigherLayerDesc>()); // Signify no higher layers
    }

    void initRandom(
        const Int3 &hiddenSize, // Hidden/output size
        const Array<VisibleLayerDesc> &visibleLayerDescs, // Descriptors for visible layers
        const Array<HigherLayerDesc> &higherLayerDescs
    );

    // Activate the sparse coder (perform sparse coding)
    void step(
        const Array<const FloatBuffer*> &inputs, // Input states
        bool learnEnabled // Whether to learn
    );

    void reconstruct(
        const IntBuffer* reconCIs
    );

    const FloatBuffer &getReconstruction(
        int vli
    ) const {
        return visibleLayers[vli].reconstruction;
    }

    int getNumHigherLayers() const {
        return higherLayers.size();
    }

    Encoder &getHigherLayer(
        int l
    ) {
        return higherLayers[l];
    }

    const Encoder &getHigherLayer(
        int l
    ) const {
        return higherLayers[l];
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

    // Get the output states
    const IntBuffer &getOutputCIs() const {
        if (higherLayers.size() > 0)
            return higherLayers[higherLayers.size() - 1].getHiddenCIs();

        return hiddenCIs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }

    // Get the output size
    const Int3 &getOutputSize() const {
        if (higherLayers.size() > 0)
            return higherLayers[higherLayers.size() - 1].getHiddenSize();

        return hiddenSize;
    }
};
} // namespace aon

