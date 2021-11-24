// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Encoder.h"
#include "Decoder.h"

namespace aon {
// Type of hierarchy input layer
enum IOType {
    none = 0,
    prediction = 1
};

// A SPH
class Hierarchy {
public:
    struct IODesc {
        Int3 size;
        IOType type;

        int eRadius; // Encoder radius
        int dRadius; // Decoder radius

        IODesc()
        :
        size(4, 4, 16),
        type(prediction),
        eRadius(2),
        dRadius(2)
        {}

        IODesc(
            const Int3 &size,
            IOType type,
            int eRadius,
            int dRadius
        )
        :
        size(size),
        type(type),
        eRadius(eRadius),
        dRadius(dRadius)
        {}
    };

    // Describes a layer for construction. For the first layer, the IODesc overrides the parameters that are the same name
    struct LayerDesc {
        Int3 hiddenSize; // Size of hidden layer

        int eRadius; // Encoder radius
        int rRadius; // Recurrent radius
        int dRadius; // Decoder radius

        LayerDesc()
        :
        hiddenSize(4, 4, 16),
        eRadius(2),
        rRadius(2),
        dRadius(2)
        {}

        LayerDesc(
            const Int3 &hiddenSize,
            int eRadius,
            int rRadius,
            int dRadius
        )
        :
        hiddenSize(hiddenSize),
        eRadius(eRadius),
        rRadius(rRadius),
        dRadius(dRadius)
        {}
    };

private:
    // Layers
    Array<Encoder> eLayers;
    Array<Array<Decoder>> dLayers;

    // For mapping first layer decoders
    IntBuffer iIndices;
    IntBuffer dIndices;

    // Input dimensions
    Array<Int3> ioSizes;

public:
    // Default
    Hierarchy() {}
    
    // Create a randomly initialized hierarchy
    void initRandom(
        const Array<IODesc> &ioDescs, // Input-output descriptors
        const Array<LayerDesc> &layerDescs // Descriptors for layers
    );

    // Simulation step/tick
    void step(
        const Array<const IntBuffer*> &inputCIs, // Inputs to remember
        const IntBuffer* topProgCIs,
        bool learnEnabled = true // Whether learning is enabled
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

    // Get the number of layers (eLayers)
    int getNumLayers() const {
        return eLayers.size();
    }

    // Get state of highest layer (less verbose when dealing with program-driven learning)
    const IntBuffer &getTopHiddenCIs() const {
        return eLayers[eLayers.size() - 1].getHiddenCIs();
    }

    // Get size of highest layer (less verbose when dealing with program-driven learning)
    const Int3 &getTopHiddenSize() const {
        return eLayers[eLayers.size() - 1].getHiddenSize();
    }

    bool dLayerExists(
        int i
    ) const {
        return dIndices[i] != -1;
    }

    // Importance control
    void setInputImportance(
        int i,
        float importance
    ) {
        eLayers[0].getVisibleLayer(i).importance = importance;
    }

    // Importance control
    float getInputImportance(
        int i
    ) const {
        return eLayers[0].getVisibleLayer(i).importance;
    }

    void setRecurrentImportance(
        int l,
        float importance
    ) {
        eLayers[l].getVisibleLayer(eLayers[l].getNumVisibleLayers() - 1).importance = importance;
    }

    float getRecurrentImportance(
        int l
    ) const {
        return eLayers[l].getVisibleLayer(eLayers[l].getNumVisibleLayers() - 1).importance;
    }

    void setProgImportance(
        int l,
        float importance
    ) {
        cLayers[l].getVisibleLayer(0).importance = importance;
    }

    float getProgImportance(
        int l
    ) const {
        return cLayers[l].getVisibleLayer(0).importance;
    }

    // Retrieve predictions
    const IntBuffer &getPredictionCIs(
        int i // Index of input layer to get predictions for
    ) const {
        return dLayers[0][dIndices[i]].getHiddenCIs();
    }

    // Get input sizes
    const Array<Int3> &getIOSizes() const {
        return ioSizes;
    }

    // Retrieve a sparse coding layer
    Encoder &getELayer(
        int l // Layer index
    ) {
        return eLayers[l];
    }

    // Retrieve a sparse coding layer, const version
    const Encoder &getELayer(
        int l // Layer index
    ) const {
        return eLayers[l];
    }

    // Retrieve predictor layer(s)
    Array<Decoder> &getDLayers(
        int l // Layer index
    ) {
        return dLayers[l];
    }

    // Retrieve predictor layer(s), const version
    const Array<Decoder> &getDLayers(
        int l // Layer index
    ) const {
        return dLayers[l];
    }

    const IntBuffer &getIIndices() const {
        return iIndices;
    }

    const IntBuffer &getDIndices() const {
        return dIndices;
    }
};
} // namespace aon
