// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Encoder.h"
#include "Decoder.h"
#include "Actor.h"

namespace aon {
// Type of hierarchy input layer
enum IOType {
    none = 0,
    prediction = 1,
    action = 2
};

// A SPH
class Hierarchy {
public:
    struct IODesc {
        Int3 size;
        IOType type;

        int eRadius; // Encoder radius
        int dRadius; // Decoder radius

        int historyCapacity;

        IODesc()
        :
        size(4, 4, 16),
        type(prediction),
        eRadius(2),
        dRadius(2),
        historyCapacity(64)
        {}

        IODesc(
            const Int3 &size,
            IOType type,
            int eRadius,
            int dRadius,
            int historyCapacity
        )
        :
        size(size),
        type(type),
        eRadius(eRadius),
        dRadius(dRadius),
        historyCapacity(historyCapacity)
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
        rRadius(0),
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
    Array<Actor> aLayers;
    Array<IntBuffer> hiddenCIsPrev;

    // For mapping first layer decoders
    IntBuffer iIndices;
    IntBuffer dIndices;

    // Input dimensions
    Array<Int3> ioSizes;
    Array<Byte> ioTypes;

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
        bool learnEnabled = true, // Whether learning is enabled
        float reward = 0.0f // Reward
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

    // Get state of highest layer (less verbose when dealing with goal-driven learning)
    const IntBuffer &getTopHiddenCIs() const {
        return eLayers[eLayers.size() - 1].getHiddenCIs();
    }

    // Get size of highest layer (less verbose when dealing with goal-driven learning)
    const Int3 &getTopHiddenSize() const {
        return eLayers[eLayers.size() - 1].getHiddenSize();
    }

    bool ioLayerExists(
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

    float getInputImportance(
        int i
    ) const {
        return eLayers[0].getVisibleLayer(i).importance;
    }

    void setRecurrentImportance(
        int l,
        float importance
    ) {
        assert(l == 0 ? eLayers[l].getNumVisibleLayers() > ioSizes.size() : eLayers[l].getNumVisibleLayers() > 1);

        eLayers[l].getVisibleLayer(eLayers[l].getNumVisibleLayers() - 1).importance = importance;
    }

    float getRecurrentImportance(
        int l
    ) const {
        assert(l == 0 ? eLayers[l].getNumVisibleLayers() > ioSizes.size() : eLayers[l].getNumVisibleLayers() > 1);

        return eLayers[l].getVisibleLayer(eLayers[l].getNumVisibleLayers() - 1).importance;
    }

    // Retrieve predictions
    const IntBuffer &getPredictionCIs(
        int i
    ) const {
        if (ioTypes[i] == action)
            return aLayers[dIndices[i]].getHiddenCIs();

        return dLayers[0][dIndices[i]].getHiddenCIs();
    }

    // Get input/output sizes
    const Array<Int3> &getIOSizes() const {
        return ioSizes;
    }

    // Get input/output types
    IOType getIOType(
        int i
    ) const {
        return static_cast<IOType>(ioTypes[i]);
    }

    // Retrieve a sparse coding layer
    Encoder &getELayer(
        int l
    ) {
        return eLayers[l];
    }

    // Retrieve a sparse coding layer, const version
    const Encoder &getELayer(
        int l
    ) const {
        return eLayers[l];
    }

    // Retrieve deocder layer(s)
    Array<Decoder> &getDLayers(
        int l
    ) {
        return dLayers[l];
    }

    const Array<Decoder> &getDLayers(
        int l
    ) const {
        return dLayers[l];
    }

    // Retrieve actor layer(s)
    Array<Actor> &getALayers() {
        return aLayers;
    }

    const Array<Actor> &getALayers() const {
        return aLayers;
    }

    // Retrieve by index
    Decoder &getDLayer(
        int l,
        int i
    ) {
        if (l == 0)
            return dLayers[l][dIndices[i]];

        return dLayers[l][i];
    }

    const Decoder &getDLayer(
        int l,
        int i
    ) const {
        if (l == 0)
            return dLayers[l][dIndices[i]];

        return dLayers[l][i];
    }

    Actor &getALayer(
        int i
    ) {
        return aLayers[dIndices[i]];
    }

    const Actor &getALayer(
        int i
    ) const {
        return aLayers[dIndices[i]];
    }

    const IntBuffer &getIIndices() const {
        return iIndices;
    }

    const IntBuffer &getDIndices() const {
        return dIndices;
    }
};
} // namespace aon
