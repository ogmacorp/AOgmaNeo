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

        int ffRadius; // Feed forward radius
        int fbRadius; // Feed back radius

        IODesc()
        :
        size(4, 4, 16),
        type(prediction),
        ffRadius(2),
        fbRadius(2)
        {}

        IODesc(
            const Int3 &size,
            IOType type,
            int ffRadius,
            int fbRadius
        )
        :
        size(size),
        type(type),
        ffRadius(ffRadius),
        fbRadius(fbRadius)
        {}
    };

    // Describes a layer for construction. For the first layer, the IODesc overrides the parameters that are the same name
    struct LayerDesc {
        Int3 hiddenSize; // Size of hidden layer
        Int2 clumpSize;

        int ffRadius; // Encoder radius
        int rRadius; // Decoder radius
        int fbRadius; // Feed back radius

        LayerDesc()
        :
        hiddenSize(4, 4, 16),
        clumpSize(2, 2),
        ffRadius(2),
        rRadius(2),
        fbRadius(2)
        {}

        LayerDesc(
            const Int3 &hiddenSize,
            const Int2 &clumpSize,
            int ffRadius,
            int rRadius,
            int fbRadius
        )
        :
        hiddenSize(hiddenSize),
        clumpSize(clumpSize),
        ffRadius(ffRadius),
        rRadius(rRadius),
        fbRadius(fbRadius)
        {}
    };

private:
    // Layers
    Array<Encoder> eLayers;
    Array<Array<Ptr<Decoder>>> dLayers;
    Array<IntBuffer> hiddenCIsPrev;

    // Input dimensions
    Array<Int3> inputSizes;

public:
    // Default
    Hierarchy() {}

    // Copy
    Hierarchy(
        const Hierarchy &other // Hierarchy to copy from
    ) {
        *this = other;
    }

    // Assignment
    const Hierarchy &operator=(
        const Hierarchy &other // Hierarchy to assign from
    );
    
    // Create a randomly initialized hierarchy
    void initRandom(
        const Array<IODesc> &ioDescs, // Input-output descriptors
        const Array<LayerDesc> &layerDescs // Descriptors for layers
    );

    // Simulation step/tick
    void step(
        const Array<const IntBuffer*> &inputCIs, // Inputs to remember
        const IntBuffer* topGoalCIs,
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

    void setImportance(
        int i,
        float importance
    ) {
        eLayers[0].getVisibleLayer(i).importance = importance;
    }

    float getImportance(
        int i
    ) const {
        return eLayers[0].getVisibleLayer(i).importance;
    }

    void setRecurrence(
        int l,
        float recurrence
    ) {
        eLayers[l].getVisibleLayer(eLayers[l].getNumVisibleLayers() - 1).importance = recurrence;
    }

    float getRecurrence(
        int l
    ) const {
        return eLayers[l].getVisibleLayer(eLayers[l].getNumVisibleLayers() - 1).importance;
    }

    // Retrieve predictions
    const IntBuffer &getPredictionCIs(
        int i // Index of input layer to get predictions for
    ) const {
        assert(dLayers[0][i] != nullptr);

        return dLayers[0][i]->getHiddenCIs();
    }

    // Get input sizes
    const Array<Int3> &getInputSizes() const {
        return inputSizes;
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
    Array<Ptr<Decoder>> &getDLayers(
        int l // Layer index
    ) {
        return dLayers[l];
    }

    // Retrieve predictor layer(s), const version
    const Array<Ptr<Decoder>> &getDLayers(
        int l // Layer index
    ) const {
        return dLayers[l];
    }
};
} // namespace aon
