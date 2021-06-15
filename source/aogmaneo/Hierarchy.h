// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Encoder.h"
#include "Decoder.h"

namespace aon {
// A SPH
class Hierarchy {
public:
    struct IODesc {
        Int3 size;

        int eRadius; // Encoder radius
        int dRadius; // Decoder radius

        int historyCapacity;

        IODesc()
        :
        size(4, 4, 16),
        eRadius(2),
        dRadius(2),
        historyCapacity(16)
        {}

        IODesc(
            const Int3 &size,
            int eRadius,
            int dRadius,
            int historyCapacity
        )
        :
        size(size),
        eRadius(eRadius),
        dRadius(dRadius),
        historyCapacity(historyCapacity)
        {}
    };

    // Describes a layer for construction. For the first layer, the IODesc overrides the parameters that are the same name
    struct LayerDesc {
        Int3 hiddenSize; // Size of hidden layer

        int eRadius; // Encoder radius
        int dRadius; // Decoder radius
        int rRadius; // Recurrent radius

        int historyCapacity;

        int ticksPerUpdate; // Number of ticks a layer takes to update (relative to previous layer)
        int temporalHorizon; // Temporal distance into the past addressed by the layer. Should be greater than or equal to ticksPerUpdate

        LayerDesc()
        :
        hiddenSize(4, 4, 16),
        eRadius(2),
        dRadius(2),
        rRadius(2),
        historyCapacity(16),
        ticksPerUpdate(2),
        temporalHorizon(2)
        {}

        LayerDesc(
            const Int3 &hiddenSize,
            int eRadius,
            int dRadius,
            int historyCapacity,
            int ticksPerUpdate,
            int temporalHorizon
        )
        :
        hiddenSize(hiddenSize),
        eRadius(eRadius),
        dRadius(dRadius),
        historyCapacity(historyCapacity),
        ticksPerUpdate(ticksPerUpdate),
        temporalHorizon(temporalHorizon)
        {}
    };

private:
    // Layers
    Array<Encoder> eLayers;
    Array<Array<Decoder>> dLayers;

    // Input dimensions
    Array<Int3> inputSizes;

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

    // Importance control
    void setImportance(
        int i,
        float importance
    ) {
        eLayers[0].getVisibleLayer(i).importance = importance;
    }

    // Importance control
    float getImportance(
        int i
    ) const {
        return eLayers[0].getVisibleLayer(i).importance;
    }

    // Retrieve predictions
    const IntBuffer &getPredictionCIs(
        int i // Index of input layer to get predictions for
    ) const {
        return dLayers[0][i].getHiddenCIs();
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
};
} // namespace aon
