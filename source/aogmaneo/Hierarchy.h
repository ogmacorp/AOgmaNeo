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

        int historyCapacity;

        IODesc()
        :
        size(4, 4, 16),
        type(prediction),
        eRadius(2),
        dRadius(2),
        historyCapacity(8)
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
        int dRadius; // Decoder radius

        int historyCapacity;

        int ticksPerUpdate; // Number of ticks a layer takes to update (relative to previous layer)
        int temporalHorizon; // Temporal distance into the past addressed by the layer. Should be greater than or equal to ticksPerUpdate

        LayerDesc()
        :
        hiddenSize(4, 4, 16),
        eRadius(2),
        dRadius(2),
        historyCapacity(8),
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

    // For mapping first layer decoders
    IntBuffer iIndices;
    IntBuffer dIndices;

    // Histories
    Array<Array<CircleBuffer<IntBuffer>>> histories;

    // Per-layer values
    ByteBuffer updates;

    IntBuffer ticks;
    IntBuffer ticksPerUpdate;

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

    // Get state of highest layer (less verbose when dealing with goal-driven learning)
    const IntBuffer &getTopHiddenCIs() const {
        return eLayers[eLayers.size() - 1].getHiddenCIs();
    }

    // Get size of highest layer (less verbose when dealing with goal-driven learning)
    const Int3 &getTopHiddenSize() const {
        return eLayers[eLayers.size() - 1].getHiddenSize();
    }

    bool getTopUpdate() const {
        return updates[updates.size() - 1];
    }

    // Importance control
    void setImportance(
        int i,
        float importance
    ) {
        for (int t = 0; t < histories[0][i].size(); t++)
            eLayers[0].getVisibleLayer(i * histories[0][i].size() + t).importance = importance;
    }

    // Importance control
    float getImportance(
        int i
    ) const {
        return eLayers[0].getVisibleLayer(i * histories[0][i].size()).importance;
    }

    // Retrieve predictions
    const IntBuffer &getPredictionCIs(
        int i // Index of input layer to get predictions for
    ) const {
        return dLayers[0][dIndices[i]].getHiddenCIs();
    }

    // Whether this layer received on update this timestep
    bool getUpdate(
        int l // Layer index
    ) const {
        return updates[l];
    }

    // Get current layer ticks, relative to previous layer
    int getTicks(
        int l // Layer Index
    ) const {
        return ticks[l];
    }

    // Get layer ticks per update, relative to previous layer
    int getTicksPerUpdate(
        int l // Layer Index
    ) const {
        return ticksPerUpdate[l];
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

    const IntBuffer &getDIndices() const {
        return dIndices;
    }

    const Array<CircleBuffer<IntBuffer>> &getHistories(
        int l
    ) const {
        return histories[l];
    }
};
} // namespace aon
