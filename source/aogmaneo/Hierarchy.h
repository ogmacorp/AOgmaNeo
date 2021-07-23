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

        int eRadius; // Feed forward radius
        int dRadius; // Feed back radius

        int historyCapacity;

        IODesc()
        :
        size(4, 4, 16),
        type(none),
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

        int eRadius; // Feed forward radius
        int dRadius; // Prediction radius

        int ticksPerUpdate; // Number of ticks a layer takes to update (relative to previous layer)
        int temporalHorizon; // Temporal distance into the past addressed by the layer. Should be greater than or equal to ticksPerUpdate

        LayerDesc()
        :
        hiddenSize(4, 4, 16),
        eRadius(2),
        dRadius(2),
        ticksPerUpdate(2),
        temporalHorizon(2)
        {}

        LayerDesc(
            const Int3 &hiddenSize,
            int eRadius,
            int dRadius,
            int ticksPerUpdate,
            int temporalHorizon
        )
        :
        hiddenSize(hiddenSize),
        eRadius(eRadius),
        dRadius(dRadius),
        ticksPerUpdate(ticksPerUpdate),
        temporalHorizon(temporalHorizon)
        {}
    };

private:
    // Layers
    Array<Encoder> eLayers;
    Array<Array<Ptr<Decoder>>> dLayers;
    Array<Ptr<Actor>> aLayers;

    // Histories
    Array<Array<CircleBuffer<ByteBuffer>>> histories;

    // Per-layer values
    ByteBuffer updates;

    ByteBuffer ticks;
    ByteBuffer ticksPerUpdate;

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
        const Array<const ByteBuffer*> &inputCIs, // Inputs to remember
        bool learnEnabled = true, // Whether learning is enabled
        float reward = 0.0f, // Reinforcement signal
        bool mimic = false // Imitation learning
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
        for (int t = 0; t < histories[0][i].size(); t++)
            eLayers[0].getVisibleLayer(i * histories[0][i].size() + t).importance = importance;
    }

    float getImportance(
        int i
    ) const {
        return eLayers[0].getVisibleLayer(i * histories[0][i].size()).importance;
    }

    // Retrieve predictions
    const ByteBuffer &getPredictionCIs(
        int i // Index of input layer to get predictions for
    ) const {
        if (aLayers[i] != nullptr) // If is an action layer
            return aLayers[i]->getHiddenCIs();

        return dLayers[0][i]->getHiddenCIs();
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

    Array<Ptr<Decoder>> &getDLayers(
        int l // Layer index
    ) {
        return dLayers[l];
    }

    const Array<Ptr<Decoder>> &getDLayers(
        int l // Layer index
    ) const {
        return dLayers[l];
    }

    // Retrieve predictor layer(s)
    Array<Ptr<Actor>> &getALayers() {
        return aLayers;
    }

    // Retrieve predictor layer(s), const version
    const Array<Ptr<Actor>> &getALayers() const {
        return aLayers;
    }

    const Array<CircleBuffer<ByteBuffer>> &getHistories(
        int l
    ) const {
        return histories[l];
    }
};
} // namespace aon
