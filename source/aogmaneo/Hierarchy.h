// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "SparseCoder.h"
#include "Predictor.h"
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

        int ffRadius; // Feed forward radius
        int pRadius; // Prediction radius
        int aRadius; // Actor radius

        int historyCapacity; // Actor history capacity

        IOType type;

        IODesc()
        :
        size(4, 4, 16),
        type(none),
        ffRadius(2),
        pRadius(2),
        aRadius(2),
        historyCapacity(32)
        {}

        IODesc(
            const Int3 &size,
            IOType type,
            int ffRadius,
            int pRadius,
            int aRadius,
            int historyCapacity
        )
        :
        size(size),
        type(type),
        ffRadius(ffRadius),
        pRadius(pRadius),
        aRadius(aRadius),
        historyCapacity(historyCapacity)
        {}
    };

    // Describes a layer for construction. For the first layer, the IODesc overrides the parameters that are the same name
    struct LayerDesc {
        Int3 hiddenSize; // Size of hidden layer

        int ffRadius; // Feed forward radius
        int pRadius; // Prediction radius

        int ticksPerUpdate; // Number of ticks a layer takes to update (relative to previous layer)
        int temporalHorizon; // Temporal distance into the past addressed by the layer. Should be greater than or equal to ticksPerUpdate

        LayerDesc()
        :
        hiddenSize(4, 4, 16),
        ffRadius(2),
        pRadius(2),
        ticksPerUpdate(2),
        temporalHorizon(2)
        {}

        LayerDesc(
            const Int3 &hiddenSize,
            int ffRadius,
            int pRadius,
            int ticksPerUpdate,
            int temporalHorizon
        )
        :
        hiddenSize(hiddenSize),
        ffRadius(ffRadius),
        pRadius(pRadius),
        ticksPerUpdate(ticksPerUpdate),
        temporalHorizon(temporalHorizon)
        {}
    };

private:
    // Layers
    Array<SparseCoder> scLayers;
    Array<Array<Ptr<Predictor>>> pLayers;
    Array<Ptr<Actor>> aLayers;

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
        bool learnEnabled = true, // Whether learning is enabled
        float reward = 0.0f // Reinforcement signal
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

    // Get the number of layers (scLayers)
    int getNumLayers() const {
        return scLayers.size();
    }

    // Retrieve predictions
    const IntBuffer &getPredictionCIs(
        int i // Index of input layer to get predictions for
    ) const {
        if (aLayers[i] != nullptr) // If is an action layer
            return aLayers[i]->getHiddenCIs();

        return pLayers[0][i]->getHiddenCIs();
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
    SparseCoder &getSCLayer(
        int l // Layer index
    ) {
        return scLayers[l];
    }

    // Retrieve a sparse coding layer, const version
    const SparseCoder &getSCLayer(
        int l // Layer index
    ) const {
        return scLayers[l];
    }

    // Retrieve predictor layer(s)
    Array<Ptr<Predictor>> &getPLayers(
        int l // Layer index
    ) {
        return pLayers[l];
    }

    // Retrieve predictor layer(s), const version
    const Array<Ptr<Predictor>> &getPLayers(
        int l // Layer index
    ) const {
        return pLayers[l];
    }

    // Retrieve predictor layer(s)
    Array<Ptr<Actor>> &getALayers() {
        return aLayers;
    }

    // Retrieve predictor layer(s), const version
    const Array<Ptr<Actor>> &getALayers() const {
        return aLayers;
    }

    const Array<CircleBuffer<IntBuffer>> &getHistories(
        int l
    ) const {
        return histories[l];
    }
};
} // namespace aon
