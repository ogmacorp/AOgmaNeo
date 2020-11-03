// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Layer.h"
#include "Actor.h"

namespace aon {
// Type of hierarchy input layer
enum IOType {
    prediction = 0,
    action = 1
};

// A SPH
class Hierarchy {
public:
    struct IODesc {
        Int3 size;

        IOType type;
        
        int layerRadius; // Feed forward radius
        int actorRadius; // Actor radius

        int historyCapacity; // Actor history capacity

        IODesc()
        :
        size(4, 4, 16),
        type(prediction),
        layerRadius(2),
        actorRadius(2),
        historyCapacity(32)
        {}

        IODesc(
            const Int3 &size,
            IOType type,
            int layerRadius,
            int actorRadius,
            int historyCapacity
        )
        :
        size(size),
        type(type),
        layerRadius(layerRadius),
        actorRadius(actorRadius),
        historyCapacity(historyCapacity)
        {}
    };

    // Describes a layer for construction. For the first layer, the IODesc overrides the parameters that are the same name
    struct LayerDesc {
        Int3 hiddenSize; // Size of hidden layer

        int layerRadius; // Feed forward radius

        int ticksPerUpdate; // Number of ticks a layer takes to update (relative to previous layer)
        int temporalHorizon; // Temporal distance into the past addressed by the layer. Should be greater than or equal to ticksPerUpdate

        LayerDesc()
        :
        hiddenSize(4, 4, 16),
        layerRadius(2),
        ticksPerUpdate(2),
        temporalHorizon(2)
        {}

        LayerDesc(
            const Int3 &hiddenSize,
            int layerRadius,
            int ticksPerUpdate,
            int temporalHorizon
        )
        :
        hiddenSize(hiddenSize),
        layerRadius(layerRadius),
        ticksPerUpdate(ticksPerUpdate),
        temporalHorizon(temporalHorizon)
        {}
    };

private:
    // Layers
    Array<Layer> layers;
    Array<Ptr<Actor>> actors;

    // Histories
    Array<Array<CircleBuffer<IntBuffer>>> histories;
    Array<Array<CircleBuffer<IntBuffer>>> historiesPrev; // 1-step delay

    Array<IntBuffer> feedBackCIsPrev;

    // Per-layer values
    IntBuffer updates;

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
        float reward = 0.0f, // Reinforcement signal
        bool mimic = false // Whether to treat Actors like Predictors
    );

    // Serialization
    void write(
        StreamWriter &writer
    ) const;

    void read(
        StreamReader &reader
    );

    // Get the number of layers (scLayers)
    int getNumLayers() const {
        return layers.size();
    }

    // Retrieve predictions
    const IntBuffer &getPredictionCIs(
        int i // Index of input layer to get predictions for
    ) const {
        if (actors[i] != nullptr) // If is an action layer
            return actors[i]->getHiddenCIs();

        int predStartIndex = inputSizes.size() * histories[0][0].size();

        return layers[0].getVisibleLayer(predStartIndex + i * ticksPerUpdate[0] + 0).visibleCIs;
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
    Layer &getLayer(
        int l // Layer index
    ) {
        return layers[l];
    }

    // Retrieve a sparse coding layer, const version
    const Layer &getLayer(
        int l // Layer index
    ) const {
        return layers[l];
    }

    // Retrieve predictor layer(s)
    Array<Ptr<Actor>> &getActors() {
        return actors;
    }

    // Retrieve predictor layer(s), const version
    const Array<Ptr<Actor>> &getActors() const {
        return actors;
    }

    const Array<CircleBuffer<IntBuffer>> &getHistories(
        int l
    ) const {
        return histories[l];
    }
};
} // namespace aon
