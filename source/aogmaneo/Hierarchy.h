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
enum InputType {
    none = 0,
    prediction = 1,
    action = 2
};

// State of hierarchy
struct State {
    Array<ByteBuffer> hiddenCs;
    Array<ByteBuffer> hiddenCsPrev;
    Array<Array<Array<ByteBuffer>>> predInputCsPrev;
    Array<Array<ByteBuffer>> predHiddenCs;

    Array<Array<CircleBuffer<ByteBuffer>>> histories;

    Array<char> updates;
    Array<int> ticks;
};

// A SPH
class Hierarchy {
public:
    // Describes a layer for construction
    struct LayerDesc {
        Int3 hiddenSize; // Size of hidden layer

        int ffRadius; // Feed forward radius
        int pRadius; // Prediction radius
        int aRadius; // Actor radius

        int ticksPerUpdate; // Number of ticks a layer takes to update (relative to previous layer)

        int temporalHorizon; // Temporal distance into a the past addressed by the layer. Should be greater than or equal to ticksPerUpdate

        // History capacity of actors
        int historyCapacity;

        LayerDesc()
        :
        hiddenSize(4, 4, 16),
        ffRadius(2),
        pRadius(2),
        aRadius(2),
        ticksPerUpdate(2),
        temporalHorizon(4),
        historyCapacity(32)
        {}
    };
private:
    // Layers
    Array<SparseCoder> scLayers;
    Array<Array<Ptr<Predictor>>> pLayers;
    Array<Ptr<Actor>> aLayers;

    // Histories
    Array<Array<CircleBuffer<ByteBuffer>>> histories;

    // Per-layer values
    Array<char> updates;

    Array<int> ticks;
    Array<int> ticksPerUpdate;

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
        const Array<Int3> &inputSizes, // Sizes of input layers
        const Array<InputType> &inputTypes, // Types of input layers (same size as inputSizes)
        const Array<LayerDesc> &layerDescs // Descriptors for layers
    );

    // Simulation step/tick
    void step(
        const Array<const ByteBuffer*> &inputCs, // Inputs to remember
        bool learnEnabled = true, // Whether learning is enabled
        float reward = 0.0f, // Optional reward for actor layers
        bool mimic = false
    );

    // State get
    void getState(
        State &state
    ) const;

    // State set
    void setState(
        const State &state
    );

    // Get the number of layers (scLayers)
    int getNumLayers() const {
        return scLayers.size();
    }

    // Retrieve predictions
    const ByteBuffer &getPredictionCs(
        int i // Index of input layer to get predictions for
    ) const {
        if (aLayers[i] != nullptr) // If is an action layer
            return aLayers[i]->getHiddenCs();

        return pLayers[0][i]->getHiddenCs();
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
};
} // namespace ogmaneo
