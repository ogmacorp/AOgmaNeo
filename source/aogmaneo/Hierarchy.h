// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "HiddenEncoder.h"
#include "ErrorEncoder.h"
#include "Decoder.h"
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

        int hRadius; // Hidden encoder radius
        int eRadius; // Error encoder radius
        int dRadius; // Decoder radius
        int bRadius; // Feed back radius

        int historyCapacity; // Actor history capacity

        IODesc()
        :
        size(4, 4, 16),
        type(prediction),
        hRadius(2),
        eRadius(2),
        dRadius(2),
        bRadius(2),
        historyCapacity(32)
        {}

        IODesc(
            const Int3 &size,
            IOType type,
            int hRadius,
            int eRadius,
            int dRadius,
            int bRadius,
            int historyCapacity
        )
        :
        size(size),
        type(type),
        hRadius(hRadius),
        eRadius(eRadius),
        dRadius(dRadius),
        bRadius(bRadius),
        historyCapacity(historyCapacity)
        {}
    };

    // Describes a layer for construction. For the first layer, the IODesc overrides the parameters that are the same name
    struct LayerDesc {
        Int3 hiddenSize; // Size of hidden layer
        Int3 errorSize; // Size of error layer

        int hRadius; // Feed forward hidden radius
        int eRadius; // Feed forward error radius
        int dRadius; // Prediction radius
        int bRadius; // Feed back radius

        int ticksPerUpdate; // Number of ticks a layer takes to update (relative to previous layer)
        int temporalHorizon; // Temporal distance into the past addressed by the layer. Should be greater than or equal to ticksPerUpdate

        LayerDesc()
        :
        hiddenSize(4, 4, 16),
        errorSize(4, 4, 16),
        hRadius(2),
        eRadius(2),
        dRadius(2),
        bRadius(2),
        ticksPerUpdate(2),
        temporalHorizon(2)
        {}

        LayerDesc(
            const Int3 &hiddenSize,
            const Int3 &errorSize,
            int hRadius,
            int eRadius,
            int dRadius,
            int bRadius,
            int ticksPerUpdate,
            int temporalHorizon
        )
        :
        hiddenSize(hiddenSize),
        errorSize(errorSize),
        hRadius(hRadius),
        eRadius(eRadius),
        dRadius(dRadius),
        bRadius(bRadius),
        ticksPerUpdate(ticksPerUpdate),
        temporalHorizon(temporalHorizon)
        {}
    };

    struct EncLayerPair {
        HiddenEncoder hidden;
        ErrorEncoder error;
    };

private:
    // Layers
    Array<EncLayerPair> encLayers;
    Array<Array<Array<Decoder>>> dLayers;
    Array<Ptr<Actor>> aLayers;
    Array<FloatBuffer> errors;

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
        float reward = 0.0f, // Reinforcement signal
        bool mimic = false // Whether to treat Actors like Decoders
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

    // Get the number of layers (encLayers)
    int getNumLayers() const {
        return encLayers.size();
    }

    // Retrieve predictions
    const IntBuffer &getPredictionCIs(
        int i // Index of input layer to get predictions for
    ) const {
        if (aLayers[i] != nullptr) // If is an action layer
            return aLayers[i]->getHiddenCIs();

        return dLayers[0][i][0].getHiddenCIs();
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
    EncLayerPair &getEncLayer(
        int l // Layer index
    ) {
        return encLayers[l];
    }

    // Retrieve a sparse coding layer, const version
    const EncLayerPair &getEncLayer(
        int l // Layer index
    ) const {
        return encLayers[l];
    }

    // Retrieve predictor layer(s)
    Array<Array<Decoder>> &getDLayers(
        int l // Layer index
    ) {
        return dLayers[l];
    }

    // Retrieve predictor layer(s), const version
    const Array<Array<Decoder>> &getDLayers(
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

    const Array<CircleBuffer<IntBuffer>> &getHistories(
        int l
    ) const {
        return histories[l];
    }
};
} // namespace aon
