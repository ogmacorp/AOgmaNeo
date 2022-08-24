// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
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
        int aRadius; // Actor radius

        int historyCapacity;

        IODesc(
            const Int3 &size = Int3(4, 4, 16),
            IOType type = prediction,
            int eRadius = 2,
            int aRadius = 2,
            int historyCapacity = 64
        )
        :
        size(size),
        type(type),
        eRadius(eRadius),
        aRadius(aRadius),
        historyCapacity(historyCapacity)
        {}
    };

    // Describes a layer for construction. For the first layer, the IODesc overrides the parameters that are the same name
    struct LayerDesc {
        Int3 hiddenSize; // Size of hidden layer

        int eRadius; // Encoder radius
        int pRadius; // Predictor radius

        int ticksPerUpdate; // Number of ticks a layer takes to update (relative to previous layer)
        int temporalHorizon; // Temporal distance into the past addressed by the layer. Should be greater than or equal to ticksPerUpdate

        LayerDesc(
            const Int3 &hiddenSize = Int3(4, 4, 16),
            int eRadius = 2,
            int pRadius = 2,
            int ticksPerUpdate = 2,
            int temporalHorizon = 4
        )
        :
        hiddenSize(hiddenSize),
        eRadius(eRadius),
        pRadius(pRadius),
        ticksPerUpdate(ticksPerUpdate),
        temporalHorizon(temporalHorizon)
        {}
    };

private:
    // Layers
    Array<Layer> layers;
    Array<Actor> actors;

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
        float reward = 0.0f, // Reward
        bool mimic = false // Mimicry mode - treat actors as regular decoders
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

    // Get the number of layers (layers)
    int getNumLayers() const {
        return layers.size();
    }

    bool ioLayerExists(
        int i
    ) const {
        return dIndices[i] != -1;
    }

    // Importance control
    void setImportance(
        int i,
        float importance
    ) {
        for (int t = 0; t < histories[0][i].size(); t++)
            layers[0].enc.getVisibleLayer(i * histories[0][i].size() + t).importance = importance;
    }

    // Importance control
    float getImportance(
        int i
    ) const {
        return layers[0].enc.getVisibleLayer(i * histories[0][i].size()).importance;
    }

    // Retrieve predictions
    const IntBuffer &getPredictionCIs(
        int i
    ) const {
        if (ioTypes[i] == action)
            return actors[dIndices[i]].getHiddenCIs();

        return layers[0].getPredCIs(i * histories[0][i].size());
    }

    // Whether this layer received on update this timestep
    bool getUpdate(
        int l
    ) const {
        return updates[l];
    }

    // Get current layer ticks, relative to previous layer
    int getTicks(
        int l
    ) const {
        return ticks[l];
    }

    // Get layer ticks per update, relative to previous layer
    int getTicksPerUpdate(
        int l
    ) const {
        return ticksPerUpdate[l];
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
    Layer &getLayer(
        int l
    ) {
        return layers[l];
    }

    // Retrieve a sparse coding layer, const version
    const Layer &getLayer(
        int l
    ) const {
        return layers[l];
    }

    // Retrieve actor layer(s)
    Array<Actor> &getActors() {
        return actors;
    }

    const Array<Actor> &getActors() const {
        return actors;
    }

    Actor &getActor(
        int i
    ) {
        return actors[dIndices[i]];
    }

    const Actor &getActor(
        int i
    ) const {
        return actors[dIndices[i]];
    }

    const IntBuffer &getIIndices() const {
        return iIndices;
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
