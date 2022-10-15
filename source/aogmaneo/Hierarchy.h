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

        IODesc(
            const Int3 &size = Int3(4, 4, 16),
            IOType type = prediction,
            int eRadius = 2,
            int dRadius = 2,
            int historyCapacity = 64
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

        int ticksPerUpdate; // Number of ticks a layer takes to update (relative to previous layer)
        int temporalHorizon; // Temporal distance into the past addressed by the layer. Should be greater than or equal to ticksPerUpdate

        LayerDesc(
            const Int3 &hiddenSize = Int3(4, 4, 16),
            int eRadius = 2,
            int dRadius = 2,
            int ticksPerUpdate = 2,
            int temporalHorizon = 4
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
    Array<Array<Decoder>> dLayers;
    Array<Actor> aLayers;
    Array<IntBuffer> hiddenCIsPrev;

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
        float reward = 0.0f // Reward
    );

    void clearState();

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
        int i
    ) const {
        if (ioTypes[i] == action)
            return aLayers[dIndices[i]].getHiddenCIs();

        return dLayers[0][dIndices[i]].getHiddenCIs();
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

    const Array<CircleBuffer<IntBuffer>> &getHistories(
        int l
    ) const {
        return histories[l];
    }
};
} // namespace aon
