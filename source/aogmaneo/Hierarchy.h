// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Encoder.h"
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

        int ffRadius;
        int fbRadius;

        int historyCapacity;

        IODesc(
            const Int3 &size = Int3(4, 4, 16),
            IOType type = prediction,
            int ffRadius = 2,
            int fbRadius = 2,
            int historyCapacity = 64
        )
        :
        size(size),
        type(type),
        ffRadius(ffRadius),
        fbRadius(fbRadius),
        historyCapacity(historyCapacity)
        {}
    };

    // Describes a layer for construction. For the first layer, the IODesc overrides the parameters that are the same name
    struct LayerDesc {
        Int3 hiddenSize; // Size of hidden layer

        int ffRadius;
        int fbRadius;

        int ticksPerUpdate; // Number of ticks a layer takes to update (relative to previous layer)
        int temporalHorizon; // Temporal distance into the past addressed by the layer. Should be greater than or equal to ticksPerUpdate

        LayerDesc(
            const Int3 &hiddenSize = Int3(4, 4, 16),
            int ffRadius = 2,
            int fbRadius = 2,
            int ticksPerUpdate = 2,
            int temporalHorizon = 2
        )
        :
        hiddenSize(hiddenSize),
        ffRadius(ffRadius),
        fbRadius(fbRadius),
        ticksPerUpdate(ticksPerUpdate),
        temporalHorizon(temporalHorizon)
        {}
    };

private:
    // Layers
    Array<Encoder> eLayers;
    Array<Actor> aLayers;

    // For mapping first layer decoders
    IntBuffer iIndices;
    IntBuffer aIndices;

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
        bool mimic = false // Mimicry mode
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

    bool aLayerExists(
        int i
    ) const {
        return aIndices[i] != -1;
    }

    // Importance control
    void setInputImportance(
        int i,
        float importance
    ) {
        for (int t = 0; t < histories[0][i].size(); t++)
            eLayers[0].getVisibleLayer(i * histories[0][i].size() + t).importance = importance;
    }

    float getInputImportance(
        int i
    ) const {
        return eLayers[0].getVisibleLayer(i * histories[0][i].size()).importance;
    }

    // Retrieve predictions
    const IntBuffer &getPredictionCIs(
        int i
    ) const {
        if (ioTypes[i] == action)
            return aLayers[aIndices[i]].getHiddenCIs();

        return eLayers[0].getReconCIs(histories[0].size() * histories[0][0].size() + i);
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

    // Number of IO layers
    int getNumIO() const {
        return ioSizes.size();
    }

    // Get input/output sizes
    const Int3 &getIOSize(
        int i
    ) const {
        return ioSizes[i];
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

    // Retrieve actor layer(s)
    Array<Actor> &getALayers() {
        return aLayers;
    }

    const Array<Actor> &getALayers() const {
        return aLayers;
    }

    Actor &getALayer(
        int i
    ) {
        return aLayers[aIndices[i]];
    }

    const Actor &getALayer(
        int i
    ) const {
        return aLayers[aIndices[i]];
    }

    const IntBuffer &getIIndices() const {
        return iIndices;
    }

    const IntBuffer &getDIndices() const {
        return aIndices;
    }

    const Array<CircleBuffer<IntBuffer>> &getHistories(
        int l
    ) const {
        return histories[l];
    }
};
} // namespace aon
