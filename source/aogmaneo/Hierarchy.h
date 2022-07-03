// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Layer.h"

namespace aon {
// A SPH
class Hierarchy {
public:
    struct IODesc {
        Int3 size;

        int radius; // Layer radius

        IODesc()
        :
        size(4, 4, 16),
        radius(2)
        {}

        IODesc(
            const Int3 &size,
            int radius
        )
        :
        size(size),
        radius(radius)
        {}
    };

    // Describes a layer for construction. For the first layer, the IODesc overrides the parameters that are the same name
    struct LayerDesc {
        Int3 hiddenSize; // Size of hidden layer

        int radius; // Layer radius

        int ticksPerUpdate; // Number of ticks a layer takes to update (relative to previous layer)
        int temporalHorizon; // Temporal distance into the past addressed by the layer. Should be greater than or equal to ticksPerUpdate

        LayerDesc()
        :
        hiddenSize(4, 4, 16),
        radius(2),
        ticksPerUpdate(2),
        temporalHorizon(2)
        {}

        LayerDesc(
            const Int3 &hiddenSize,
            int radius,
            int ticksPerUpdate,
            int temporalHorizon
        )
        :
        hiddenSize(hiddenSize),
        radius(radius),
        ticksPerUpdate(ticksPerUpdate),
        temporalHorizon(temporalHorizon)
        {}
    };

private:
    // Layers
    Array<Layer> layers;

    // Histories
    Array<Array<CircleBuffer<IntBuffer>>> histories;

    // Per-layer values
    ByteBuffer updates;

    IntBuffer ticks;
    IntBuffer ticksPerUpdate;

    // Input dimensions
    Array<Int3> ioSizes;

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

    // Get the number of layers (layers)
    int getNumLayers() const {
        return layers.size();
    }

    // Get state of highest layer (less verbose when dealing with program-driven learning)
    const IntBuffer &getTopHiddenCIs() const {
        return layers[layers.size() - 1].getHiddenCIs();
    }

    // Get size of highest layer (less verbose when dealing with program-driven learning)
    const Int3 &getTopHiddenSize() const {
        return layers[layers.size() - 1].getHiddenSize();
    }

    bool getTopUpdate() const {
        return updates[updates.size() - 1];
    }

    // Importance control
    void setInputImportance(
        int i,
        float importance
    ) {
        for (int t = 0; t < histories[0][i].size(); t++)
            layers[0].getVisibleLayer(i * histories[0][i].size() + t).importance = importance;
    }

    // Importance control
    float getInputImportance(
        int i
    ) const {
        return layers[0].getVisibleLayer(i * histories[0][i].size()).importance;
    }

    // Retrieve predictions
    const IntBuffer &getPredictionCIs(
        int i // Index of input layer to get predictions for
    ) const {
        return layers[0].getReconstruction(i * histories[0][i].size());
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
    const Array<Int3> &getIOSizes() const {
        return ioSizes;
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

    const Array<CircleBuffer<IntBuffer>> &getHistories(
        int l
    ) const {
        return histories[l];
    }
};
} // namespace aon
