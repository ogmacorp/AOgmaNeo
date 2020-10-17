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
    prediction = 0,
    action = 1
};

// A SPH
class Hierarchy {
public:
    // Describes a layer for construction
    struct LayerDesc {
        Int3 hiddenSize; // Size of hidden layer

        int ffRadius; // Feed forward radius
        int rRadius; // Recurrent radius
        int pRadius; // Prediction radius
        int aRadius; // Actor radius

        int historyCapacity;

        LayerDesc()
        :
        hiddenSize(4, 4, 16),
        ffRadius(2),
        rRadius(2),
        pRadius(2),
        aRadius(2),
        historyCapacity(32)
        {}
    };

private:
    // Layers
    Array<SparseCoder> scLayers;
    Array<Predictor> pLayers;
    Array<Ptr<Actor>> aLayers;
    Array<Ptr<Predictor>> ipLayers;

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
        const Array<const IntBuffer*> &inputCs, // Inputs to remember
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
        return scLayers.size();
    }

    // Retrieve predictions
    const IntBuffer &getPredictionCs(
        int i // Index of input layer to get predictions for
    ) const {
        if (aLayers[i] != nullptr) // If is an action layer
            return aLayers[i]->getHiddenCs();

        return ipLayers[i]->getHiddenCs();
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
    Predictor &getPLayer(
        int l // Layer index
    ) {
        return pLayers[l];
    }

    // Retrieve predictor layer(s), const version
    const Predictor &getPLayer(
        int l // Layer index
    ) const {
        return pLayers[l];
    }

    // Retrieve predictor layer(s)
    Array<Ptr<Predictor>> &getIPLayers() {
        return ipLayers;
    }

    // Retrieve predictor layer(s), const version
    const Array<Ptr<Predictor>> &getIPLayers() const {
        return ipLayers;
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
} // namespace aon
