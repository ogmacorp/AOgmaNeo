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

namespace ogmaneo {
// Type of hierarchy input layer
enum InputType {
    none = 0,
    prediction = 1,
    action = 2
};

// Describes a layer for construction
struct HierarchyLayerDesc {
    Int3 hiddenSize; // Size of hidden layer

    int ffRadius; // Feed forward radius
    int pRadius; // Prediction radius
    int aRadius; // Actor radius

    int ticksPerUpdate; // Number of ticks a layer takes to update (relative to previous layer)
    int temporalHorizon; // Temporal distance into a the past addressed by the layer. Should be greater than or equal to ticksPerUpdate

    int historyCapacity;

    HierarchyLayerDesc()
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

// A SPH
template <typename T>
class Hierarchy {
private:
    // Layers
    Array<SparseCoder<T>> scLayers;
    Array<Array<Ptr<Predictor<T>>>> pLayers;
    Array<Ptr<Actor<T>>> aLayers;

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
    ) {
        // Layers
        scLayers = other.scLayers;

        updates = other.updates;
        ticks = other.ticks;
        ticksPerUpdate = other.ticksPerUpdate;
        inputSizes = other.inputSizes;
        histories = other.histories;

        pLayers.resize(other.pLayers.size());
        
        for (int l = 0; l < scLayers.size(); l++) {
            pLayers[l].resize(other.pLayers[l].size());

            for (int v = 0; v < pLayers[l].size(); v++) {
                if (other.pLayers[l][v] != nullptr) {
                    pLayers[l][v].make();

                    (*pLayers[l][v]) = (*other.pLayers[l][v]);
                }
                else
                    pLayers[l][v] = nullptr;
            }
        }

        aLayers.resize(inputSizes.size());
    
        for (int v = 0; v < aLayers.size(); v++) {
            if (other.aLayers[v] != nullptr) {
                aLayers[v].make();

                (*aLayers[v]) = (*other.aLayers[v]);
            }
            else
                aLayers[v] = nullptr;
        }

        return *this;
    }
    
    // Create a randomly initialized hierarchy
    void initRandom(
        const Array<Int3> &inputSizes, // Sizes of input layers
        const Array<InputType> &inputTypes, // Types of input layers (same size as inputSizes)
        const Array<HierarchyLayerDesc> &layerDescs // Descriptors for layers
    ) {
        // Create layers
        scLayers.resize(layerDescs.size());
        pLayers.resize(layerDescs.size());

        ticks.resize(layerDescs.size(), 0);

        histories.resize(layerDescs.size());
        
        ticksPerUpdate.resize(layerDescs.size());

        // Default update state is no update
        updates.resize(layerDescs.size(), false);

        // Cache input sizes
        this->inputSizes = inputSizes;

        // Determine ticks per update, first layer is always 1
        for (int l = 0; l < layerDescs.size(); l++)
            ticksPerUpdate[l] = l == 0 ? 1 : layerDescs[l].ticksPerUpdate; // First layer always 1

        // Iterate through layers
        for (int l = 0; l < layerDescs.size(); l++) {
            // Create sparse coder visible layer descriptors
            Array<SparseCoderVisibleLayerDesc> scVisibleLayerDescs;

            // If first layer
            if (l == 0) {
                scVisibleLayerDescs.resize(inputSizes.size() * layerDescs[l].temporalHorizon);

                for (int i = 0; i < inputSizes.size(); i++) {
                    for (int t = 0; t < layerDescs[l].temporalHorizon; t++) {
                        int index = t + layerDescs[l].temporalHorizon * i;

                        scVisibleLayerDescs[index].size = inputSizes[i];
                        scVisibleLayerDescs[index].radius = layerDescs[l].ffRadius;
                    }
                }
                
                // Initialize history buffers
                histories[l].resize(inputSizes.size());

                for (int i = 0; i < histories[l].size(); i++) {
                    int inSize = inputSizes[i].x * inputSizes[i].y;

                    histories[l][i].resize(layerDescs[l].temporalHorizon);
                    
                    for (int t = 0; t < histories[l][i].size(); t++)
                        histories[l][i][t] = ByteBuffer(inSize, 0);
                }

                // Predictors
                pLayers[l].resize(inputSizes.size());

                // Predictor visible layer descriptors
                Array<PredictorVisibleLayerDesc> pVisibleLayerDescs(l < scLayers.size() - 1 ? 2 : 1);

                pVisibleLayerDescs[0].size = layerDescs[l].hiddenSize;
                pVisibleLayerDescs[0].radius = layerDescs[l].pRadius;

                if (l < scLayers.size() - 1)
                    pVisibleLayerDescs[1] = pVisibleLayerDescs[0];

                // Actors
                aLayers.resize(inputSizes.size());

                // Actor visible layer descriptors
                Array<ActorVisibleLayerDesc> aVisibleLayerDescs(l < scLayers.size() - 1 ? 2 : 1);

                aVisibleLayerDescs[0].size = layerDescs[l].hiddenSize;
                aVisibleLayerDescs[0].radius = layerDescs[l].aRadius;

                if (l < scLayers.size() - 1)
                    aVisibleLayerDescs[1] = aVisibleLayerDescs[0];

                // Create predictors
                for (int p = 0; p < pLayers[l].size(); p++) {
                    if (inputTypes[p] == InputType::prediction) {
                        pLayers[l][p].make();

                        pLayers[l][p]->initRandom(inputSizes[p], pVisibleLayerDescs);
                    }
                    else if (inputTypes[p] == InputType::action) {
                        aLayers[p].make();

                        aLayers[p]->initRandom(inputSizes[p], layerDescs[l].historyCapacity, aVisibleLayerDescs);
                    }
                }
            }
            else {
                scVisibleLayerDescs.resize(layerDescs[l].temporalHorizon);

                for (int t = 0; t < layerDescs[l].temporalHorizon; t++) {
                    scVisibleLayerDescs[t].size = layerDescs[l - 1].hiddenSize;
                    scVisibleLayerDescs[t].radius = layerDescs[l].ffRadius;
                }

                histories[l].resize(1);

                int inSize = layerDescs[l - 1].hiddenSize.x * layerDescs[l - 1].hiddenSize.y;

                histories[l][0].resize(layerDescs[l].temporalHorizon);

                for (int t = 0; t < histories[l][0].size(); t++)
                    histories[l][0][t] = ByteBuffer(inSize, 0);

                pLayers[l].resize(layerDescs[l].ticksPerUpdate);

                // Predictor visible layer descriptors
                Array<PredictorVisibleLayerDesc> pVisibleLayerDescs(l < scLayers.size() - 1 ? 2 : 1);

                pVisibleLayerDescs[0].size = layerDescs[l].hiddenSize;
                pVisibleLayerDescs[0].radius = layerDescs[l].pRadius;

                if (l < scLayers.size() - 1)
                    pVisibleLayerDescs[1] = pVisibleLayerDescs[0];

                // Create actors
                for (int p = 0; p < pLayers[l].size(); p++) {
                    pLayers[l][p].make();

                    pLayers[l][p]->initRandom(layerDescs[l - 1].hiddenSize, pVisibleLayerDescs);
                }
            }
            
            // Create the sparse coding layer
            scLayers[l].initRandom(layerDescs[l].hiddenSize, scVisibleLayerDescs);
        }
    }

    // Simulation step/tick
    void step(
        const Array<const ByteBuffer*> &inputCs, // Inputs to remember
        bool learnEnabled = true, // Whether learning is enabled
        float reward = 0.0f, // Reinforcement signal
        bool mimic = false // For imitation learning
    ) {
        // First tick is always 0
        ticks[0] = 0;

        // Add input to first layer history   
        for (int i = 0; i < inputSizes.size(); i++) {
            histories[0][i].pushFront();

            histories[0][i][0] = *inputCs[i];
        }

        // Set all updates to no update, will be set to true if an update occurred later
        for (int i = 0; i < updates.size(); i++)
            updates[i] = false;

        // Forward
        for (int l = 0; l < scLayers.size(); l++) {
            // If is time for layer to tick
            if (l == 0 || ticks[l] >= ticksPerUpdate[l]) {
                // Reset tick
                ticks[l] = 0;

                // Updated
                updates[l] = true;

                Array<const ByteBuffer*> layerInputCs(histories[l].size() * histories[l][0].size());

                int index = 0;

                for (int i = 0; i < histories[l].size(); i++) {
                    for (int t = 0; t < histories[l][i].size(); t++)
                        layerInputCs[index++] = &histories[l][i][t];
                }

                // Activate sparse coder
                scLayers[l].step(layerInputCs, learnEnabled);

                // Add to next layer's history
                if (l < scLayers.size() - 1) {
                    int lNext = l + 1;

                    histories[lNext][0].pushFront();

                    histories[lNext][0][0] = scLayers[l].getHiddenCs();

                    ticks[lNext]++;
                }
            }
        }

        // Backward
        for (int l = scLayers.size() - 1; l >= 0; l--) {
            if (updates[l]) {
                // Feed back is current layer state and next higher layer prediction
                Array<const ByteBuffer*> feedBackCs(l < scLayers.size() - 1 ? 2 : 1);

                feedBackCs[0] = &scLayers[l].getHiddenCs();

                if (l < scLayers.size() - 1)
                    feedBackCs[1] = &pLayers[l + 1][ticksPerUpdate[l + 1] - 1 - ticks[l + 1]]->getHiddenCs();

                // Step actor layers
                for (int p = 0; p < pLayers[l].size(); p++) {
                    if (pLayers[l][p] != nullptr) {
                        if (learnEnabled)
                            pLayers[l][p]->learn(l == 0 ? inputCs[p] : &histories[l][0][p]);

                        pLayers[l][p]->activate(feedBackCs);
                    }
                }

                if (l == 0) {
                    // Step actors
                    for (int p = 0; p < aLayers.size(); p++) {
                        if (aLayers[p] != nullptr)
                            aLayers[p]->step(feedBackCs, inputCs[p], reward, learnEnabled, mimic);
                    }
                }
            }
        }
    }

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
    SparseCoder<T> &getSCLayer(
        int l // Layer index
    ) {
        return scLayers[l];
    }

    // Retrieve a sparse coding layer, const version
    const SparseCoder<T> &getSCLayer(
        int l // Layer index
    ) const {
        return scLayers[l];
    }

    // Retrieve predictor layer(s)
    Array<Ptr<Predictor<T>>> &getPLayers(
        int l // Layer index
    ) {
        return pLayers[l];
    }

    // Retrieve predictor layer(s), const version
    const Array<Ptr<Predictor<T>>> &getPLayers(
        int l // Layer index
    ) const {
        return pLayers[l];
    }

    // Retrieve predictor layer(s)
    Array<Ptr<Actor<T>>> &getALayers() {
        return aLayers;
    }

    // Retrieve predictor layer(s), const version
    const Array<Ptr<Actor<T>>> &getALayers() const {
        return aLayers;
    }
};
} // namespace ogmaneo
