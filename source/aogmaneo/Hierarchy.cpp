// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Hierarchy.h"

using namespace aon;

const Hierarchy &Hierarchy::operator=(
    const Hierarchy &other // Hierarchy to assign from
) {
    // Layers
    scLayers = other.scLayers;
    pLayers = other.pLayers;

    updates = other.updates;
    ticks = other.ticks;
    ticksPerUpdate = other.ticksPerUpdate;
    inputSizes = other.inputSizes;
    histories = other.histories;

    ipLayers.resize(inputSizes.size());
    aLayers.resize(inputSizes.size());

    for (int i = 0; i < inputSizes.size(); i++) {
        if (other.ipLayers[i] != nullptr) {
            ipLayers[i].make();

            (*ipLayers[i]) = (*other.ipLayers[i]);
        }
        else
            ipLayers[i] = nullptr;

        if (other.aLayers[i] != nullptr) {
            aLayers[i].make();

            (*aLayers[i]) = (*other.aLayers[i]);
        }
        else
            aLayers[i] = nullptr;
    }

    return *this;
}

void Hierarchy::initRandom(
    const Array<Int3> &inputSizes, // Sizes of input layers
    const Array<InputType> &inputTypes, // Types of input layers (same size as inputSizes)
    const Array<LayerDesc> &layerDescs // Descriptors for layers
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
        Array<SparseCoder::VisibleLayerDesc> scVisibleLayerDescs;

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
                    histories[l][i][t] = IntBuffer(inSize, 0);
            }

            // Predictor visible layer descriptors
            Array<Predictor::VisibleLayerDesc> pVisibleLayerDescs(l < scLayers.size() - 1 ? 2 : 1);

            pVisibleLayerDescs[0].size = layerDescs[l].hiddenSize;
            pVisibleLayerDescs[0].radius = layerDescs[l].pRadius;

            if (l < scLayers.size() - 1) {
                pVisibleLayerDescs[1].size = layerDescs[l + 1].hiddenSize;
                pVisibleLayerDescs[1].radius = layerDescs[l].pRadius;
            }

            pLayers[l].initRandom(layerDescs[l].hiddenSize, pVisibleLayerDescs);

            // Actors and input predictor layers
            ipLayers.resize(inputSizes.size());
            aLayers.resize(inputSizes.size());

            // Actor visible layer descriptors
            Array<Predictor::VisibleLayerDesc> ipVisibleLayerDescs(1);

            ipVisibleLayerDescs[0].size = layerDescs[l].hiddenSize;
            ipVisibleLayerDescs[0].radius = layerDescs[l].pRadius;

            // Actor visible layer descriptors
            Array<Actor::VisibleLayerDesc> aVisibleLayerDescs(1);

            aVisibleLayerDescs[0].size = layerDescs[l].hiddenSize;
            aVisibleLayerDescs[0].radius = layerDescs[l].aRadius;

            // Create predictors
            for (int i = 0; i < inputSizes.size(); i++) {
                if (inputTypes[i] == InputType::prediction) {
                    ipLayers[i].make();

                    ipLayers[i]->initRandom(inputSizes[i], ipVisibleLayerDescs);
                }

                if (inputTypes[i] == InputType::action) {
                    aLayers[i].make();

                    aLayers[i]->initRandom(inputSizes[i], layerDescs[l].historyCapacity, aVisibleLayerDescs);
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
                histories[l][0][t] = IntBuffer(inSize, 0);

            // Predictor visible layer descriptors
            Array<Predictor::VisibleLayerDesc> pVisibleLayerDescs(l < scLayers.size() - 1 ? 2 : 1);

            pVisibleLayerDescs[0].size = layerDescs[l].hiddenSize;
            pVisibleLayerDescs[0].radius = layerDescs[l].pRadius;

            if (l < scLayers.size() - 1) {
                pVisibleLayerDescs[1].size = layerDescs[l + 1].hiddenSize;
                pVisibleLayerDescs[1].radius = layerDescs[l].pRadius;
            }

            pLayers[l].initRandom(layerDescs[l].hiddenSize, pVisibleLayerDescs);
        }
        
        // Create the sparse coding layer
        scLayers[l].initRandom(layerDescs[l].hiddenSize, scVisibleLayerDescs);
    }
}

// Simulation step/tick
void Hierarchy::step(
    const Array<const IntBuffer*> &inputCs,
    bool learnEnabled,
    float reward,
    bool mimic
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

            Array<const IntBuffer*> layerInputCs(histories[l].size() * histories[l][0].size());

            int index = 0;

            for (int i = 0; i < histories[l].size(); i++) {
                for (int t = 0; t < histories[l][i].size(); t++)
                    layerInputCs[index++] = &histories[l][i][t];
            }

            // Activate sparse coder
            scLayers[l].step(layerInputCs, &pLayers[l].getHiddenActivations(), learnEnabled);

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
            Array<const IntBuffer*> feedBackCs(l < scLayers.size() - 1 ? 2 : 1);

            feedBackCs[0] = &scLayers[l].getHiddenCs();

            if (l < scLayers.size() - 1)
                feedBackCs[1] = &pLayers[l + 1].getHiddenCs();

            if (learnEnabled)
                pLayers[l].learn(&scLayers[l].getHiddenCs());

            pLayers[l].activate(feedBackCs);

            if (l == 0) {
                for (int i = 0; i < inputSizes.size(); i++) {
                    if (ipLayers[i] != nullptr) {
                        if (learnEnabled)
                            ipLayers[i]->learn(&histories[l][i][0]);

                        ipLayers[i]->activate(feedBackCs);
                    }

                    if (aLayers[i] != nullptr)
                        aLayers[i]->step(feedBackCs, &histories[l][i][0], reward, learnEnabled, mimic);
                }
            }
        }
    }
}

void Hierarchy::write(
    StreamWriter &writer
) const {
    int numLayers = scLayers.size();

    writer.write(reinterpret_cast<const void*>(&numLayers), sizeof(int));

    int numInputs = inputSizes.size();

    writer.write(reinterpret_cast<const void*>(&numInputs), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&inputSizes[0]), numInputs * sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&updates[0]), updates.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&ticks[0]), ticks.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&ticksPerUpdate[0]), ticksPerUpdate.size() * sizeof(int));

    for (int l = 0; l < numLayers; l++) {
        int numInputs = histories[l].size();

        writer.write(reinterpret_cast<const void*>(&numInputs), sizeof(int));

        for (int i = 0; i < histories[l].size(); i++) {
            int historySize = histories[l][i].size();

            writer.write(reinterpret_cast<const void*>(&historySize), sizeof(int));

            int historyStart = histories[l][i].start;

            writer.write(reinterpret_cast<const void*>(&historyStart), sizeof(int));

            for (int t = 0; t < histories[l][i].size(); t++) {
                int bufferSize = histories[l][i][t].size();

                writer.write(reinterpret_cast<const void*>(&bufferSize), sizeof(int));

                writer.write(reinterpret_cast<const void*>(&histories[l][i][t][0]), histories[l][i][t].size() * sizeof(int));
            }
        }

        scLayers[l].write(writer);
        pLayers[l].write(writer);
    }

    // Actors
    for (int i = 0; i < inputSizes.size(); i++) {
        {
            unsigned char exists = ipLayers[i] != nullptr;

            writer.write(reinterpret_cast<const void*>(&exists), sizeof(unsigned char));

            if (exists)
                ipLayers[i]->write(writer);
        }

        {
            unsigned char exists = aLayers[i] != nullptr;

            writer.write(reinterpret_cast<const void*>(&exists), sizeof(unsigned char));

            if (exists)
                aLayers[i]->write(writer);
        }
    }
}

void Hierarchy::read(
    StreamReader &reader
) {
    int numLayers;

    reader.read(reinterpret_cast<void*>(&numLayers), sizeof(int));

    int numInputs;

    reader.read(reinterpret_cast<void*>(&numInputs), sizeof(int));

    inputSizes.resize(numInputs);

    reader.read(reinterpret_cast<void*>(&inputSizes[0]), numInputs * sizeof(Int3));

    scLayers.resize(numLayers);
    pLayers.resize(numLayers);

    histories.resize(numLayers);
    
    updates.resize(numLayers);
    ticks.resize(numLayers);
    ticksPerUpdate.resize(numLayers);

    reader.read(reinterpret_cast<void*>(&updates[0]), updates.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&ticks[0]), ticks.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&ticksPerUpdate[0]), ticksPerUpdate.size() * sizeof(int));
    
    for (int l = 0; l < numLayers; l++) {
        int numInputs;
        
        reader.read(reinterpret_cast<void*>(&numInputs), sizeof(int));

        histories[l].resize(numInputs);

        for (int i = 0; i < histories[l].size(); i++) {
            int historySize;

            reader.read(reinterpret_cast<void*>(&historySize), sizeof(int));

            int historyStart;
            
            reader.read(reinterpret_cast<void*>(&historyStart), sizeof(int));

            histories[l][i].resize(historySize);
            histories[l][i].start = historyStart;

            for (int t = 0; t < histories[l][i].size(); t++) {
                int bufferSize;

                reader.read(reinterpret_cast<void*>(&bufferSize), sizeof(int));

                histories[l][i][t].resize(bufferSize);

                reader.read(reinterpret_cast<void*>(&histories[l][i][t][0]), histories[l][i][t].size() * sizeof(int));
            }
        }

        scLayers[l].read(reader);
        pLayers[l].read(reader);
    }

    // Actors
    aLayers.resize(inputSizes.size());

    for (int i = 0; i < aLayers.size(); i++) {
        {
            unsigned char exists;

            reader.read(reinterpret_cast<void*>(&exists), sizeof(unsigned char));

            if (exists) {
                ipLayers[i].make();
                ipLayers[i]->read(reader);
            }
        }

        {
            unsigned char exists;

            reader.read(reinterpret_cast<void*>(&exists), sizeof(unsigned char));

            if (exists) {
                aLayers[i].make();
                aLayers[i]->read(reader);
            }
        }
    }
}
