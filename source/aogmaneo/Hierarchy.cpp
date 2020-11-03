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
    layers = other.layers;

    updates = other.updates;
    ticks = other.ticks;
    ticksPerUpdate = other.ticksPerUpdate;
    inputSizes = other.inputSizes;
    histories = other.histories;

    actors.resize(inputSizes.size());

    for (int v = 0; v < actors.size(); v++) {
        if (other.actors[v] != nullptr) {
            actors[v].make();

            (*actors[v]) = (*other.actors[v]);
        }
        else
            actors[v] = nullptr;
    }

    return *this;
}

void Hierarchy::initRandom(
    const Array<IODesc> &ioDescs,
    const Array<LayerDesc> &layerDescs
) {
    // Create layers
    layers.resize(layerDescs.size());

    ticks.resize(layerDescs.size(), 0);

    histories.resize(layerDescs.size());
    
    ticksPerUpdate.resize(layerDescs.size());

    // Default update state is no update
    updates.resize(layerDescs.size(), false);

    feedBackCIsPrev.resize(layerDescs.size());

    // Cache input sizes
    inputSizes.resize(ioDescs.size());

    for (int i = 0; i < inputSizes.size(); i++)
        inputSizes[i] = ioDescs[i].size;

    // Determine ticks per update, first layer is always 1
    for (int l = 0; l < layerDescs.size(); l++)
        ticksPerUpdate[l] = l == 0 ? 1 : layerDescs[l].ticksPerUpdate; // First layer always 1

    // Iterate through layers
    for (int l = 0; l < layerDescs.size(); l++) {
        // Create sparse coder visible layer descriptors
        Array<Layer::VisibleLayerDesc> visibleLayerDescs;

        // If first layer
        if (l == 0) {
            visibleLayerDescs.resize(inputSizes.size() * layerDescs[l].temporalHorizon + inputSizes.size() * layerDescs[l].ticksPerUpdate + (l < layerDescs.size() - 1 ? 1 : 0));

            int index = 0;

            for (int i = 0; i < inputSizes.size(); i++) {
                for (int t = 0; t < layerDescs[l].temporalHorizon; t++) {
                    visibleLayerDescs[index].size = inputSizes[i];
                    visibleLayerDescs[index].radius = ioDescs[i].layerRadius;

                    index++;
                }
            }
            
            for (int i = 0; i < inputSizes.size(); i++) {
                for (int t = 0; t < layerDescs[l].ticksPerUpdate; t++) {
                    visibleLayerDescs[index].size = inputSizes[i];
                    visibleLayerDescs[index].radius = ioDescs[i].layerRadius;

                    index++;
                }
            }

            // Feed back
            if (l < layerDescs.size() - 1) {
                visibleLayerDescs[index].size = layerDescs[l].hiddenSize;
                visibleLayerDescs[index].radius = layerDescs[l].layerRadius;
            }

            // Initialize history buffers
            histories[l].resize(inputSizes.size());

            for (int i = 0; i < histories[l].size(); i++) {
                int inSize = inputSizes[i].x * inputSizes[i].y;

                histories[l][i].resize(layerDescs[l].temporalHorizon);
                
                for (int t = 0; t < histories[l][i].size(); t++)
                    histories[l][i][t] = IntBuffer(inSize, 0);
            }

            actors.resize(inputSizes.size());

            // Create predictors
            for (int i = 0; i < inputSizes.size(); i++) {
                if (ioDescs[i].type == IOType::action) {
                    // Actor visible layer descriptors
                    Array<Actor::VisibleLayerDesc> aVisibleLayerDescs(l < layers.size() - 1 ? 2 : 1);

                    aVisibleLayerDescs[0].size = layerDescs[l].hiddenSize;
                    aVisibleLayerDescs[0].radius = ioDescs[i].actorRadius;

                    if (l < layers.size() - 1)
                        aVisibleLayerDescs[1] = aVisibleLayerDescs[0];

                    actors[i].make();

                    actors[i]->initRandom(inputSizes[i], ioDescs[i].historyCapacity, aVisibleLayerDescs);
                }
            }
        }
        else {
            visibleLayerDescs.resize(layerDescs[l].temporalHorizon + layerDescs[l].ticksPerUpdate + (l < layerDescs.size() - 1 ? 1 : 0));

            int index = 0;

            for (int t = 0; t < layerDescs[l].temporalHorizon; t++) {
                visibleLayerDescs[index].size = layerDescs[l - 1].hiddenSize;
                visibleLayerDescs[index].radius = layerDescs[l].layerRadius;

                index++;
            }

            for (int t = 0; t < layerDescs[l].ticksPerUpdate; t++) {
                visibleLayerDescs[index].size = layerDescs[l - 1].hiddenSize;
                visibleLayerDescs[index].radius = layerDescs[l].layerRadius;

                index++;
            }

            // Feed back
            if (l < layerDescs.size() - 1) {
                visibleLayerDescs[index].size = layerDescs[l].hiddenSize;
                visibleLayerDescs[index].radius = layerDescs[l].layerRadius;
            }

            histories[l].resize(1);

            int inSize = layerDescs[l - 1].hiddenSize.x * layerDescs[l - 1].hiddenSize.y;

            histories[l][0].resize(layerDescs[l].temporalHorizon);

            for (int t = 0; t < histories[l][0].size(); t++)
                histories[l][0][t] = IntBuffer(inSize, 0);
        }
        
        // Create the sparse coding layer
        layers[l].initRandom(layerDescs[l].hiddenSize, visibleLayerDescs);

        feedBackCIsPrev[l] = layers[l].getHiddenCIs();
    }

    historiesPrev = histories;
}

void Hierarchy::step(
    const Array<const IntBuffer*> &inputCIs,
    bool learnEnabled,
    float reward,
    bool mimic
) {
    // Backup
    historiesPrev = histories;

    // First tick is always 0
    ticks[0] = 0;

    // Add input to first layer history   
    for (int i = 0; i < inputSizes.size(); i++) {
        histories[0][i].pushFront();

        histories[0][i][0] = *inputCIs[i];
    }

    // Set all updates to no update, will be set to true if an update occurred later
    for (int i = 0; i < updates.size(); i++)
        updates[i] = false;

    // Forward
    for (int l = 0; l < layers.size(); l++) {
        // If is time for layer to tick
        if (l == 0 || ticks[l] >= ticksPerUpdate[l]) {
            // Reset tick
            ticks[l] = 0;

            // Updated
            updates[l] = true;

            Array<const IntBuffer*> layerInputCIs(layers[l].getNumVisibleLayers(), nullptr);

            // Fill
            int index = 0;

            for (int i = 0; i < histories[l].size(); i++) {
                for (int t = 0; t < histories[l][i].size(); t++)
                    layerInputCIs[index++] = &histories[l][i][t];
            }

            // Activate sparse coder
            layers[l].activate(layerInputCIs, false);

            // Add to next layer's history
            if (l < layers.size() - 1) {
                int lNext = l + 1;

                histories[lNext][0].pushFront();

                histories[lNext][0][0] = layers[l].getHiddenCIs();

                ticks[lNext]++;
            }
        }
    }

    // Backward
    for (int l = layers.size() - 1; l >= 0; l--) {
        if (updates[l]) {
            Array<const IntBuffer*> layerInputCIs(layers[l].getNumVisibleLayers(), nullptr);

            // --- Learning phase ---

            // Fill
            int index = 0;

            for (int i = 0; i < historiesPrev[l].size(); i++) {
                for (int t = 0; t < historiesPrev[l][i].size(); t++)
                    layerInputCIs[index++] = &historiesPrev[l][i][t];
            }

            // Targets
            for (int i = 0; i < histories[l].size(); i++) {
                for (int t = 0; t < ticksPerUpdate[l]; t++)
                    layerInputCIs[index++] = &histories[l][i][t];
            }

            // Add feedback if available
            if (l < layers.size() - 1) {
                int predStartIndex = histories[l + 1][0].size();

                layerInputCIs[index] = &feedBackCIsPrev[l];
            }

            layers[l].activate(layerInputCIs, false);
            layers[l].learn(layerInputCIs);

            // --- Prediction phase ---
            
            // Fill
            index = 0;

            for (int i = 0; i < historiesPrev[l].size(); i++) {
                for (int t = 0; t < historiesPrev[l][i].size(); t++)
                    layerInputCIs[index++] = &histories[l][i][t];
            }

            // Targets
            for (int i = 0; i < histories[l].size(); i++) {
                for (int t = 0; t < ticksPerUpdate[l]; t++)
                    layerInputCIs[index++] = nullptr;
            }

            // Add feedback if available
            if (l < layers.size() - 1) {
                int predStartIndex = histories[l + 1][0].size();

                layerInputCIs[index] = &layers[l + 1].getVisibleLayer(predStartIndex + ticksPerUpdate[l + 1] - 1 - ticks[l + 1]).visibleCIs;
            }

            layers[l].activate(layerInputCIs, true);

            feedBackCIsPrev[l] = *layerInputCIs[index];
        }
    }

    Array<const IntBuffer*> feedBackCIs(layers.size() > 1 ? 2 : 1);

    feedBackCIs[0] = &layers[0].getHiddenCIs();
    
    if (layers.size() > 1)
        feedBackCIs[1] = &feedBackCIsPrev[0];

    // Step actors
    for (int p = 0; p < actors.size(); p++) {
        if (actors[p] != nullptr)
            actors[p]->step(feedBackCIs, &histories[0][p][0], reward, learnEnabled, mimic);
    }
}

void Hierarchy::write(
    StreamWriter &writer
) const {
    int numLayers = layers.size();

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

        layers[l].write(writer);

        writer.write(reinterpret_cast<const void*>(&feedBackCIsPrev[l]), feedBackCIsPrev[l].size() * sizeof(int));
    }

    // Actors
    for (int v = 0; v < actors.size(); v++) {
        unsigned char exists = actors[v] != nullptr;

        writer.write(reinterpret_cast<const void*>(&exists), sizeof(unsigned char));

        if (exists)
            actors[v]->write(writer);
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

    layers.resize(numLayers);

    histories.resize(numLayers);

    feedBackCIsPrev.resize(numLayers);
    
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

        layers[l].read(reader);

        feedBackCIsPrev[l].resize(layers[l].getHiddenCIs().size());

        reader.read(reinterpret_cast<void*>(&feedBackCIsPrev[l]), feedBackCIsPrev[l].size() * sizeof(int));
    }

    // Actors
    actors.resize(inputSizes.size());

    for (int v = 0; v < actors.size(); v++) {
        unsigned char exists;

        reader.read(reinterpret_cast<void*>(&exists), sizeof(unsigned char));

        if (exists) {
            actors[v].make();
            actors[v]->read(reader);
        }
    }

    historiesPrev = histories;
}
