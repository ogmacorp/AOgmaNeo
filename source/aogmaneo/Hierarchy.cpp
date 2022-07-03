// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Hierarchy.h"

using namespace aon;

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

    // Cache input sizes
    ioSizes.resize(ioDescs.size());

    // Determine ticks per update, first layer is always 1
    for (int l = 0; l < layerDescs.size(); l++)
        ticksPerUpdate[l] = (l == 0 ? 1 : layerDescs[l].ticksPerUpdate); // First layer always 1

    // Iterate through layers
    for (int l = 0; l < layerDescs.size(); l++) {
        // Create sparse coder visible layer descriptors
        Array<Layer::VisibleLayerDesc> eVisibleLayerDescs;

        // If first layer
        if (l == 0) {
            eVisibleLayerDescs.resize(ioSizes.size() * layerDescs[l].temporalHorizon);

            for (int i = 0; i < ioSizes.size(); i++) {
                for (int t = 0; t < layerDescs[l].temporalHorizon; t++) {
                    int index = t + layerDescs[l].temporalHorizon * i;

                    eVisibleLayerDescs[index].size = ioSizes[i];
                    eVisibleLayerDescs[index].radius = ioDescs[i].radius;
                }
            }
            
            // Initialize history buffers
            histories[l].resize(ioSizes.size());

            for (int i = 0; i < histories[l].size(); i++) {
                int inSize = ioSizes[i].x * ioSizes[i].y;

                histories[l][i].resize(layerDescs[l].temporalHorizon);
                
                for (int t = 0; t < histories[l][i].size(); t++)
                    histories[l][i][t] = IntBuffer(inSize, 0);
            }
        }
        else {
            eVisibleLayerDescs.resize(layerDescs[l].temporalHorizon);

            for (int t = 0; t < layerDescs[l].temporalHorizon; t++) {
                eVisibleLayerDescs[t].size = layerDescs[l - 1].hiddenSize;
                eVisibleLayerDescs[t].radius = layerDescs[l].radius;
            }

            histories[l].resize(1);

            int inSize = layerDescs[l - 1].hiddenSize.x * layerDescs[l - 1].hiddenSize.y;

            histories[l][0].resize(layerDescs[l].temporalHorizon);

            for (int t = 0; t < histories[l][0].size(); t++)
                histories[l][0][t] = IntBuffer(inSize, 0);
        }
        
        // Create the sparse coding layer
        layers[l].initRandom(layerDescs[l].hiddenSize, eVisibleLayerDescs);
    }
}

// Simulation step/tick
void Hierarchy::step(
    const Array<const IntBuffer*> &inputCIs,
    const IntBuffer* topProgCIs,
    bool learnEnabled
) {
    // First tick is always 0
    ticks[0] = 0;

    // Add input to first layer history   
    for (int i = 0; i < ioSizes.size(); i++) {
        histories[0][i].pushFront();

        histories[0][i][0] = *inputCIs[i];
    }

    // Set all updates to no update, will be set to true if an update occurred later
    updates.fill(false);

    // Forward
    for (int l = 0; l < layers.size(); l++) {
        // If is time for layer to tick
        if (l == 0 || ticks[l] >= ticksPerUpdate[l]) {
            // Reset tick
            ticks[l] = 0;

            // Updated
            updates[l] = true;

            Array<const IntBuffer*> upCIs(layers[l].getNumVisibleLayers());

            int index = 0;

            for (int i = 0; i < histories[l].size(); i++) {
                for (int t = 0; t < histories[l][i].size(); t++)
                    upCIs[index++] = &histories[l][i][t];
            }

            layers[l].stepUp(upCIs, learnEnabled);

            // Add to next layer's history
            if (l < layers.size() - 1) {
                int lNext = l + 1;

                histories[lNext][0].pushFront();

                histories[lNext][0][0] = layers[l].getHiddenCIs();

                ticks[lNext]++;
            }
        }
    }

    // Backward infer
    for (int l = layers.size() - 1; l >= 0; l--) {
        if (l < layers.size() - 1)
            layers[l + 1].reconstruct(ticksPerUpdate[l + 1] - 1 - ticks[l + 1]);

        layers[l].stepDown(l < layers.size() - 1 ? &layers[l + 1].getReconstruction(ticksPerUpdate[l + 1] - 1 - ticks[l + 1]) : topProgCIs);
    }

    // Reconstruct predictions
    for (int i = 0; i < ioSizes.size(); i++)
        layers[0].reconstruct(i * histories[0][i].size());
}

int Hierarchy::size() const {
    int size = 2 * sizeof(int) + ioSizes.size() * sizeof(Int3) + updates.size() * sizeof(Byte) + 2 * ticks.size() * sizeof(int);

    for (int l = 0; l < layers.size(); l++) {
        size += sizeof(int);

        for (int i = 0; i < histories[l].size(); i++) {
            size += 2 * sizeof(int);

            for (int t = 0; t < histories[l][i].size(); t++)
                size += sizeof(int) + histories[l][i][t].size() * sizeof(int);
        }

        size += layers[l].size();
    }

    return size;
}

int Hierarchy::stateSize() const {
    int size = updates.size() * sizeof(Byte) + ticks.size() * sizeof(int);

    for (int l = 0; l < layers.size(); l++) {
        for (int i = 0; i < histories[l].size(); i++) {
            size += sizeof(int);

            for (int t = 0; t < histories[l][i].size(); t++)
                size += histories[l][i][t].size() * sizeof(int);
        }

        size += layers[l].stateSize();
    }

    return size;
}

void Hierarchy::write(
    StreamWriter &writer
) const {
    int numLayers = layers.size();

    writer.write(reinterpret_cast<const void*>(&numLayers), sizeof(int));

    int numIO = ioSizes.size();

    writer.write(reinterpret_cast<const void*>(&numIO), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&ioSizes[0]), numIO * sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&updates[0]), updates.size() * sizeof(Byte));
    writer.write(reinterpret_cast<const void*>(&ticks[0]), ticks.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&ticksPerUpdate[0]), ticksPerUpdate.size() * sizeof(int));

    for (int l = 0; l < numLayers; l++) {
        int numLayerInputs = histories[l].size();

        writer.write(reinterpret_cast<const void*>(&numLayerInputs), sizeof(int));

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
    }
}

void Hierarchy::read(
    StreamReader &reader
) {
    int numLayers;

    reader.read(reinterpret_cast<void*>(&numLayers), sizeof(int));

    int numIO;

    reader.read(reinterpret_cast<void*>(&numIO), sizeof(int));

    ioSizes.resize(numIO);

    reader.read(reinterpret_cast<void*>(&ioSizes[0]), numIO * sizeof(Int3));

    layers.resize(numLayers);

    histories.resize(numLayers);
    
    updates.resize(numLayers);
    ticks.resize(numLayers);
    ticksPerUpdate.resize(numLayers);

    reader.read(reinterpret_cast<void*>(&updates[0]), updates.size() * sizeof(Byte));
    reader.read(reinterpret_cast<void*>(&ticks[0]), ticks.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&ticksPerUpdate[0]), ticksPerUpdate.size() * sizeof(int));

    for (int l = 0; l < numLayers; l++) {
        int numLayerInputs;
        
        reader.read(reinterpret_cast<void*>(&numLayerInputs), sizeof(int));

        histories[l].resize(numLayerInputs);

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
    }
}

void Hierarchy::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&updates[0]), updates.size() * sizeof(Byte));
    writer.write(reinterpret_cast<const void*>(&ticks[0]), ticks.size() * sizeof(int));

    for (int l = 0; l < layers.size(); l++) {
        for (int i = 0; i < histories[l].size(); i++) {
            int historyStart = histories[l][i].start;

            writer.write(reinterpret_cast<const void*>(&historyStart), sizeof(int));

            for (int t = 0; t < histories[l][i].size(); t++)
                writer.write(reinterpret_cast<const void*>(&histories[l][i][t][0]), histories[l][i][t].size() * sizeof(int));
        }

        layers[l].writeState(writer);
    }
}

void Hierarchy::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&updates[0]), updates.size() * sizeof(Byte));
    reader.read(reinterpret_cast<void*>(&ticks[0]), ticks.size() * sizeof(int));
    
    for (int l = 0; l < layers.size(); l++) {
        for (int i = 0; i < histories[l].size(); i++) {
            int historyStart;
            
            reader.read(reinterpret_cast<void*>(&historyStart), sizeof(int));

            histories[l][i].start = historyStart;

            for (int t = 0; t < histories[l][i].size(); t++)
                reader.read(reinterpret_cast<void*>(&histories[l][i][t][0]), histories[l][i][t].size() * sizeof(int));
        }

        layers[l].readState(reader);
    }
}
