// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
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
    eLayers.resize(layerDescs.size());
    dLayers.resize(layerDescs.size());
    errors.resize(layerDescs.size());

    ticks.resize(layerDescs.size(), 0);

    histories.resize(layerDescs.size());
    
    ticksPerUpdate.resize(layerDescs.size());

    // Default update state is no update
    updates.resize(layerDescs.size(), false);

    // Cache input sizes
    ioSizes.resize(ioDescs.size());
    ioTypes.resize(ioDescs.size());

    // Determine ticks per update, first layer is always 1
    for (int l = 0; l < layerDescs.size(); l++)
        ticksPerUpdate[l] = (l == 0 ? 1 : layerDescs[l].ticksPerUpdate);

    int numPredictions = 0;
    int numActions = 0;

    for (int i = 0; i < ioSizes.size(); i++) {
        ioSizes[i] = ioDescs[i].size;
        ioTypes[i] = static_cast<Byte>(ioDescs[i].type);

        if (ioDescs[i].type == prediction)
            numPredictions++;
        else if (ioDescs[i].type == action) {
            numPredictions++; // Actions also implicitly have a regular prediction for error purposes
            numActions++;
        }
    }

    // Iterate through layers
    for (int l = 0; l < layerDescs.size(); l++) {
        // Create sparse coder visible layer descriptors
        Array<Encoder::VisibleLayerDesc> eVisibleLayerDescs;

        // If first layer
        if (l == 0) {
            eVisibleLayerDescs.resize(ioSizes.size() * layerDescs[l].temporalHorizon);

            for (int i = 0; i < ioSizes.size(); i++) {
                for (int t = 0; t < layerDescs[l].temporalHorizon; t++) {
                    int index = t + layerDescs[l].temporalHorizon * i;

                    eVisibleLayerDescs[index].size = ioSizes[i];
                    eVisibleLayerDescs[index].radius = ioDescs[i].eRadius;
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

            dLayers[l].resize(numPredictions);
            aLayers.resize(numActions);

            iIndices.resize(ioSizes.size() * 2);
            dIndices = IntBuffer(ioSizes.size(), -1);

            // Create decoders and actors
            int dIndex = 0;

            for (int i = 0; i < ioSizes.size(); i++) {
                if (ioDescs[i].type == prediction || ioDescs[i].type == action) {
                    // Decoder visible layer descriptors
                    Array<Decoder::VisibleLayerDesc> dVisibleLayerDescs(l < eLayers.size() - 1 ? 2 : 1);

                    dVisibleLayerDescs[0].size = layerDescs[l].hiddenSize;
                    dVisibleLayerDescs[0].radius = ioDescs[i].dRadius;

                    if (l < eLayers.size() - 1)
                        dVisibleLayerDescs[1] = dVisibleLayerDescs[0];

                    dVisibleLayerDescs[0].hasBWeights = true;

                    dLayers[l][dIndex].initRandom(ioSizes[i], dVisibleLayerDescs);

                    iIndices[dIndex] = i;
                    dIndices[i] = dIndex;
                    dIndex++;
                }
            }

            dIndex = 0;

            for (int i = 0; i < ioSizes.size(); i++) {
                if (ioDescs[i].type == action) {
                    // Decoder visible layer descriptors
                    Array<Actor::VisibleLayerDesc> aVisibleLayerDescs(l < eLayers.size() - 1 ? 2 : 1);

                    aVisibleLayerDescs[0].size = layerDescs[l].hiddenSize;
                    aVisibleLayerDescs[0].radius = ioDescs[i].dRadius;

                    if (l < eLayers.size() - 1)
                        aVisibleLayerDescs[1] = aVisibleLayerDescs[0];

                    aLayers[dIndex].initRandom(ioSizes[i], ioDescs[i].historyCapacity, aVisibleLayerDescs);

                    iIndices[ioSizes.size() + dIndex] = i;
                    dIndices[i] = dIndex;
                    dIndex++;
                }
            }
        }
        else {
            eVisibleLayerDescs.resize(layerDescs[l].temporalHorizon);

            for (int t = 0; t < layerDescs[l].temporalHorizon; t++) {
                eVisibleLayerDescs[t].size = layerDescs[l - 1].hiddenSize;
                eVisibleLayerDescs[t].radius = layerDescs[l].eRadius;
            }

            histories[l].resize(1);

            int inSize = layerDescs[l - 1].hiddenSize.x * layerDescs[l - 1].hiddenSize.y;

            histories[l][0].resize(layerDescs[l].temporalHorizon);

            for (int t = 0; t < histories[l][0].size(); t++)
                histories[l][0][t] = IntBuffer(inSize, 0);

            dLayers[l].resize(layerDescs[l].ticksPerUpdate);

            // Decoder visible layer descriptors
            Array<Decoder::VisibleLayerDesc> dVisibleLayerDescs(l < eLayers.size() - 1 ? 2 : 1);

            dVisibleLayerDescs[0].size = layerDescs[l].hiddenSize;
            dVisibleLayerDescs[0].radius = layerDescs[l].dRadius;

            if (l < eLayers.size() - 1)
                dVisibleLayerDescs[1] = dVisibleLayerDescs[0];

            dVisibleLayerDescs[0].hasBWeights = true;

            // Create decoders
            for (int t = 0; t < dLayers[l].size(); t++)
                dLayers[l][t].initRandom(layerDescs[l - 1].hiddenSize, dVisibleLayerDescs);
        }
        
        // Create the sparse coding layer
        eLayers[l].initRandom(layerDescs[l].hiddenSize, eVisibleLayerDescs);

        errors[l] = FloatBuffer(eLayers[l].getHiddenCIs().size(), 0.0f);
    }
}

void Hierarchy::step(
    const Array<const IntBuffer*> &inputCIs,
    bool learnEnabled,
    float reward,
    bool mimic
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
    for (int l = 0; l < eLayers.size(); l++) {
        // If is time for layer to tick
        if (l == 0 || ticks[l] >= ticksPerUpdate[l]) {
            // Reset tick
            ticks[l] = 0;

            // Updated
            updates[l] = true;

            Array<const IntBuffer*> layerInputCIs(eLayers[l].getNumVisibleLayers());

            int index = 0;

            for (int i = 0; i < histories[l].size(); i++) {
                for (int t = 0; t < histories[l][i].size(); t++)
                    layerInputCIs[index++] = &histories[l][i][t];
            }

            if (learnEnabled) {
                errors[l].fill(0.0f);

                for (int d = 0; d < dLayers[l].size(); d++)
                    dLayers[l][d].generateErrors(&histories[l][l == 0 ? iIndices[d] : 0][l == 0 ? 0 : d], &errors[l], 0);

                // Rescale
                float scale = 1.0f / dLayers[l].size();

                for (int i = 0; i < errors[l].size(); i++)
                    errors[l][i] *= scale;
            }

            // Activate sparse coder
            eLayers[l].step(layerInputCIs, &errors[l], learnEnabled);

            // Add to next layer's history
            if (l < eLayers.size() - 1) {
                int lNext = l + 1;

                histories[lNext][0].pushFront();

                histories[lNext][0][0] = eLayers[l].getHiddenCIs();

                ticks[lNext]++;
            }
        }
    }

    // Backward
    for (int l = dLayers.size() - 1; l >= 0; l--) {
        if (updates[l]) {
            Array<const IntBuffer*> layerInputCIs(l < eLayers.size() - 1 ? 2 : 1);

            layerInputCIs[0] = &eLayers[l].getHiddenCIs();
            
            if (l < eLayers.size() - 1)
                layerInputCIs[1] = &dLayers[l + 1][ticksPerUpdate[l + 1] - 1 - ticks[l + 1]].getHiddenCIs();

            for (int d = 0; d < dLayers[l].size(); d++) {
                if (learnEnabled)
                    dLayers[l][d].learn(&histories[l][l == 0 ? iIndices[d] : 0][l == 0 ? 0 : d]);

                dLayers[l][d].activate(layerInputCIs);
            }

            if (l == 0) {
                for (int d = 0; d < aLayers.size(); d++)
                    aLayers[d].step(layerInputCIs, inputCIs[iIndices[d + ioSizes.size()]], reward, learnEnabled, mimic);
            }
        }
    }
}

int Hierarchy::size() const {
    int size = 4 * sizeof(int) + ioSizes.size() * sizeof(Int3) + updates.size() * sizeof(Byte) + 2 * ticks.size() * sizeof(int) + iIndices.size() * sizeof(int) + dIndices.size() * sizeof(int);

    for (int l = 0; l < eLayers.size(); l++) {
        size += sizeof(int);

        for (int i = 0; i < histories[l].size(); i++) {
            size += 2 * sizeof(int);

            for (int t = 0; t < histories[l][i].size(); t++)
                size += sizeof(int) + histories[l][i][t].size() * sizeof(int);
        }

        size += eLayers[l].size();

        for (int d = 0; d < dLayers[l].size(); d++)
            size += dLayers[l][d].size();
    }

    // Actors
    for (int d = 0; d < aLayers.size(); d++)
        size += aLayers[d].size();

    return size;
}

int Hierarchy::stateSize() const {
    int size = updates.size() * sizeof(Byte) + ticks.size() * sizeof(int);

    for (int l = 0; l < eLayers.size(); l++) {
        for (int i = 0; i < histories[l].size(); i++) {
            size += sizeof(int);

            for (int t = 0; t < histories[l][i].size(); t++)
                size += histories[l][i][t].size() * sizeof(int);
        }

        size += eLayers[l].stateSize();
        
        // Decoders
        for (int d = 0; d < dLayers[l].size(); d++)
            size += dLayers[l][d].stateSize();
    }

    // Actors
    for (int d = 0; d < aLayers.size(); d++)
        size += aLayers[d].stateSize();

    return size;
}

void Hierarchy::write(
    StreamWriter &writer
) const {
    int numLayers = eLayers.size();

    writer.write(reinterpret_cast<const void*>(&numLayers), sizeof(int));

    int numIO = ioSizes.size();

    writer.write(reinterpret_cast<const void*>(&numIO), sizeof(int));

    int numPredictions = dLayers[0].size();
    int numActions = aLayers.size();

    writer.write(reinterpret_cast<const void*>(&numPredictions), sizeof(int));
    writer.write(reinterpret_cast<const void*>(&numActions), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&ioSizes[0]), numIO * sizeof(Int3));
    writer.write(reinterpret_cast<const void*>(&ioTypes[0]), numIO * sizeof(Byte));

    writer.write(reinterpret_cast<const void*>(&updates[0]), updates.size() * sizeof(Byte));
    writer.write(reinterpret_cast<const void*>(&ticks[0]), ticks.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&ticksPerUpdate[0]), ticksPerUpdate.size() * sizeof(int));

    writer.write(reinterpret_cast<const void*>(&iIndices[0]), iIndices.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&dIndices[0]), dIndices.size() * sizeof(int));

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

        eLayers[l].write(writer);

        // Decoders
        for (int d = 0; d < dLayers[l].size(); d++)
            dLayers[l][d].write(writer);
    }
    
    // Actors
    for (int d = 0; d < aLayers.size(); d++)
        aLayers[d].write(writer);
}

void Hierarchy::read(
    StreamReader &reader
) {
    int numLayers;

    reader.read(reinterpret_cast<void*>(&numLayers), sizeof(int));

    int numIO;

    reader.read(reinterpret_cast<void*>(&numIO), sizeof(int));

    int numPredictions;
    int numActions;

    reader.read(reinterpret_cast<void*>(&numPredictions), sizeof(int));
    reader.read(reinterpret_cast<void*>(&numActions), sizeof(int));

    ioSizes.resize(numIO);
    ioTypes.resize(numIO);

    reader.read(reinterpret_cast<void*>(&ioSizes[0]), numIO * sizeof(Int3));
    reader.read(reinterpret_cast<void*>(&ioTypes[0]), numIO * sizeof(Byte));

    eLayers.resize(numLayers);
    dLayers.resize(numLayers);
    errors.resize(numLayers);

    histories.resize(numLayers);
    
    updates.resize(numLayers);
    ticks.resize(numLayers);
    ticksPerUpdate.resize(numLayers);

    reader.read(reinterpret_cast<void*>(&updates[0]), updates.size() * sizeof(Byte));
    reader.read(reinterpret_cast<void*>(&ticks[0]), ticks.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&ticksPerUpdate[0]), ticksPerUpdate.size() * sizeof(int));

    iIndices.resize(numIO * 2);
    dIndices.resize(numIO);

    reader.read(reinterpret_cast<void*>(&iIndices[0]), iIndices.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&dIndices[0]), dIndices.size() * sizeof(int));
    
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

        eLayers[l].read(reader);
        
        dLayers[l].resize(l == 0 ? numPredictions : ticksPerUpdate[l]);

        // Decoders
        for (int d = 0; d < dLayers[l].size(); d++)
            dLayers[l][d].read(reader);

        errors[l] = FloatBuffer(eLayers[l].getHiddenCIs().size(), 0.0f);
    }

    aLayers.resize(numActions);

    // Actors
    for (int d = 0; d < aLayers.size(); d++)
        aLayers[d].read(reader);
}

void Hierarchy::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&updates[0]), updates.size() * sizeof(Byte));
    writer.write(reinterpret_cast<const void*>(&ticks[0]), ticks.size() * sizeof(int));

    for (int l = 0; l < eLayers.size(); l++) {
        for (int i = 0; i < histories[l].size(); i++) {
            int historyStart = histories[l][i].start;

            writer.write(reinterpret_cast<const void*>(&historyStart), sizeof(int));

            for (int t = 0; t < histories[l][i].size(); t++)
                writer.write(reinterpret_cast<const void*>(&histories[l][i][t][0]), histories[l][i][t].size() * sizeof(int));
        }

        eLayers[l].writeState(writer);

        // Decoders
        for (int d = 0; d < dLayers[l].size(); d++)
            dLayers[l][d].writeState(writer);
    }

    for (int d = 0; d < aLayers.size(); d++)
        aLayers[d].writeState(writer);
}

void Hierarchy::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&updates[0]), updates.size() * sizeof(Byte));
    reader.read(reinterpret_cast<void*>(&ticks[0]), ticks.size() * sizeof(int));
    
    for (int l = 0; l < eLayers.size(); l++) {
        for (int i = 0; i < histories[l].size(); i++) {
            int historyStart;
            
            reader.read(reinterpret_cast<void*>(&historyStart), sizeof(int));

            histories[l][i].start = historyStart;

            for (int t = 0; t < histories[l][i].size(); t++)
                reader.read(reinterpret_cast<void*>(&histories[l][i][t][0]), histories[l][i][t].size() * sizeof(int));
        }

        eLayers[l].readState(reader);
        
        // Decoders
        for (int d = 0; d < dLayers[l].size(); d++)
            dLayers[l][d].readState(reader);
    }

    // Actors
    for (int d = 0; d < aLayers.size(); d++)
        aLayers[d].readState(reader);
}

void Hierarchy::clearState() {
    updates.fill(false);
    ticks.fill(0);

    for (int l = 0; l < eLayers.size(); l++) {
        for (int i = 0; i < histories[l].size(); i++) {
            for (int t = 0; t < histories[l][i].size(); t++)
                histories[l][i][t].fill(0);
        }

        eLayers[l].clearState();
        
        // Decoders
        for (int d = 0; d < dLayers[l].size(); d++)
            dLayers[l][d].clearState();
    }

    // Actors
    for (int d = 0; d < aLayers.size(); d++)
        aLayers[d].clearState();
}
