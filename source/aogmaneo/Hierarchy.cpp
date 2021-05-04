// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
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
    encLayers = other.encLayers;
    dLayers = other.dLayers;

    updates = other.updates;
    ticks = other.ticks;
    ticksPerUpdate = other.ticksPerUpdate;
    inputSizes = other.inputSizes;
    histories = other.histories;
    errors = other.errors;

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

void Hierarchy::initRandom(
    const Array<IODesc> &ioDescs,
    const Array<LayerDesc> &layerDescs
) {
    // Create layers
    encLayers.resize(layerDescs.size());
    dLayers.resize(layerDescs.size());
    errors.resize(layerDescs.size());

    ticks.resize(layerDescs.size(), 0);

    histories.resize(layerDescs.size());
    
    ticksPerUpdate.resize(layerDescs.size());

    // Default update state is no update
    updates.resize(layerDescs.size(), false);

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
        Array<HiddenEncoder::VisibleLayerDesc> hVisibleLayerDescs;
        Array<ErrorEncoder::VisibleLayerDesc> eVisibleLayerDescs;

        // If first layer
        if (l == 0) {
            hVisibleLayerDescs.resize(inputSizes.size() * layerDescs[l].temporalHorizon);
            eVisibleLayerDescs.resize(hVisibleLayerDescs.size());

            for (int i = 0; i < inputSizes.size(); i++) {
                for (int t = 0; t < layerDescs[l].temporalHorizon; t++) {
                    int index = t + layerDescs[l].temporalHorizon * i;

                    hVisibleLayerDescs[index].size = inputSizes[i];
                    hVisibleLayerDescs[index].radius = ioDescs[i].hRadius;

                    eVisibleLayerDescs[index].size = inputSizes[i];
                    eVisibleLayerDescs[index].radius = ioDescs[i].eRadius;
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

            dLayers[l].resize(inputSizes.size());
            aLayers.resize(inputSizes.size());

            // Create predictors
            for (int i = 0; i < inputSizes.size(); i++) {
                // Decoder visible layer descriptors
                Array<Decoder::VisibleLayerDesc> dVisibleLayerDescs(l < encLayers.size() - 1 ? 2 : 1);

                dVisibleLayerDescs[0].size = layerDescs[l].errorSize;
                dVisibleLayerDescs[0].radius = ioDescs[i].dRadius;

                if (l < encLayers.size() - 1) {
                    dVisibleLayerDescs[1].size = layerDescs[l].hiddenSize;
                    dVisibleLayerDescs[1].radius = ioDescs[i].dRadius;
                }

                dLayers[l][i].resize(1);
                dLayers[l][i][0].initRandom(inputSizes[i], dVisibleLayerDescs);

                if (ioDescs[i].type == IOType::action) {
                    // Actor visible layer descriptors
                    Array<Actor::VisibleLayerDesc> aVisibleLayerDescs(l < encLayers.size() - 1 ? 2 : 1);

                    aVisibleLayerDescs[0].size = layerDescs[l].errorSize;
                    aVisibleLayerDescs[0].radius = ioDescs[i].dRadius;

                    if (l < encLayers.size() - 1) {
                        aVisibleLayerDescs[1].size = layerDescs[l].hiddenSize;
                        aVisibleLayerDescs[1].radius = ioDescs[i].dRadius;
                    }

                    aLayers[i].make();

                    aLayers[i]->initRandom(inputSizes[i], aVisibleLayerDescs);
                }
            }
        }
        else {
            hVisibleLayerDescs.resize(layerDescs[l].temporalHorizon);
            eVisibleLayerDescs.resize(layerDescs[l].temporalHorizon);

            for (int t = 0; t < layerDescs[l].temporalHorizon; t++) {
                hVisibleLayerDescs[t].size = layerDescs[l - 1].hiddenSize;
                hVisibleLayerDescs[t].radius = layerDescs[l].hRadius;

                eVisibleLayerDescs[t].size = layerDescs[l - 1].hiddenSize;
                eVisibleLayerDescs[t].radius = layerDescs[l].eRadius;
            }

            histories[l].resize(1);

            int inSize = layerDescs[l - 1].hiddenSize.x * layerDescs[l - 1].hiddenSize.y;

            histories[l][0].resize(layerDescs[l].temporalHorizon);

            for (int t = 0; t < histories[l][0].size(); t++)
                histories[l][0][t] = IntBuffer(inSize, 0);

            dLayers[l].resize(1);
            dLayers[l][0].resize(layerDescs[l].ticksPerUpdate);

            // Decoder visible layer descriptors
            Array<Decoder::VisibleLayerDesc> dVisibleLayerDescs(l < encLayers.size() - 1 ? 2 : 1);

            dVisibleLayerDescs[0].size = layerDescs[l].errorSize;
            dVisibleLayerDescs[0].radius = layerDescs[l].dRadius;

            if (l < encLayers.size() - 1) {
                dVisibleLayerDescs[1].size = layerDescs[l].hiddenSize;
                dVisibleLayerDescs[1].radius = layerDescs[l].dRadius;
            }

            // Create actors
            for (int t = 0; t < dLayers[l][0].size(); t++)
                dLayers[l][0][t].initRandom(layerDescs[l - 1].hiddenSize, dVisibleLayerDescs);
        }
        
        // Create the sparse coding layer
        encLayers[l].hidden.initRandom(layerDescs[l].hiddenSize, hVisibleLayerDescs);
        encLayers[l].error.initRandom(layerDescs[l].errorSize, eVisibleLayerDescs);

        errors[l] = FloatBuffer(encLayers[l].error.getHiddenCIs().size(), 0.0f);
    }
}

// Simulation step/tick
void Hierarchy::step(
    const Array<const IntBuffer*> &inputCIs,
    bool learnEnabled,
    float reward
) {
    // First tick is always 0
    ticks[0] = 0;

    // Add input to first layer history   
    for (int i = 0; i < inputSizes.size(); i++) {
        histories[0][i].pushFront();

        histories[0][i][0] = *inputCIs[i];
    }

    // Set all updates to no update, will be set to true if an update occurred later
    updates.fill(false);

    // Forward
    for (int l = 0; l < encLayers.size(); l++) {
        // If is time for layer to tick
        if (l == 0 || ticks[l] >= ticksPerUpdate[l]) {
            // Reset tick
            ticks[l] = 0;

            // Updated
            updates[l] = true;

            // Clear hidden errors
            errors[l].fill(0.0f);

            // Accumulate
            for (int i = 0; i < dLayers[l].size(); i++) {
                for (int t = 0; t < dLayers[l][i].size(); t++)
                    dLayers[l][i][t].generateErrors(&histories[l][i][t], &errors[l], 0);
            }

            Array<const IntBuffer*> layerInputCIs(histories[l].size() * histories[l][0].size());

            int index = 0;

            for (int i = 0; i < histories[l].size(); i++) {
                for (int t = 0; t < histories[l][i].size(); t++)
                    layerInputCIs[index++] = &histories[l][i][t];
            }

            // Activate sparse coder
            encLayers[l].hidden.step(layerInputCIs, learnEnabled);
            encLayers[l].error.step(layerInputCIs, &errors[l], learnEnabled);

            // Add to next layer's history
            if (l < encLayers.size() - 1) {
                int lNext = l + 1;

                histories[lNext][0].pushFront();

                histories[lNext][0][0] = encLayers[l].hidden.getHiddenCIs();

                ticks[lNext]++;
            }
        }
    }

    // Backward
    for (int l = encLayers.size() - 1; l >= 0; l--) {
        if (updates[l]) {
            // Feed back is current layer state and next higher layer prediction
            Array<const IntBuffer*> feedBackCIs(l < encLayers.size() - 1 ? 2 : 1);

            feedBackCIs[0] = &encLayers[l].error.getHiddenCIs();

            if (l < encLayers.size() - 1)
                feedBackCIs[1] = &dLayers[l + 1][0][ticksPerUpdate[l + 1] - 1 - ticks[l + 1]].getHiddenCIs();

            // Step actor layers
            for (int i = 0; i < dLayers[l].size(); i++) {
                for (int t = 0; t < dLayers[l][i].size(); t++) {
                    if (learnEnabled)
                        dLayers[l][i][t].learn(&histories[l][i][t]);

                    dLayers[l][i][t].activate(feedBackCIs);
                }
            }

            if (l == 0) {
                // Step actors
                for (int i = 0; i < aLayers.size(); i++) {
                    if (aLayers[i] != nullptr)
                        aLayers[i]->step(feedBackCIs, &histories[l][i][0], reward, learnEnabled);
                }
            }
        }
    }
}

int Hierarchy::size() const {
    int size = 2 * sizeof(int) + inputSizes.size() * sizeof(Int3) + updates.size() * sizeof(unsigned char) + 2 * ticks.size() * sizeof(int);

    for (int l = 0; l < encLayers.size(); l++) {
        size += sizeof(int);

        for (int i = 0; i < histories[l].size(); i++) {
            size += 2 * sizeof(int);

            for (int t = 0; t < histories[l][i].size(); t++)
                size += sizeof(int) + histories[l][i][t].size() * sizeof(int);
        }

        size += encLayers[l].hidden.size() + encLayers[l].error.size();

        for (int i = 0; i < dLayers[l].size(); i++) {
            for (int t = 0; t < dLayers[l][i].size(); t++)
                size += dLayers[l][i][t].size();
        }
    }

    for (int v = 0; v < aLayers.size(); v++) {
        size += sizeof(unsigned char);

        if (aLayers[v] != nullptr)
            size += aLayers[v]->size();
    }

    return size;
}

int Hierarchy::stateSize() const {
    int size = updates.size() * sizeof(unsigned char) + ticks.size() * sizeof(int);

    for (int l = 0; l < encLayers.size(); l++) {
        for (int i = 0; i < histories[l].size(); i++) {
            size += sizeof(int);

            for (int t = 0; t < histories[l][i].size(); t++)
                size += histories[l][i][t].size() * sizeof(int);
        }

        size += encLayers[l].hidden.stateSize() + encLayers[l].error.stateSize();
        
        // Decoders
        for (int i = 0; i < dLayers[l].size(); i++) {
            for (int t = 0; t < dLayers[l][i].size(); t++)
                size += dLayers[l][i][t].stateSize();
        }
    }

    // Actors
    for (int v = 0; v < aLayers.size(); v++) {
        if (aLayers[v] != nullptr)
            size += aLayers[v]->stateSize();
    }

    return size;
}

void Hierarchy::write(
    StreamWriter &writer
) const {
    int numLayers = encLayers.size();

    writer.write(reinterpret_cast<const void*>(&numLayers), sizeof(int));

    int numInputs = inputSizes.size();

    writer.write(reinterpret_cast<const void*>(&numInputs), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&inputSizes[0]), numInputs * sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&updates[0]), updates.size() * sizeof(unsigned char));
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

        encLayers[l].hidden.write(writer);
        encLayers[l].error.write(writer);

        // Decoders
        for (int i = 0; i < dLayers[l].size(); i++) {
            for (int t = 0; t < dLayers[l][i].size(); t++)
                dLayers[l][i][t].write(writer);
        }
    }

    // Actors
    for (int v = 0; v < aLayers.size(); v++) {
        unsigned char exists = aLayers[v] != nullptr;

        writer.write(reinterpret_cast<const void*>(&exists), sizeof(unsigned char));

        if (exists)
            aLayers[v]->write(writer);
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

    encLayers.resize(numLayers);
    dLayers.resize(numLayers);

    histories.resize(numLayers);
    
    updates.resize(numLayers);
    ticks.resize(numLayers);
    ticksPerUpdate.resize(numLayers);
    errors.resize(numLayers);

    reader.read(reinterpret_cast<void*>(&updates[0]), updates.size() * sizeof(unsigned char));
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

        encLayers[l].hidden.read(reader);
        encLayers[l].error.read(reader);
        
        dLayers[l].resize(l == 0 ? inputSizes.size() : 1);

        // Decoders
        for (int i = 0; i < dLayers[l].size(); i++) {
            dLayers[l][i].resize(l == 0 ? 1 : ticksPerUpdate[l]);

            for (int t = 0; t < dLayers[l][i].size(); t++)
                dLayers[l][i][t].read(reader);
        }

        errors[l] = FloatBuffer(encLayers[l].error.getHiddenCIs().size(), 0.0f);
    }

    // Actors
    aLayers.resize(inputSizes.size());

    for (int v = 0; v < aLayers.size(); v++) {
        unsigned char exists;

        reader.read(reinterpret_cast<void*>(&exists), sizeof(unsigned char));

        if (exists) {
            aLayers[v].make();
            aLayers[v]->read(reader);
        }
    }
}

void Hierarchy::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&updates[0]), updates.size() * sizeof(unsigned char));
    writer.write(reinterpret_cast<const void*>(&ticks[0]), ticks.size() * sizeof(int));

    for (int l = 0; l < encLayers.size(); l++) {
        for (int i = 0; i < histories[l].size(); i++) {
            int historyStart = histories[l][i].start;

            writer.write(reinterpret_cast<const void*>(&historyStart), sizeof(int));

            for (int t = 0; t < histories[l][i].size(); t++)
                writer.write(reinterpret_cast<const void*>(&histories[l][i][t][0]), histories[l][i][t].size() * sizeof(int));
        }

        encLayers[l].hidden.writeState(writer);
        encLayers[l].error.writeState(writer);

        // Decoders
        for (int i = 0; i < dLayers[l].size(); i++) {
            for (int t = 0; t < dLayers[l][i].size(); t++)
                dLayers[l][i][t].writeState(writer);
        }
    }

    // Actors
    for (int v = 0; v < aLayers.size(); v++) {
        if (aLayers[v] != nullptr)
            aLayers[v]->writeState(writer);
    }
}

void Hierarchy::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&updates[0]), updates.size() * sizeof(unsigned char));
    reader.read(reinterpret_cast<void*>(&ticks[0]), ticks.size() * sizeof(int));
    
    for (int l = 0; l < encLayers.size(); l++) {
        for (int i = 0; i < histories[l].size(); i++) {
            int historyStart;
            
            reader.read(reinterpret_cast<void*>(&historyStart), sizeof(int));

            histories[l][i].start = historyStart;

            for (int t = 0; t < histories[l][i].size(); t++)
                reader.read(reinterpret_cast<void*>(&histories[l][i][t][0]), histories[l][i][t].size() * sizeof(int));
        }

        encLayers[l].hidden.readState(reader);
        encLayers[l].error.readState(reader);
        
        // Decoders
        for (int i = 0; i < dLayers[l].size(); i++) {
            for (int t = 0; t < dLayers[l][i].size(); t++)
                dLayers[l][i][t].readState(reader);
        }
    }

    // Actors
    for (int v = 0; v < aLayers.size(); v++) {
        if (aLayers[v] != nullptr)
            aLayers[v]->readState(reader);
    }
}
