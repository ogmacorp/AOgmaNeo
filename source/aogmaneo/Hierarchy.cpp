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
    eLayers = other.eLayers;
    dLayers = other.dLayers;

    inputSizes = other.inputSizes;
    hiddenCIsPrev = other.hiddenCIsPrev;

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
    eLayers.resize(layerDescs.size());
    dLayers.resize(layerDescs.size());
    hiddenCIsPrev.resize(layerDescs.size());

    // Cache input sizes
    inputSizes.resize(ioDescs.size());

    for (int i = 0; i < inputSizes.size(); i++)
        inputSizes[i] = ioDescs[i].size;

    // Iterate through layers
    for (int l = 0; l < layerDescs.size(); l++) {
        // Create sparse coder visible layer descriptors
        Array<Encoder::VisibleLayerDesc> eVisibleLayerDescs;

        // If first layer
        if (l == 0) {
            eVisibleLayerDescs.resize(inputSizes.size() + (layerDescs[l].rRadius != -1 ? 1 : 0));

            for (int i = 0; i < inputSizes.size(); i++) {
                eVisibleLayerDescs[i].size = inputSizes[i];
                eVisibleLayerDescs[i].radius = ioDescs[i].ffRadius;
            }

            if (layerDescs[l].rRadius != -1) {
                eVisibleLayerDescs[inputSizes.size()].size = layerDescs[l].hiddenSize;
                eVisibleLayerDescs[inputSizes.size()].radius = layerDescs[l].rRadius;
            }
            
            dLayers[l].resize(inputSizes.size());
            aLayers.resize(inputSizes.size());

            // Create predictors
            for (int i = 0; i < inputSizes.size(); i++) {
                // Decoder visible layer descriptors
                Array<Decoder::VisibleLayerDesc> dVisibleLayerDescs(l < eLayers.size() - 1 ? 2 : 1);

                dVisibleLayerDescs[0].size = layerDescs[l].hiddenSize;
                dVisibleLayerDescs[0].radius = ioDescs[i].fbRadius;

                if (l < eLayers.size() - 1) {
                    dVisibleLayerDescs[1].size = layerDescs[l].hiddenSize;
                    dVisibleLayerDescs[1].radius = ioDescs[i].fbRadius;
                }

                dLayers[l][i].initRandom(inputSizes[i], dVisibleLayerDescs);

                if (ioDescs[i].type == IOType::action) {
                    // Actor visible layer descriptors
                    Array<Actor::VisibleLayerDesc> aVisibleLayerDescs(l < eLayers.size() - 1 ? 2 : 1);

                    aVisibleLayerDescs[0].size = layerDescs[l].hiddenSize;
                    aVisibleLayerDescs[0].radius = ioDescs[i].fbRadius;

                    if (l < eLayers.size() - 1) {
                        aVisibleLayerDescs[1].size = layerDescs[l].hiddenSize;
                        aVisibleLayerDescs[1].radius = ioDescs[i].fbRadius;
                    }

                    aLayers[i].make();

                    aLayers[i]->initRandom(inputSizes[i], ioDescs[i].historyCapacity, aVisibleLayerDescs);
                }
            }
        }
        else {
            eVisibleLayerDescs.resize(layerDescs[l].rRadius != -1 ? 2 : 1);

            eVisibleLayerDescs[0].size = layerDescs[l - 1].hiddenSize;
            eVisibleLayerDescs[0].radius = layerDescs[l].ffRadius;

            if (layerDescs[l].rRadius != -1) {
                eVisibleLayerDescs[1].size = layerDescs[l].hiddenSize;
                eVisibleLayerDescs[1].radius = layerDescs[l].rRadius;
            }

            dLayers[l].resize(1);

            // Decoder visible layer descriptors
            Array<Decoder::VisibleLayerDesc> dVisibleLayerDescs(l < eLayers.size() - 1 ? 2 : 1);

            dVisibleLayerDescs[0].size = layerDescs[l].hiddenSize;
            dVisibleLayerDescs[0].radius = layerDescs[l].fbRadius;

            if (l < eLayers.size() - 1) {
                dVisibleLayerDescs[1].size = layerDescs[l].hiddenSize;
                dVisibleLayerDescs[1].radius = layerDescs[l].fbRadius;
            }

            // Create predictor
            dLayers[l][0].initRandom(layerDescs[l - 1].hiddenSize, dVisibleLayerDescs);
        }
        
        // Create the sparse coding layer
        eLayers[l].initRandom(layerDescs[l].hiddenSize, eVisibleLayerDescs);

        hiddenCIsPrev[l] = eLayers[l].getHiddenCIs();

        // Default recurrence
        setRecurrence(l, 0.25f);
    }
}

// Simulation step/tick
void Hierarchy::step(
    const Array<const IntBuffer*> &inputCIs,
    bool learnEnabled,
    float reward,
    bool mimic
) {
    // Forward
    for (int l = 0; l < eLayers.size(); l++) {
        if (l == 0) {
            if (eLayers[l].getNumVisibleLayers() > inputSizes.size()) {
                Array<const IntBuffer*> layerInputCIs(inputSizes.size() + 1);

                for (int i = 0; i < inputSizes.size(); i++)
                    layerInputCIs[i] = inputCIs[i];

                layerInputCIs[inputSizes.size()] = &hiddenCIsPrev[l];

                eLayers[l].step(layerInputCIs, learnEnabled);
            }
            else
                eLayers[l].step(inputCIs, learnEnabled);
        }
        else {
            if (eLayers[l].getNumVisibleLayers() > 1) {
                Array<const IntBuffer*> layerInputCIs(2);

                layerInputCIs[0] = &eLayers[l - 1].getHiddenCIs();
                layerInputCIs[1] = &hiddenCIsPrev[l];

                eLayers[l].step(layerInputCIs, learnEnabled);
            }
            else {
                Array<const IntBuffer*> layerInputCIs(1);

                layerInputCIs[0] = &eLayers[l - 1].getHiddenCIs();

                eLayers[l].step(layerInputCIs, learnEnabled);
            }
        }

        hiddenCIsPrev[l] = eLayers[l].getHiddenCIs();
    }

    // Backward
    for (int l = eLayers.size() - 1; l >= 0; l--) {
        // Feed back is current layer state and next higher layer prediction
        Array<const IntBuffer*> feedBackCIs(l < eLayers.size() - 1 ? 2 : 1);

        feedBackCIs[0] = &eLayers[l].getHiddenCIs();

        if (l < eLayers.size() - 1)
            feedBackCIs[1] = &dLayers[l + 1][0].getHiddenCIs();

        if (l == 0) {
            // Step predictors
            for (int i = 0; i < dLayers[l].size(); i++) {
                if (learnEnabled)
                    dLayers[l][i].learn(inputCIs[i]);

                dLayers[l][i].activate(feedBackCIs);
            }

            // Step actors
            for (int i = 0; i < aLayers.size(); i++) {
                if (aLayers[i] != nullptr)
                    aLayers[i]->step(feedBackCIs, inputCIs[i], reward, learnEnabled, mimic);
            }
        }
        else {
            // Step predictors
            if (learnEnabled)
                dLayers[l][0].learn(&eLayers[l].getHiddenCIs());

            dLayers[l][0].activate(feedBackCIs);
        }
    }
}

int Hierarchy::size() const {
    int size = 2 * sizeof(int) + inputSizes.size() * sizeof(Int3);

    for (int l = 0; l < eLayers.size(); l++) {
        size += eLayers[l].size();

        for (int i = 0; i < dLayers[l].size(); i++)
            size += dLayers[l][i].size();
    }

    for (int v = 0; v < aLayers.size(); v++) {
        size += sizeof(Byte);

        if (aLayers[v] != nullptr)
            size += aLayers[v]->size();
    }

    return size;
}

int Hierarchy::stateSize() const {
    int size = 0;

    for (int l = 0; l < eLayers.size(); l++) {
        size += eLayers[l].stateSize();
        
        // Decoders
        for (int i = 0; i < dLayers[l].size(); i++)
            size += dLayers[l][i].stateSize();
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
    int numLayers = eLayers.size();

    writer.write(reinterpret_cast<const void*>(&numLayers), sizeof(int));

    int numInputs = inputSizes.size();

    writer.write(reinterpret_cast<const void*>(&numInputs), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&inputSizes[0]), numInputs * sizeof(Int3));

    for (int l = 0; l < numLayers; l++) {
        eLayers[l].write(writer);

        // Decoders
        for (int i = 0; i < dLayers[l].size(); i++)
            dLayers[l][i].write(writer);

        writer.write(reinterpret_cast<const void*>(&hiddenCIsPrev[l][0]), hiddenCIsPrev[l].size() * sizeof(int));
    }

    // Actors
    for (int v = 0; v < aLayers.size(); v++) {
        Byte exists = aLayers[v] != nullptr;

        writer.write(reinterpret_cast<const void*>(&exists), sizeof(Byte));

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

    eLayers.resize(numLayers);
    dLayers.resize(numLayers);

    hiddenCIsPrev.resize(numLayers);
    
    for (int l = 0; l < numLayers; l++) {
        eLayers[l].read(reader);
        
        dLayers[l].resize(l == 0 ? inputSizes.size() : 1);

        // Decoders
        for (int i = 0; i < dLayers[l].size(); i++)
            dLayers[l][i].read(reader);

        hiddenCIsPrev[l].resize(eLayers[l].getHiddenCIs().size());

        reader.read(reinterpret_cast<void*>(&hiddenCIsPrev[l][0]), hiddenCIsPrev[l].size() * sizeof(int));
    }

    // Actors
    aLayers.resize(inputSizes.size());

    for (int v = 0; v < aLayers.size(); v++) {
        Byte exists;

        reader.read(reinterpret_cast<void*>(&exists), sizeof(Byte));

        if (exists) {
            aLayers[v].make();
            aLayers[v]->read(reader);
        }
    }
}

void Hierarchy::writeState(
    StreamWriter &writer
) const {
    for (int l = 0; l < eLayers.size(); l++) {
        eLayers[l].writeState(writer);

        // Decoders
        for (int i = 0; i < dLayers[l].size(); i++)
            dLayers[l][i].writeState(writer);
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
    for (int l = 0; l < eLayers.size(); l++) {
        eLayers[l].readState(reader);
        
        // Decoders
        for (int i = 0; i < dLayers[l].size(); i++)
            dLayers[l][i].readState(reader);
    }

    // Actors
    for (int v = 0; v < aLayers.size(); v++) {
        if (aLayers[v] != nullptr)
            aLayers[v]->readState(reader);
    }
}
