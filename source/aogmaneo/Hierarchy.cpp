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

    inputSizes = other.inputSizes;
    hiddenCIsPrev = other.hiddenCIsPrev;

    dLayers.resize(eLayers.size());

    for (int l = 0; l < dLayers.size(); l++) {
        dLayers[l].resize(other.dLayers[l].size());

        for (int v = 0; v < dLayers[l].size(); v++) {
            if (other.dLayers[l][v] != nullptr) {
                dLayers[l][v].make();
                
                (*dLayers[l][v]) = (*other.dLayers[l][v]);
            }
            else
                dLayers[l][v] = nullptr;
        }
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

            // Create predictors
            for (int i = 0; i < inputSizes.size(); i++) {
                if (ioDescs[i].type == IOType::prediction) {
                    // Decoder visible layer descriptors
                    Decoder::VisibleLayerDesc dVisibleLayerDesc;

                    dVisibleLayerDesc.size = layerDescs[l].hiddenSize;
                    dVisibleLayerDesc.radius = ioDescs[i].fbRadius;

                    dLayers[l][i].make();

                    dLayers[l][i]->initRandom(inputSizes[i], layerDescs[l].historyCapacity, dVisibleLayerDesc);
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
            Decoder::VisibleLayerDesc dVisibleLayerDesc;

            dVisibleLayerDesc.size = layerDescs[l].hiddenSize;
            dVisibleLayerDesc.radius = layerDescs[l].fbRadius;

            // Create decoder
            dLayers[l][0].make();

            dLayers[l][0]->initRandom(layerDescs[l - 1].hiddenSize, layerDescs[l].historyCapacity, dVisibleLayerDesc);
        }
        
        // Create the sparse coding layer
        eLayers[l].initRandom(layerDescs[l].hiddenSize, layerDescs[l].clumpSize, eVisibleLayerDescs);

        hiddenCIsPrev[l] = eLayers[l].getHiddenCIs();

        // Default recurrences
        setRecurrence(l, 0.5f);
    }
}

// Simulation step/tick
void Hierarchy::step(
    const Array<const IntBuffer*> &inputCIs,
    const IntBuffer* topGoalCIs,
    bool learnEnabled
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
        for (int i = 0; i < dLayers[l].size(); i++) {
            if (dLayers[l][i] != nullptr)
                dLayers[l][i]->step(l < eLayers.size() - 1 ? &dLayers[l + 1][0]->getHiddenCIs() : topGoalCIs, &eLayers[l].getHiddenCIs(), l == 0 ? inputCIs[i] : &eLayers[l - 1].getHiddenCIs(), learnEnabled, true);
        }
    }
}

int Hierarchy::size() const {
    int size = 2 * sizeof(int) + inputSizes.size() * sizeof(Int3);

    for (int l = 0; l < eLayers.size(); l++) {
        size += eLayers[l].size();

        for (int v = 0; v < dLayers[l].size(); v++) {
            size += sizeof(Byte);

            if (dLayers[l][v] != nullptr)
                size += dLayers[l][v]->size();
        }
    }

    return size;
}

int Hierarchy::stateSize() const {
    int size = 0;

    for (int l = 0; l < eLayers.size(); l++) {
        size += eLayers[l].stateSize();
        
        // Decoders
        for (int v = 0; v < dLayers[l].size(); v++) {
            if (dLayers[l][v] != nullptr)
                size += dLayers[l][v]->stateSize();
        }
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
        for (int v = 0; v < dLayers[l].size(); v++) {
            Byte exists = dLayers[l][v] != nullptr;

            writer.write(reinterpret_cast<const void*>(&exists), sizeof(Byte));

            if (exists)
                dLayers[l][v]->write(writer);
        }

        writer.write(reinterpret_cast<const void*>(&hiddenCIsPrev[l][0]), hiddenCIsPrev[l].size() * sizeof(int));
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
        for (int v = 0; v < dLayers[l].size(); v++) {
            Byte exists;

            reader.read(reinterpret_cast<void*>(&exists), sizeof(Byte));

            if (exists) {
                dLayers[l][v].make();

                dLayers[l][v]->read(reader);
            }
        }

        hiddenCIsPrev[l].resize(eLayers[l].getHiddenCIs().size());

        reader.read(reinterpret_cast<void*>(&hiddenCIsPrev[l][0]), hiddenCIsPrev[l].size() * sizeof(int));
    }
}

void Hierarchy::writeState(
    StreamWriter &writer
) const {
    for (int l = 0; l < eLayers.size(); l++) {
        eLayers[l].writeState(writer);

        // Decoders
        for (int v = 0; v < dLayers[l].size(); v++) {
            if (dLayers[l][v] != nullptr)
                dLayers[l][v]->writeState(writer);
        }
    }
}

void Hierarchy::readState(
    StreamReader &reader
) {
    for (int l = 0; l < eLayers.size(); l++) {
        eLayers[l].readState(reader);
        
        // Decoders
        for (int v = 0; v < dLayers[l].size(); v++) {
            if (dLayers[l][v] != nullptr)
                dLayers[l][v]->readState(reader);
        }
    }
}
