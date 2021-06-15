// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
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
            eVisibleLayerDescs.resize(inputSizes.size() * layerDescs[l].temporalHorizon);

            for (int i = 0; i < inputSizes.size(); i++) {
                for (int t = 0; t < layerDescs[l].temporalHorizon; t++) {
                    int index = t + layerDescs[l].temporalHorizon * i;

                    eVisibleLayerDescs[index].size = inputSizes[i];
                    eVisibleLayerDescs[index].radius = ioDescs[i].eRadius;
                }
            }
            
            dLayers[l].resize(inputSizes.size());

            // Create predictors
            for (int i = 0; i < inputSizes.size(); i++) {
                // Decoder visible layer descriptors
                Decoder::VisibleLayerDesc dVisibleLayerDesc;

                dVisibleLayerDesc.size = layerDescs[l].hiddenSize;
                dVisibleLayerDesc.radius = ioDescs[i].dRadius;

                dLayers[l][i].initRandom(inputSizes[i], ioDescs[i].historyCapacity, dVisibleLayerDesc);
            }
        }
        else {
            eVisibleLayerDescs.resize(layerDescs[l].temporalHorizon);

            for (int t = 0; t < layerDescs[l].temporalHorizon; t++) {
                eVisibleLayerDescs[t].size = layerDescs[l - 1].hiddenSize;
                eVisibleLayerDescs[t].radius = layerDescs[l].eRadius;
            }

            dLayers[l].resize(1);

            // Decoder visible layer descriptors
            Decoder::VisibleLayerDesc dVisibleLayerDesc;

            dVisibleLayerDesc.size = layerDescs[l].hiddenSize;
            dVisibleLayerDesc.radius = layerDescs[l].dRadius;

            dLayers[l][0].initRandom(layerDescs[l - 1].hiddenSize, layerDescs[l].historyCapacity, dVisibleLayerDesc);
        }
        
        // Create the sparse coding layer
        eLayers[l].initRandom(layerDescs[l].hiddenSize, eVisibleLayerDescs);
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
        Array<const IntBuffer*> layerInputCIs;

        if (l == 0) {
            layerInputCIs.resize(inputCIs.size());

            for (int i = 0; i < inputCIs.size(); i++)
                layerInputCIs[i] = inputCIs[i];
        }
        else {
            layerInputCIs.resize(1);

            layerInputCIs[0] = &eLayers[l - 1].getHiddenCIs();
        }
            
        eLayers[l].step(layerInputCIs, learnEnabled);
    }

    // Backward
    for (int l = eLayers.size() - 1; l >= 0; l--) {
        for (int i = 0; i < dLayers[l].size(); i++)
            dLayers[l][i].step(l < eLayers.size() - 1 ? &dLayers[l + 1][0].getHiddenCIs() : topGoalCIs, &eLayers[l].getHiddenCIs(), l == 0 ? inputCIs[i] : &eLayers[l - 1].getHiddenCIs(), learnEnabled, true);
    }
}

int Hierarchy::size() const {
    int size = 2 * sizeof(int) + inputSizes.size() * sizeof(Int3);

    for (int l = 0; l < eLayers.size(); l++) {
        size += sizeof(int);

        size += eLayers[l].size();

        for (int i = 0; i < dLayers[l].size(); i++)
            size += dLayers[l][i].size();
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

    for (int l = 0; l < numLayers; l++) {
        eLayers[l].read(reader);
        
        dLayers[l].resize(l == 0 ? inputSizes.size() : 1);

        // Decoders
        for (int i = 0; i < dLayers[l].size(); i++)
            dLayers[l][i].read(reader);
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
}
