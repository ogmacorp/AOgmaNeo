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
    eLayers.resize(layerDescs.size());
    dLayers.resize(layerDescs.size());

    // Cache input sizes
    ioSizes.resize(ioDescs.size());

    int numPredictions = 0;

    for (int i = 0; i < ioSizes.size(); i++) {
        ioSizes[i] = ioDescs[i].size;

        if (ioDescs[i].type == prediction)
            numPredictions++;
    }

    // Iterate through layers
    for (int l = 0; l < layerDescs.size(); l++) {
        // Create sparse coder visible layer descriptors
        Array<Encoder::VisibleLayerDesc> eVisibleLayerDescs;

        // If first layer
        if (l == 0) {
            eVisibleLayerDescs.resize(ioSizes.size() + (layerDescs[l].rRadius != -1 ? 1 : 0));

            for (int i = 0; i < ioSizes.size(); i++) {
                eVisibleLayerDescs[i].size = ioSizes[i];
                eVisibleLayerDescs[i].radius = ioDescs[i].eRadius;
            }

            if (layerDescs[l].rRadius != -1) {
                eVisibleLayerDescs[ioSizes.size()].size = layerDescs[l].hiddenSize;
                eVisibleLayerDescs[ioSizes.size()].radius = layerDescs[l].rRadius;
            }

            dLayers[l].resize(numPredictions);

            iIndices.resize(numPredictions);
            dIndices = IntBuffer(ioSizes.size(), -1);

            // Create decoders
            int dIndex = 0;

            for (int i = 0; i < ioSizes.size(); i++) {
                if (ioDescs[i].type == prediction) {
                    // Decoder visible layer descriptors
                    Decoder::VisibleLayerDesc dVisibleLayerDesc;

                    dVisibleLayerDesc.size = layerDescs[l].hiddenSize;
                    dVisibleLayerDesc.radius = ioDescs[i].dRadius;

                    dLayers[l][dIndex].initRandom(ioSizes[i], ioDescs[i].historyCapacity, dVisibleLayerDesc);

                    iIndices[dIndex] = i;
                    dIndices[i] = dIndex;
                    dIndex++;
                }
            }
        }
        else {
            eVisibleLayerDescs.resize(1 + (layerDescs[l].rRadius != -1 ? 1 : 0));

            eVisibleLayerDescs[0].size = layerDescs[l - 1].hiddenSize;
            eVisibleLayerDescs[0].radius = layerDescs[l].eRadius;

            if (layerDescs[l].rRadius != -1) {
                eVisibleLayerDescs[1].size = layerDescs[l].hiddenSize;
                eVisibleLayerDescs[1].radius = layerDescs[l].rRadius;
            }

            dLayers[l].resize(1);

            // Decoder visible layer descriptors
            Decoder::VisibleLayerDesc dVisibleLayerDesc;

            dVisibleLayerDesc.size = layerDescs[l].hiddenSize;
            dVisibleLayerDesc.radius = layerDescs[l].dRadius;

            // Create decoders
            dLayers[l][0].initRandom(layerDescs[l - 1].hiddenSize, layerDescs[l].historyCapacity, dVisibleLayerDesc);
        }
        
        // Create the sparse coding layer
        eLayers[l].initRandom(layerDescs[l].hiddenSize, eVisibleLayerDescs);
    }
}

// Simulation step/tick
void Hierarchy::step(
    const Array<const IntBuffer*> &inputCIs,
    const IntBuffer* topProgCIs,
    bool learnEnabled
) {
    // Forward
    for (int l = 0; l < eLayers.size(); l++) {
        IntBuffer hiddenCIsPrev = eLayers[l].getHiddenCIs();

        Array<const IntBuffer*> encoderCIs(eLayers[l].getNumVisibleLayers());

        if (l == 0) {
            for (int i = 0; i < ioSizes.size(); i++)
                encoderCIs[i] = inputCIs[i];

            if (ioSizes.size() < encoderCIs.size())
                encoderCIs[ioSizes.size()] = &hiddenCIsPrev;
        }
        else {
            encoderCIs[0] = &eLayers[l - 1].getHiddenCIs();

            if (encoderCIs.size() > 1)
                encoderCIs[1] = &hiddenCIsPrev;
        }

        // Activate sparse coder
        eLayers[l].step(encoderCIs, learnEnabled);
    }

    // Backward
    for (int l = dLayers.size() - 1; l >= 0; l--) {
        for (int d = 0; d < dLayers[l].size(); d++)
            dLayers[l][d].step(l < eLayers.size() - 1 ? &dLayers[l + 1][0].getHiddenCIs() : topProgCIs, &eLayers[l].getHiddenCIs(), l == 0 ? inputCIs[iIndices[d]] : &eLayers[l - 1].getHiddenCIs(), learnEnabled, true);
    }
}

int Hierarchy::size() const {
    int size = 3 * sizeof(int) + ioSizes.size() * sizeof(Int3) + iIndices.size() * sizeof(int) + dIndices.size() * sizeof(int);

    for (int l = 0; l < eLayers.size(); l++) {
        size += eLayers[l].size();

        for (int d = 0; d < dLayers[l].size(); d++)
            size += dLayers[l][d].size();
    }

    return size;
}

int Hierarchy::stateSize() const {
    int size = 0;

    for (int l = 0; l < eLayers.size(); l++) {
        size += eLayers[l].stateSize();
        
        // Decoders
        for (int d = 0; d < dLayers[l].size(); d++)
            size += dLayers[l][d].stateSize();
    }

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

    writer.write(reinterpret_cast<const void*>(&numPredictions), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&ioSizes[0]), numIO * sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&iIndices[0]), iIndices.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&dIndices[0]), dIndices.size() * sizeof(int));

    for (int l = 0; l < numLayers; l++) {
        eLayers[l].write(writer);

        // Decoders
        for (int d = 0; d < dLayers[l].size(); d++)
            dLayers[l][d].write(writer);
    }
}

void Hierarchy::read(
    StreamReader &reader
) {
    int numLayers;

    reader.read(reinterpret_cast<void*>(&numLayers), sizeof(int));

    int numIO;

    reader.read(reinterpret_cast<void*>(&numIO), sizeof(int));

    int numPredictions;

    reader.read(reinterpret_cast<void*>(&numPredictions), sizeof(int));

    ioSizes.resize(numIO);

    reader.read(reinterpret_cast<void*>(&ioSizes[0]), numIO * sizeof(Int3));

    eLayers.resize(numLayers);
    dLayers.resize(numLayers);

    iIndices.resize(numPredictions);
    dIndices.resize(numIO);

    reader.read(reinterpret_cast<void*>(&iIndices[0]), iIndices.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&dIndices[0]), dIndices.size() * sizeof(int));
    
    for (int l = 0; l < numLayers; l++) {
        eLayers[l].read(reader);
        
        dLayers[l].resize(l == 0 ? numPredictions : 1);

        // Decoders
        for (int d = 0; d < dLayers[l].size(); d++)
            dLayers[l][d].read(reader);
    }
}

void Hierarchy::writeState(
    StreamWriter &writer
) const {
    for (int l = 0; l < eLayers.size(); l++) {
        eLayers[l].writeState(writer);

        // Decoders
        for (int d = 0; d < dLayers[l].size(); d++)
            dLayers[l][d].writeState(writer);
    }
}

void Hierarchy::readState(
    StreamReader &reader
) {
    for (int l = 0; l < eLayers.size(); l++) {
        eLayers[l].readState(reader);
        
        // Decoders
        for (int d = 0; d < dLayers[l].size(); d++)
            dLayers[l][d].readState(reader);
    }
}
