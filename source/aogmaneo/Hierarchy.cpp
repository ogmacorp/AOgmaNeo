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

    // Cache input sizes
    ioSizes.resize(ioDescs.size());
    ioTypes.resize(ioDescs.size());

    int numPredictions = 0;
    int numActions = 0;

    for (int i = 0; i < ioSizes.size(); i++) {
        ioSizes[i] = ioDescs[i].size;
        ioTypes[i] = static_cast<Byte>(ioDescs[i].type);

        if (ioDescs[i].type == prediction)
            numPredictions++;
        else if (ioDescs[i].type == action) {
            numActions++;
            numPredictions++;
        }
    }

    // Iterate through layers
    for (int l = 0; l < layerDescs.size(); l++) {
        // Create sparse coder visible layer descriptors
        Array<Encoder::VisibleLayerDesc> eVisibleLayerDescs;

        // If first layer
        if (l == 0) {
            eVisibleLayerDescs.resize(ioSizes.size() + (layerDescs[l].rRadius >= 0 ? 1 : 0));

            for (int i = 0; i < ioSizes.size(); i++) {
                eVisibleLayerDescs[i].size = ioSizes[i];
                eVisibleLayerDescs[i].radius = ioDescs[i].eRadius;
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
            eVisibleLayerDescs.resize(1 + (layerDescs[l].rRadius >= 0 ? 1 : 0));

            eVisibleLayerDescs[0].size = layerDescs[l - 1].hiddenSize;
            eVisibleLayerDescs[0].radius = layerDescs[l].eRadius;

            dLayers[l].resize(1);

            // Decoder visible layer descriptors
            Array<Decoder::VisibleLayerDesc> dVisibleLayerDescs(l < eLayers.size() - 1 ? 2 : 1);

            dVisibleLayerDescs[0].size = layerDescs[l].hiddenSize;
            dVisibleLayerDescs[0].radius = layerDescs[l].dRadius;

            if (l < eLayers.size() - 1)
                dVisibleLayerDescs[1] = dVisibleLayerDescs[0];

            // Create decoders
            for (int t = 0; t < dLayers[l].size(); t++)
                dLayers[l][t].initRandom(layerDescs[l - 1].hiddenSize, dVisibleLayerDescs);
        }

        if (layerDescs[l].rRadius >= 0) {
            eVisibleLayerDescs[eVisibleLayerDescs.size() - 1].size = layerDescs[l].hiddenSize;
            eVisibleLayerDescs[eVisibleLayerDescs.size() - 1].radius = layerDescs[l].rRadius;
        }
        
        // Create the sparse coding layer
        eLayers[l].initRandom(layerDescs[l].hiddenSize, eVisibleLayerDescs);

        errors[l] = FloatBuffer(eLayers[l].getHiddenActs().size(), 0.0f);
    }
}

void Hierarchy::step(
    const Array<const IntBuffer*> &inputCIs,
    bool learnEnabled,
    float reward,
    bool mimic
) {
    Array<IntBuffer> hiddenCIsPrev(eLayers.size());

    // Forward
    for (int l = 0; l < eLayers.size(); l++) {
        hiddenCIsPrev[l] = eLayers[l].getHiddenCIs();

        Array<const IntBuffer*> eInputCIs(eLayers[l].getNumVisibleLayers());
        Array<const IntBuffer*> targetCIs;

        if (l == 0) {
            for (int i = 0; i < ioSizes.size(); i++)
                eInputCIs[i] = inputCIs[i];

            if (eInputCIs.size() > ioSizes.size())
                eInputCIs[ioSizes.size()] = &hiddenCIsPrev[l];

            targetCIs.resize(dLayers[l].size());

            for (int d = 0; d < dLayers[l].size(); d++)
                targetCIs[d] = inputCIs[iIndices[d]];
        }
        else {
            eInputCIs[0] = &eLayers[l - 1].getHiddenCIs();

            if (eInputCIs.size() > 1)
                eInputCIs[1] = &hiddenCIsPrev[l];

            targetCIs.resize(1);
            targetCIs[0] = &eLayers[l - 1].getHiddenCIs();
        }

        if (learnEnabled) {
            errors[l].fill(0.0f);

            for (int d = 0; d < dLayers[l].size(); d++)
                dLayers[l][d].generateErrors(targetCIs[d], &errors[l], 0);
        }

        // Activate sparse coder
        eLayers[l].step(eInputCIs, &errors[l], learnEnabled);
    }

    // Backward
    for (int l = dLayers.size() - 1; l >= 0; l--) {
        Array<const IntBuffer*> dInputCIs(l < eLayers.size() - 1 ? 2 : 1);
        Array<const FloatBuffer*> dInputActs(dInputCIs.size());

        dInputCIs[0] = &eLayers[l].getHiddenCIs();
        dInputActs[0] = &eLayers[l].getHiddenActs();
        
        if (l < eLayers.size() - 1) {
            dInputCIs[1] = &dLayers[l + 1][0].getHiddenCIs();
            dInputActs[1] = nullptr;
        }

        Array<const IntBuffer*> targetCIs;

        if (l == 0) {
            targetCIs.resize(dLayers[l].size());

            for (int d = 0; d < dLayers[l].size(); d++)
                targetCIs[d] = inputCIs[iIndices[d]];
        }
        else {
            targetCIs.resize(1);
            targetCIs[0] = &eLayers[l - 1].getHiddenCIs();
        }

        if (learnEnabled)
            errors[l].fill(0.0f);

        if (l == 0) {
            for (int d = 0; d < aLayers.size(); d++)
                aLayers[d].step(dInputCIs, inputCIs[iIndices[d + ioSizes.size()]], reward, learnEnabled, mimic);
        }

        for (int d = 0; d < dLayers[l].size(); d++) {
            if (learnEnabled) {
                dLayers[l][d].generateErrors(targetCIs[d], &errors[l], 0);

                dLayers[l][d].learn(targetCIs[d]);
            }

            dLayers[l][d].activate(dInputCIs, dInputActs);
        }
    }
}

int Hierarchy::size() const {
    int size = 4 * sizeof(int) + ioSizes.size() * sizeof(Int3) + iIndices.size() * sizeof(int) + dIndices.size() * sizeof(int);

    for (int l = 0; l < eLayers.size(); l++) {
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
    int size = 0;

    for (int l = 0; l < eLayers.size(); l++) {
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

    writer.write(reinterpret_cast<const void*>(&iIndices[0]), iIndices.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&dIndices[0]), dIndices.size() * sizeof(int));

    for (int l = 0; l < numLayers; l++) {
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

    iIndices.resize(numIO * 2);
    dIndices.resize(numIO);

    reader.read(reinterpret_cast<void*>(&iIndices[0]), iIndices.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&dIndices[0]), dIndices.size() * sizeof(int));
    
    for (int l = 0; l < numLayers; l++) {
        eLayers[l].read(reader);
        
        dLayers[l].resize(l == 0 ? numPredictions : 1);

        // Decoders
        for (int d = 0; d < dLayers[l].size(); d++)
            dLayers[l][d].read(reader);

        errors[l] = FloatBuffer(eLayers[l].getHiddenActs().size(), 0.0f);
    }

    aLayers.resize(numActions);

    // Actors
    for (int d = 0; d < aLayers.size(); d++)
        aLayers[d].read(reader);
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

    for (int d = 0; d < aLayers.size(); d++)
        aLayers[d].writeState(writer);
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

    // Actors
    for (int d = 0; d < aLayers.size(); d++)
        aLayers[d].readState(reader);
}
