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

    inputSizes = other.inputSizes;

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

    // Cache input sizes
    this->inputSizes = inputSizes;

    // Iterate through layers
    for (int l = 0; l < layerDescs.size(); l++) {
        // Create sparse coder visible layer descriptors
        Array<SparseCoder::VisibleLayerDesc> scVisibleLayerDescs;

        // If first layer
        if (l == 0) {
            scVisibleLayerDescs.resize(inputSizes.size() + 1); // +1 for recurrent

            for (int i = 0; i < inputSizes.size(); i++) {
                scVisibleLayerDescs[i].size = inputSizes[i];
                scVisibleLayerDescs[i].radius = layerDescs[l].ffRadius;
            }

            scVisibleLayerDescs[inputSizes.size()].size = layerDescs[l].hiddenSize;
            scVisibleLayerDescs[inputSizes.size()].radius = layerDescs[l].rRadius;

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
            scVisibleLayerDescs.resize(2);

            scVisibleLayerDescs[0].size = layerDescs[l - 1].hiddenSize;
            scVisibleLayerDescs[0].radius = layerDescs[l].ffRadius;

            scVisibleLayerDescs[1].size = layerDescs[l].hiddenSize;
            scVisibleLayerDescs[1].radius = layerDescs[l].rRadius;

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
    // Forward
    for (int l = 0; l < scLayers.size(); l++) {
        Array<const IntBuffer*> layerInputCs(scLayers[l].getNumVisibleLayers());

        if (l == 0) {
            for (int i = 0; i < inputSizes.size(); i++)
                layerInputCs[i] = inputCs[i];
        }
        else
            layerInputCs[0] = &scLayers[l - 1].getHiddenCs();

        layerInputCs[layerInputCs.size() - 1] = &scLayers[l].getHiddenCsPrev();

        // Activate sparse coder
        scLayers[l].step(layerInputCs, &pLayers[l].getHiddenActivations(), learnEnabled);
    }

    // Backward
    for (int l = scLayers.size() - 1; l >= 0; l--) {
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
                        ipLayers[i]->learn(inputCs[i]);

                    ipLayers[i]->activate(feedBackCs);
                }

                if (aLayers[i] != nullptr)
                    aLayers[i]->step(feedBackCs, inputCs[i], reward, learnEnabled, mimic);
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

    for (int l = 0; l < numLayers; l++) {
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

    for (int l = 0; l < numLayers; l++) {
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
