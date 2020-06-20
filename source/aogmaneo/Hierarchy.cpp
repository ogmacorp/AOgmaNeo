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
    const Array<Int3> &inputSizes,
    const Array<InputType> &inputTypes,
    const Array<LayerDesc> &layerDescs
) {
    // Create layers
    scLayers.resize(layerDescs.size());
    pLayers.resize(layerDescs.size());

    ticks.resize(layerDescs.size(), 0);

    histories.resize(layerDescs.size());
    
    ticksPerUpdate.resize(layerDescs.size());

    // Default update state is no update
    updates.resize(layerDescs.size(), false);

    // Cache input sizes
    this->inputSizes = inputSizes;

    // Determine ticks per update, first layer is always 1
    for (int l = 0; l < layerDescs.size(); l++)
        ticksPerUpdate[l] = l == 0 ? 1 : layerDescs[l].ticksPerUpdate; // First layer always 1

    // Iterate through layers
    for (int l = 0; l < layerDescs.size(); l++) {
        // Create sparse coder visible layer descriptors
        Array<SparseCoder::VisibleLayerDesc> scVisibleLayerDescs;

        // If first layer
        if (l == 0) {
            scVisibleLayerDescs.resize(inputSizes.size() * layerDescs[l].temporalHorizon);

            for (int i = 0; i < inputSizes.size(); i++) {
                for (int t = 0; t < layerDescs[l].temporalHorizon; t++) {
                    int index = t + layerDescs[l].temporalHorizon * i;

                    scVisibleLayerDescs[index].size = inputSizes[i];
                    scVisibleLayerDescs[index].radius = layerDescs[l].ffRadius;
                }
            }
            
            // Initialize history buffers
            histories[l].resize(inputSizes.size());

			for (int i = 0; i < histories[l].size(); i++) {
                int inSize = inputSizes[i].x * inputSizes[i].y;

                histories[l][i].resize(layerDescs[l].temporalHorizon);
				
                for (int t = 0; t < histories[l][i].size(); t++)
				    histories[l][i][t] = ByteBuffer(inSize, 0);
			}

            // Predictors
            pLayers[l].resize(inputSizes.size());
            aLayers.resize(inputSizes.size());

            // Predictor visible layer descriptors
            Array<Predictor::VisibleLayerDesc> pVisibleLayerDescs(l < scLayers.size() - 1 ? 2 : 1);

            pVisibleLayerDescs[0].size = layerDescs[l].hiddenSize;
            pVisibleLayerDescs[0].radius = layerDescs[l].pRadius;

            if (l < scLayers.size() - 1)
                pVisibleLayerDescs[1] = pVisibleLayerDescs[0];

            // Actor visible layer descriptors
            Array<Actor::VisibleLayerDesc> aVisibleLayerDescs(l < scLayers.size() - 1 ? 2 : 1);

            aVisibleLayerDescs[0].size = layerDescs[l].hiddenSize;
            aVisibleLayerDescs[0].radius = layerDescs[l].aRadius;

            if (l < scLayers.size() - 1)
                aVisibleLayerDescs[1] = aVisibleLayerDescs[0];

            // Create actors
            for (int p = 0; p < pLayers[l].size(); p++) {
                if (inputTypes[p] == InputType::prediction) {
                    pLayers[l][p].make();

                    pLayers[l][p]->initRandom(inputSizes[p], pVisibleLayerDescs);
                }
                else if (inputTypes[p] == InputType::action) {
                    aLayers[p].make();

                    aLayers[p]->initRandom(inputSizes[p], layerDescs[l].historyCapacity, aVisibleLayerDescs);
                }
            }
        }
        else {
            scVisibleLayerDescs.resize(layerDescs[l].temporalHorizon);

            for (int t = 0; t < layerDescs[l].temporalHorizon; t++) {
                scVisibleLayerDescs[t].size = layerDescs[l - 1].hiddenSize;
                scVisibleLayerDescs[t].radius = layerDescs[l].ffRadius;
            }

            histories[l].resize(1);

            int inSize = layerDescs[l - 1].hiddenSize.x * layerDescs[l - 1].hiddenSize.y;

            histories[l][0].resize(layerDescs[l].temporalHorizon);

			for (int t = 0; t < histories[l][0].size(); t++)
                histories[l][0][t] = ByteBuffer(inSize, 0);

            pLayers[l].resize(layerDescs[l].ticksPerUpdate);

            // Predictor visible layer descriptors
            Array<Predictor::VisibleLayerDesc> pVisibleLayerDescs(l < scLayers.size() - 1 ? 2 : 1);

            pVisibleLayerDescs[0].size = layerDescs[l].hiddenSize;
            pVisibleLayerDescs[0].radius = layerDescs[l].pRadius;

            if (l < scLayers.size() - 1)
                pVisibleLayerDescs[1] = pVisibleLayerDescs[0];

            // Create actors
            for (int p = 0; p < pLayers[l].size(); p++) {
                pLayers[l][p].make();

                pLayers[l][p]->initRandom(layerDescs[l - 1].hiddenSize, pVisibleLayerDescs);
            }
        }
		
        // Create the sparse coding layer
        scLayers[l].initRandom(layerDescs[l].hiddenSize, scVisibleLayerDescs);
    }
}

const Hierarchy &Hierarchy::operator=(
    const Hierarchy &other
) {
    // Layers
    scLayers = other.scLayers;

    updates = other.updates;
    ticks = other.ticks;
    ticksPerUpdate = other.ticksPerUpdate;
    inputSizes = other.inputSizes;
    histories = other.histories;

    pLayers.resize(other.pLayers.size());
    
    for (int l = 0; l < scLayers.size(); l++) {
        pLayers[l].resize(other.pLayers[l].size());

        for (int v = 0; v < pLayers[l].size(); v++) {
            if (other.pLayers[l][v] != nullptr) {
                pLayers[l][v].make();

                (*pLayers[l][v]) = (*other.pLayers[l][v]);
            }
            else
                pLayers[l][v] = nullptr;
        }
    }

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

void Hierarchy::step(
    const Array<const ByteBuffer*> &inputCs,
    bool learnEnabled,
    float reward,
    bool mimic
) {
    // First tick is always 0
    ticks[0] = 0;

    // Add input to first layer history   
    for (int i = 0; i < inputSizes.size(); i++) {
        histories[0][i].pushFront();

        // Copy
        histories[0][i][0] = *inputCs[i];
    }

    // Set all updates to no update, will be set to true if an update occurred later
    for (int i = 0; i < updates.size(); i++)
        updates[i] = false;

    // Forward
    for (int l = 0; l < scLayers.size(); l++) {
        // If is time for layer to tick
        if (l == 0 || ticks[l] >= ticksPerUpdate[l]) {
            // Reset tick
            ticks[l] = 0;

            // Updated
            updates[l] = true;

            Array<const ByteBuffer*> layerInputCs(histories[l].size() * histories[l][0].size());

            int index = 0;

            for (int i = 0; i < histories[l].size(); i++) {
                for (int t = 0; t < histories[l][i].size(); t++)
                    layerInputCs[index++] = &histories[l][i][t];
            }

            // Activate sparse coder
            scLayers[l].step(layerInputCs, learnEnabled);

            // Add to next layer's history
            if (l < scLayers.size() - 1) {
                int lNext = l + 1;

                histories[lNext][0].pushFront();

                // Copy
                histories[lNext][0][0] = scLayers[l].getHiddenCs();

                ticks[lNext]++;
            }
        }
    }

    // Backward
    for (int l = scLayers.size() - 1; l >= 0; l--) {
        if (updates[l]) {
            // Feed back is current layer state and next higher layer prediction
            Array<const ByteBuffer*> feedBackCs(l < scLayers.size() - 1 ? 2 : 1);

            feedBackCs[0] = &scLayers[l].getHiddenCs();

            if (l < scLayers.size() - 1)
                feedBackCs[1] = &pLayers[l + 1][ticksPerUpdate[l + 1] - 1 - ticks[l + 1]]->getHiddenCs();

            // Step actor layers
            for (int p = 0; p < pLayers[l].size(); p++) {
                if (pLayers[l][p] != nullptr) {
                    if (learnEnabled)
                        pLayers[l][p]->learn(l == 0 ? &histories[l][p][0] : &histories[l][0][p]);

                    pLayers[l][p]->activate(feedBackCs);
                }
            }

            if (l == 0) {
                // Step actors
                for (int p = 0; p < aLayers.size(); p++) {
                    if (aLayers[p] != nullptr)
                        aLayers[p]->step(feedBackCs, &histories[l][p][0], reward, learnEnabled, mimic);
                }
            }
        }
    }
}

void Hierarchy::getState(
    State &state
) const {
    int numLayers = scLayers.size();

    state.hiddenCs.resize(numLayers);
    state.hiddenCsPrev.resize(numLayers);
    state.histories.resize(numLayers);
    state.predHiddenCs.resize(numLayers);
    state.predInputCsPrev.resize(numLayers);

    for (int l = 0; l < numLayers; l++) {
        state.hiddenCs[l] = scLayers[l].getHiddenCs();
        state.hiddenCsPrev[l] = scLayers[l].getHiddenCsPrev();

        state.predHiddenCs[l].resize(pLayers[l].size());
        state.predInputCsPrev[l].resize(pLayers[l].size());

        for (int j = 0; j < pLayers[l].size(); j++) {
            state.predHiddenCs[l][j] = pLayers[l][j]->getHiddenCs();

            state.predInputCsPrev[l][j].resize(pLayers[l][j]->getNumVisibleLayers());

            for (int v = 0; v < pLayers[l][j]->getNumVisibleLayers(); v++) {
                state.predInputCsPrev[l][j][v] = pLayers[l][j]->getVisibleLayer(v).inputCsPrev;
            }
        }
    }

    state.histories = histories;
    state.ticks = ticks;
    state.updates = updates;
}

void Hierarchy::setState(
    const State &state
) {
    int numLayers = scLayers.size();

    for (int l = 0; l < numLayers; l++) {
        scLayers[l].hiddenCs = state.hiddenCs[l];
        scLayers[l].hiddenCsPrev = state.hiddenCsPrev[l];

        for (int j = 0; j < pLayers[l].size(); j++) {
            pLayers[l][j]->hiddenCs = state.predHiddenCs[l][j];

            for (int v = 0; v < pLayers[l][j]->getNumVisibleLayers(); v++) {
                pLayers[l][j]->visibleLayers[v].inputCsPrev = state.predInputCsPrev[l][j][v];
            }
        }
    }

    histories = state.histories;
    ticks = state.ticks;
    updates = state.updates;
}