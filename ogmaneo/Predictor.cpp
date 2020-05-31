// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Predictor.h"

using namespace ogmaneo;

void Predictor::forward(
    const Int2 &pos,
    const Array<const IntBuffer*> &inputCs
) {
    int maxIndex = 0;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float sum = 0.0f;

        // For each visible layer
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            sum += vl.weights.multiplyOHVs(*inputCs[vli], hiddenIndex, vld.size.z);
        }

        if (sum > maxActivation) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    hiddenCs[address2(pos, Int2(hiddenSize.x, hiddenSize.y))] = maxIndex;
}

void Predictor::learn(
    const Int2 &pos,
    const IntBuffer* hiddenTargetCs
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    int targetC = (*hiddenTargetCs)[hiddenColumnIndex];

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float sum = 0.0f;
        int count = 0;

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            sum += vl.weights.multiplyOHVs(vl.inputCsPrev, hiddenIndex, vld.size.z);
            count += vl.weights.count(hiddenIndex) / vld.size.z;
        }

        sum /= count;

        float delta = alpha * ((hc == targetC ? 1.0f : 0.0f) - sigmoid(sum));

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            vl.weights.deltaOHVs(vl.inputCsPrev, delta, hiddenIndex, vld.size.z);
        }
    }
}

void Predictor::initRandom(
    const Int3 &hiddenSize,
    const Array<VisibleLayerDesc> &visibleLayerDescs
) {
    this->visibleLayerDescs = visibleLayerDescs;

    this->hiddenSize = hiddenSize;

    visibleLayers.resize(visibleLayerDescs.size());

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    // Create layers
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;

        // Create weight matrix for this visible layer and initialize randomly
        initSMLocalRF(vld.size, hiddenSize, vld.radius, vl.weights);

        for (int i = 0; i < vl.weights.nonZeroValues.size(); i++)
            vl.weights.nonZeroValues[i] = randf(-0.01f, 0.01f);

        vl.inputCsPrev = IntBuffer(numVisibleColumns, 0);
    }

    // Hidden Cs
    hiddenCs = IntBuffer(numHiddenColumns, 0);
}

void Predictor::activate(
    const Array<const IntBuffer*> &inputCs
) {
    // Forward kernel
    for (int x = 0; x < hiddenSize.x; x++)
        for (int y = 0; y < hiddenSize.y; y++)
            forward(Int2(x, y), inputCs);

    // Copy to prevs
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];

        copyInt(inputCs[vli], &vl.inputCsPrev);
    }
}

void Predictor::learn(
    const IntBuffer* hiddenTargetCs
) {
    // Learn kernel
    for (int x = 0; x < hiddenSize.x; x++)
        for (int y = 0; y < hiddenSize.y; y++)
            learn(Int2(x, y), hiddenTargetCs);
}