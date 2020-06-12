// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Actor.h"

using namespace aon;

void Actor::forward(
    const Int2 &pos,
    const Array<const ByteBuffer*> &inputCs,
    unsigned long* state
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    // --- Value ---

    float value = 0.0f;
    int count = 0;

    // For each visible layer
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        value += vl.valueWeights.multiplyOHVs(*inputCs[vli], hiddenColumnIndex, vld.size.z);
        count += vl.valueWeights.count(hiddenColumnIndex) / vld.size.z;
    }

    hiddenValues[hiddenColumnIndex] = value / count;

    // --- Action ---

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float sum = 0.0f;

        // For each visible layer
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            sum += vl.actionWeights.multiplyOHVs(*inputCs[vli], hiddenIndex, vld.size.z);
        }

        if (sum > maxActivation) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }
    
    hiddenCs[hiddenColumnIndex] = maxIndex;
}

void Actor::learn(
    const Int2 &pos,
    const Array<const ByteBuffer*> &inputCsPrev,
    const ByteBuffer* hiddenTargetCsPrev,
    const FloatBuffer* hiddenValuesPrev,
    float q,
    float g,
    bool mimic
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    // --- Value Prev ---

    float newValue = q + g * hiddenValues[hiddenColumnIndex];

    float value = 0.0f;
    int count = 0;

    // For each visible layer
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        value += vl.valueWeights.multiplyOHVs(*inputCsPrev[vli], hiddenColumnIndex, vld.size.z);
        count += vl.valueWeights.count(hiddenColumnIndex) / vld.size.z;
    }

    value /= count;

    float tdErrorValue = newValue - value;
    
    float deltaValue = alpha * tdErrorValue;

    // For each visible layer
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        vl.valueWeights.deltaOHVs(*inputCsPrev[vli], deltaValue, hiddenColumnIndex, vld.size.z);
    }

    // --- Action ---

    float tdErrorAction = newValue - (*hiddenValuesPrev)[hiddenColumnIndex];

    int targetC = (*hiddenTargetCsPrev)[hiddenColumnIndex];

    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float sum = 0.0f;

        // For each visible layer
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            sum += vl.actionWeights.multiplyOHVs(*inputCsPrev[vli], hiddenIndex, vld.size.z);
        }

        sum /= count;

        hiddenActivations[hiddenIndex] = sum;

        maxActivation = max(maxActivation, sum);
    }

    float total = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        hiddenActivations[hiddenIndex] = expf(hiddenActivations[hiddenIndex] - maxActivation);
        
        total += hiddenActivations[hiddenIndex];
    }
    
    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float deltaAction = (mimic ? beta : (tdErrorAction > 0.0f ? beta : -beta)) * ((hc == targetC ? 1.0f : 0.0f) - hiddenActivations[hiddenIndex] / max(0.0001f, total));

        // For each visible layer
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            vl.actionWeights.deltaOHVs(*inputCsPrev[vli], deltaAction, hiddenIndex, vld.size.z);
        }
    }
}

void Actor::initRandom(
    const Int3 &hiddenSize,
    int historyCapacity,
    const Array<VisibleLayerDesc> &visibleLayerDescs
) {
    this->visibleLayerDescs = visibleLayerDescs;

    this->hiddenSize = hiddenSize;

    visibleLayers.resize(visibleLayerDescs.size());

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    // Create layers
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

        // Create weight matrix for this visible layer and initialize randomly
        vl.valueWeights.initSMLocalRF(vld.size, Int3(hiddenSize.x, hiddenSize.y, 1), vld.radius);
        vl.actionWeights.initSMLocalRF(vld.size, hiddenSize, vld.radius);

        for (int i = 0; i < vl.valueWeights.nonZeroValues.size(); i++)
            vl.valueWeights.nonZeroValues[i] = randf(-0.01f, 0.01f);

        for (int i = 0; i < vl.actionWeights.nonZeroValues.size(); i++)
            vl.actionWeights.nonZeroValues[i] = randf(-0.01f, 0.01f);
    }

    hiddenActivations = FloatBuffer(numHidden, 0.0f);

    hiddenCs = ByteBuffer(numHiddenColumns, 0);

    hiddenValues = FloatBuffer(numHiddenColumns, 0.0f);

    // Create (pre-allocated) history samples
    historySize = 0;
    historySamples.resize(historyCapacity);

    for (int i = 0; i < historySamples.size(); i++) {
        historySamples[i].inputCs.resize(visibleLayers.size());

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

            int numVisibleColumns = vld.size.x * vld.size.y;

            historySamples[i].inputCs[vli] = ByteBuffer(numVisibleColumns);
        }

        historySamples[i].hiddenTargetCsPrev = ByteBuffer(numHiddenColumns);

        historySamples[i].hiddenValuesPrev = FloatBuffer(numHiddenColumns);
    }
}

void Actor::step(
    const Array<const ByteBuffer*> &inputCs,
    const ByteBuffer* hiddenTargetCsPrev,
    float reward,
    bool learnEnabled,
    bool mimic
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    // Forward kernel
    int baseState = rand();

    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++) {
        unsigned long state = baseState + i;

        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCs, &state);
    }

    historySamples.pushFront();

    // If not at cap, increment
    if (historySize < historySamples.size())
        historySize++;
    
    // Add new sample
    {
        HistorySample &s = historySamples[0];

        for (int vli = 0; vli < visibleLayers.size(); vli++)
            s.inputCs[vli] = *inputCs[vli];

        // Copy hidden Cs
        s.hiddenTargetCsPrev = *hiddenTargetCsPrev;

        // Copy hidden values
        s.hiddenValuesPrev = hiddenValues;

        s.reward = reward;
    }

    // Learn (if have sufficient samples)
    if (learnEnabled && historySize > minSteps + 1) {
        for (int it = 0; it < historyIters; it++) {
            int historyIndex = rand() % (historySize - 1 - minSteps) + minSteps;

            const HistorySample &sPrev = historySamples[historyIndex + 1];
            const HistorySample &s = historySamples[historyIndex];

            // Compute (partial) values, rest is completed in the kernel
            float q = 0.0f;
            float g = 1.0f;

            for (int t = historyIndex; t >= 0; t--) {
                q += historySamples[t].reward * g;

                g *= gamma;
            }

            #pragma omp parallel for
            for (int i = 0; i < numHiddenColumns; i++)
                learn(Int2(i / hiddenSize.y, i % hiddenSize.y), constGet(sPrev.inputCs), &s.hiddenTargetCsPrev, &sPrev.hiddenValuesPrev, q, g, mimic);
        }
    }
}