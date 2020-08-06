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

    float maxProb = -999999.0f;

    for (int zi = 0; zi < supportSize; zi++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, zi), Int3(hiddenSize.x, hiddenSize.y, supportSize));

        float sum = 0.0f;
        int count = 0;

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int diam = vld.radius * 2 + 1;

            // Projection
            Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
                static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));

            Int2 visibleCenter = project(pos, hToV);

            // Lower corner
            Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

            // Bounds of receptive field, clamped to input size
            Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
            Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                    unsigned char inC = (*inputCs[vli])[visibleColumnIndex];

                    if (inC != 0) {
                        Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                        float weight = vl.valueWeights[inC + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex))];
                        
                        sum += weight;
                        count++;
                    }
                }
        }

        sum /= max(1, count);

        hiddenProbs[hiddenIndex] = sum;

        maxProb = max(maxProb, sum);
    }

    float total = 0.0f;

    for (int zi = 0; zi < supportSize; zi++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, zi), Int3(hiddenSize.x, hiddenSize.y, supportSize));

        hiddenProbs[hiddenIndex] = expf(hiddenProbs[hiddenIndex] - maxProb);

        total += hiddenProbs[hiddenIndex];
    }

    float scale = 1.0f / max(0.0001f, total);

    float value = 0.0f;

    for (int zi = 0; zi < supportSize; zi++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, zi), Int3(hiddenSize.x, hiddenSize.y, supportSize));

        hiddenProbs[hiddenIndex] *= scale;

        float supportValue = static_cast<float>(zi) / static_cast<float>(supportSize - 1) * 2.0f - 1.0f;

        value += supportValue * hiddenProbs[hiddenIndex];
    }

    // --- Action ---

    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float sum = 0.0f;
        int count = 0;

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int diam = vld.radius * 2 + 1;

            // Projection
            Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
                static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));

            Int2 visibleCenter = project(pos, hToV);

            // Lower corner
            Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

            // Bounds of receptive field, clamped to input size
            Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
            Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                    unsigned char inC = (*inputCs[vli])[visibleColumnIndex];

                    if (inC != 0) {
                        Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                        float weight = vl.actionWeights[inC - 1 + (vld.size.z - 1) * (offset.y + diam * (offset.x + diam * hiddenIndex))];
                        
                        sum += weight;
                        count++;
                    }
                }
        }

        sum /= max(1, count);

        hiddenActivations[hiddenIndex] = sum;

        maxActivation = max(maxActivation, sum);
    }

    total = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        hiddenActivations[hiddenIndex] = expf(hiddenActivations[hiddenIndex] - maxActivation);
        
        total += hiddenActivations[hiddenIndex];
    }

    float cusp = randf(state) * total;

    int selectIndex = 0;
    float sumSoFar = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        sumSoFar += hiddenActivations[hiddenIndex];

        if (sumSoFar >= cusp) {
            selectIndex = hc;

            break;
        }
    }
    
    hiddenCs[hiddenColumnIndex] = selectIndex;
}

void Actor::learn(
    const Int2 &pos,
    const Array<const ByteBuffer*> &inputCs,
    const ByteBuffer* hiddenTargetCsPrev,
    float q,
    float g,
    bool mimic
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    float supportDelta = 2.0f / static_cast<float>(supportSize - 1);

    // Zero accum
    for (int zi = 0; zi < supportSize; zi++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, zi), Int3(hiddenSize.x, hiddenSize.y, supportSize));

        hiddenTargetProbs[hiddenIndex] = 0.0f;
    }

    for (int zi = 0; zi < supportSize; zi++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, zi), Int3(hiddenSize.x, hiddenSize.y, supportSize));

        float supportValue = static_cast<float>(zi) / static_cast<float>(supportSize - 1) * 2.0f - 1.0f;

        float tz = min(1.0f, max(-1.0f, (1.0f - g) * q + g * supportValue));

        float b = (tz + 1.0f) / supportDelta;

        int lower = floorf(b);
        int upper = ceilf(b);

        // Distribute
        hiddenTargetProbs[address3(Int3(pos.x, pos.y, lower), Int3(hiddenSize.x, hiddenSize.y, supportSize))] += hiddenProbs[hiddenIndex] * (upper - b);
        hiddenTargetProbs[address3(Int3(pos.x, pos.y, upper), Int3(hiddenSize.x, hiddenSize.y, supportSize))] += hiddenProbs[hiddenIndex] * (b - lower);
    }

    float maxProb = -999999.0f;

    for (int zi = 0; zi < supportSize; zi++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, zi), Int3(hiddenSize.x, hiddenSize.y, supportSize));

        float sum = 0.0f;
        int count = 0;

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int diam = vld.radius * 2 + 1;

            // Projection
            Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
                static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));

            Int2 visibleCenter = project(pos, hToV);

            // Lower corner
            Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

            // Bounds of receptive field, clamped to input size
            Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
            Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                    unsigned char inC = (*inputCs[vli])[visibleColumnIndex];

                    if (inC != 0) {
                        Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                        float weight = vl.valueWeights[inC - 1 + (vld.size.z - 1) * (offset.y + diam * (offset.x + diam * hiddenIndex))];
                        
                        sum += weight;
                        count++;
                    }
                }
        }

        sum /= max(1, count);

        hiddenProbsTemp[hiddenIndex] = sum;

        maxProb = max(maxProb, sum);
    }

    float total = 0.0f;

    for (int zi = 0; zi < supportSize; zi++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, zi), Int3(hiddenSize.x, hiddenSize.y, supportSize));

        hiddenProbsTemp[hiddenIndex] = expf(hiddenProbsTemp[hiddenIndex] - maxProb);

        total += hiddenProbsTemp[hiddenIndex];
    }

    float scale = 1.0f / max(0.0001f, total);

    float newValue = 0.0f;
    float value = 0.0f;

    for (int zi = 0; zi < supportSize; zi++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, zi), Int3(hiddenSize.x, hiddenSize.y, supportSize));

        hiddenProbsTemp[hiddenIndex] *= scale;

        float supportValue = static_cast<float>(zi) / static_cast<float>(supportSize - 1) * 2.0f - 1.0f;

        newValue += supportValue * hiddenTargetProbs[hiddenIndex];
        value += supportValue * hiddenProbsTemp[hiddenIndex];
    }

    for (int zi = 0; zi < supportSize; zi++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, zi), Int3(hiddenSize.x, hiddenSize.y, supportSize));

        float delta = alpha * (hiddenTargetProbs[hiddenIndex] - hiddenProbsTemp[hiddenIndex]);

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int diam = vld.radius * 2 + 1;

            // Projection
            Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
                static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));

            Int2 visibleCenter = project(pos, hToV);

            // Lower corner
            Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

            // Bounds of receptive field, clamped to input size
            Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
            Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                    unsigned char inC = (*inputCs[vli])[visibleColumnIndex];

                    if (inC != 0) {
                        Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                        vl.valueWeights[inC - 1 + (vld.size.z - 1) * (offset.y + diam * (offset.x + diam * hiddenIndex))] += delta;
                    }
                }
        }
    }

    // --- Action ---

    float tdError = newValue - value;

    int targetC = (*hiddenTargetCsPrev)[hiddenColumnIndex];

    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float sum = 0.0f;
        int count = 0;

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int diam = vld.radius * 2 + 1;

            // Projection
            Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
                static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));

            Int2 visibleCenter = project(pos, hToV);

            // Lower corner
            Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

            // Bounds of receptive field, clamped to input size
            Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
            Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                    unsigned char inC = (*inputCs[vli])[visibleColumnIndex];

                    if (inC != 0) {
                        Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                        float weight = vl.actionWeights[inC - 1 + (vld.size.z - 1) * (offset.y + diam * (offset.x + diam * hiddenIndex))];
                        
                        sum += weight;
                        count++;
                    }
                }
        }

        hiddenActivations[hiddenIndex] = sum / static_cast<float>(max(1, count));

        maxActivation = max(maxActivation, hiddenActivations[hiddenIndex]);
    }

    total = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        hiddenActivations[hiddenIndex] = expf(hiddenActivations[hiddenIndex] - maxActivation);
        
        total += hiddenActivations[hiddenIndex];
    }
    
    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float deltaAction = (mimic ? beta : (tdError > 0.0f ? beta : -beta)) * ((hc == targetC ? 1.0f : 0.0f) - hiddenActivations[hiddenIndex] / max(0.0001f, total));

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int diam = vld.radius * 2 + 1;

            // Projection
            Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
                static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));

            Int2 visibleCenter = project(pos, hToV);

            // Lower corner
            Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

            // Bounds of receptive field, clamped to input size
            Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
            Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                    unsigned char inC = (*inputCs[vli])[visibleColumnIndex];

                    if (inC != 0) {
                        Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                        vl.actionWeights[inC - 1 + (vld.size.z - 1) * (offset.y + diam * (offset.x + diam * hiddenIndex))] += deltaAction;
                    }
                }
        }
    }
}

void Actor::initRandom(
    const Int3 &hiddenSize,
    int historyCapacity,
    int supportSize,
    const Array<VisibleLayerDesc> &visibleLayerDescs
) {
    this->visibleLayerDescs = visibleLayerDescs;

    this->hiddenSize = hiddenSize;

    this->supportSize = supportSize;

    visibleLayers.resize(visibleLayerDescs.size());

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;

    // Create layers
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        // Create weight matrix for this visible layer and initialize randomly
        vl.valueWeights.resize(numHiddenColumns * supportSize * area * (vld.size.z - 1));
        vl.actionWeights.resize(numHidden * area * (vld.size.z - 1));

        for (int i = 0; i < vl.valueWeights.size(); i++)
            vl.valueWeights[i] = randf(-0.01f, 0.01f);

        for (int i = 0; i < vl.actionWeights.size(); i++)
            vl.actionWeights[i] = randf(-0.01f, 0.01f);
    }

    hiddenProbs = FloatBuffer(numHiddenColumns * supportSize, 0.0f);
    hiddenProbsTemp = FloatBuffer(numHiddenColumns * supportSize, 0.0f);
    hiddenTargetProbs = FloatBuffer(numHiddenColumns * supportSize, 0.0f);
    hiddenActivations = FloatBuffer(numHidden, 0.0f);

    hiddenCs = ByteBuffer(numHiddenColumns, 0);

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

        s.reward = reward;
    }

    // Learn (if have sufficient samples)
    if (learnEnabled && historySize == historySamples.size()) {
        const HistorySample &sPrev = historySamples[historySize - 1];
        const HistorySample &s = historySamples[historySize - 2];

        // Compute (partial) values, rest is completed in the kernel
        float q = 0.0f;
        float g = 1.0f;

        for (int t = historySize - 2; t >= 0; t--) {
            q += historySamples[t].reward * g;

            g *= gamma;
        }

        q /= (historySize - 2);

        #pragma omp parallel for
        for (int i = 0; i < numHiddenColumns; i++)
            learn(Int2(i / hiddenSize.y, i % hiddenSize.y), constGet(sPrev.inputCs), &s.hiddenTargetCsPrev, q, g, mimic);
    }
}

void Actor::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));
    writer.write(reinterpret_cast<const void*>(&supportSize), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&alpha), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&beta), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&gamma), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenCs[0]), hiddenCs.size() * sizeof(unsigned char));

    int numVisibleLayers = visibleLayers.size();

    writer.write(reinterpret_cast<const void*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

        int valueWeightsSize = vl.valueWeights.size();

        writer.write(reinterpret_cast<const void*>(&valueWeightsSize), sizeof(int));

        int actionWeightsSize = vl.actionWeights.size();

        writer.write(reinterpret_cast<const void*>(&actionWeightsSize), sizeof(int));

        writer.write(reinterpret_cast<const void*>(&vl.valueWeights[0]), vl.valueWeights.size() * sizeof(float));
        writer.write(reinterpret_cast<const void*>(&vl.actionWeights[0]), vl.actionWeights.size() * sizeof(float));
    }

    writer.write(reinterpret_cast<const void*>(&historySize), sizeof(int));

    int numHistorySamples = historySamples.size();

    writer.write(reinterpret_cast<const void*>(&numHistorySamples), sizeof(int));

    int historyStart = historySamples.start;

    writer.write(reinterpret_cast<const void*>(&historyStart), sizeof(int));

    for (int t = 0; t < historySamples.size(); t++) {
        const HistorySample &s = historySamples[t];

        for (int vli = 0; vli < visibleLayers.size(); vli++)
            writer.write(reinterpret_cast<const void*>(&s.inputCs[vli][0]), s.inputCs[vli].size() * sizeof(unsigned char));

        writer.write(reinterpret_cast<const void*>(&s.hiddenTargetCsPrev[0]), s.hiddenTargetCsPrev.size() * sizeof(unsigned char));

        writer.write(reinterpret_cast<const void*>(&s.reward), sizeof(float));
    }
}

void Actor::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));
    reader.read(reinterpret_cast<void*>(&supportSize), sizeof(int));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;
    
    reader.read(reinterpret_cast<void*>(&alpha), sizeof(float));
    reader.read(reinterpret_cast<void*>(&beta), sizeof(float));
    reader.read(reinterpret_cast<void*>(&gamma), sizeof(float));

    hiddenCs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCs[0]), hiddenCs.size() * sizeof(unsigned char));

    hiddenProbs = FloatBuffer(numHiddenColumns * supportSize, 0.0f);
    hiddenProbsTemp = FloatBuffer(numHiddenColumns * supportSize, 0.0f);
    hiddenTargetProbs = FloatBuffer(numHiddenColumns * supportSize, 0.0f);
    hiddenActivations = FloatBuffer(numHidden, 0.0f);
    
    int numVisibleLayers = visibleLayers.size();

    reader.read(reinterpret_cast<void*>(&numVisibleLayers), sizeof(int));

    visibleLayers.resize(numVisibleLayers);
    visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = visibleLayerDescs[vli];

        reader.read(reinterpret_cast<void*>(&vld), sizeof(VisibleLayerDesc));

        int valueWeightsSize;

        reader.read(reinterpret_cast<void*>(&valueWeightsSize), sizeof(int));

        int actionWeightsSize;

        reader.read(reinterpret_cast<void*>(&actionWeightsSize), sizeof(int));

        vl.valueWeights.resize(valueWeightsSize);
        vl.actionWeights.resize(actionWeightsSize);

        reader.read(reinterpret_cast<void*>(&vl.valueWeights[0]), vl.valueWeights.size() * sizeof(float));
        reader.read(reinterpret_cast<void*>(&vl.actionWeights[0]), vl.actionWeights.size() * sizeof(float));
    }

    reader.read(reinterpret_cast<void*>(&historySize), sizeof(int));

    int numHistorySamples;

    reader.read(reinterpret_cast<void*>(&numHistorySamples), sizeof(int));

    int historyStart;

    reader.read(reinterpret_cast<void*>(&historyStart), sizeof(int));

    historySamples.resize(numHistorySamples);
    historySamples.start = historyStart;

    for (int t = 0; t < historySamples.size(); t++) {
        HistorySample &s = historySamples[t];

        s.inputCs.resize(numVisibleLayers);

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int numVisibleColumns = vld.size.x * vld.size.y;

            s.inputCs[vli].resize(numVisibleColumns);

            reader.read(reinterpret_cast<void*>(&s.inputCs[vli][0]), s.inputCs[vli].size() * sizeof(unsigned char));
        }

        s.hiddenTargetCsPrev.resize(numHiddenColumns);

        reader.read(reinterpret_cast<void*>(&s.hiddenTargetCsPrev[0]), s.hiddenTargetCsPrev.size() * sizeof(unsigned char));

        reader.read(reinterpret_cast<void*>(&s.reward), sizeof(float));
    }
}