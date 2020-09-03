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

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                value += vl.valueWeights[inC + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenColumnIndex))];
                count++;
            }
    }

    value /= max(1, count);

    hiddenValues[hiddenColumnIndex] = value;

    // --- Action ---

    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float sum = 0.0f;

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

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    sum += vl.actionWeights[inC + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex))];
                }
        }

        sum /= max(1, count);

        hiddenActivations[hiddenIndex] = sum;

        maxActivation = max(maxActivation, sum);
    }

    float total = 0.0f;

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

                unsigned char inC = (*inputCsPrev[vli])[visibleColumnIndex];

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                value += vl.valueWeights[inC + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenColumnIndex))];
                count++;
            }
    }

    value /= max(1, count);

    float tdErrorValue = newValue - value;
    
    float deltaValue = alpha * tdErrorValue;

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

                unsigned char inC = (*inputCsPrev[vli])[visibleColumnIndex];

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                vl.valueWeights[inC + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenColumnIndex))] += deltaValue;
            }
    }

    // --- Action ---

    float tdErrorAction = newValue - (*hiddenValuesPrev)[hiddenColumnIndex];

    int targetC = (*hiddenTargetCsPrev)[hiddenColumnIndex];

    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float sum = 0.0f;

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

                    unsigned char inC = (*inputCsPrev[vli])[visibleColumnIndex];

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    sum += vl.actionWeights[inC + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex))];
                }
        }

        hiddenActivations[hiddenIndex] = sum / max(1, count);

        maxActivation = max(maxActivation, hiddenActivations[hiddenIndex]);
    }

    float total = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        hiddenActivations[hiddenIndex] = expf(hiddenActivations[hiddenIndex] - maxActivation);
        
        total += hiddenActivations[hiddenIndex];
    }
    
    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float deltaAction = (mimic || tdErrorAction > 0.0f ? beta : -beta) * ((hc == targetC ? 1.0f : 0.0f) - hiddenActivations[hiddenIndex] / max(0.0001f, total));

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

                    unsigned char inC = (*inputCsPrev[vli])[visibleColumnIndex];

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    vl.actionWeights[inC + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex))] += deltaAction;
                }
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

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        // Create weight matrix for this visible layer and initialize randomly
        vl.valueWeights.resize(numHiddenColumns * area * vld.size.z, 0.0f);
        vl.actionWeights.resize(numHidden * area * vld.size.z);

        for (int i = 0; i < vl.actionWeights.size(); i++)
            vl.actionWeights[i] = randf(-0.01f, 0.01f);
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
    unsigned int baseState = rand();

    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++) {
        unsigned long state = baseState + i * 12345;

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

void Actor::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&alpha), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&beta), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&gamma), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&minSteps), sizeof(int));
    writer.write(reinterpret_cast<const void*>(&historyIters), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&hiddenCs[0]), hiddenCs.size() * sizeof(unsigned char));
    writer.write(reinterpret_cast<const void*>(&hiddenValues[0]), hiddenValues.size() * sizeof(float));

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
        writer.write(reinterpret_cast<const void*>(&s.hiddenValuesPrev[0]), s.hiddenValuesPrev.size() * sizeof(float));

        writer.write(reinterpret_cast<const void*>(&s.reward), sizeof(float));
    }
}

void Actor::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;
    
    reader.read(reinterpret_cast<void*>(&alpha), sizeof(float));
    reader.read(reinterpret_cast<void*>(&beta), sizeof(float));
    reader.read(reinterpret_cast<void*>(&gamma), sizeof(float));
    reader.read(reinterpret_cast<void*>(&minSteps), sizeof(int));
    reader.read(reinterpret_cast<void*>(&historyIters), sizeof(int));

    hiddenCs.resize(numHiddenColumns);
    hiddenValues.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCs[0]), hiddenCs.size() * sizeof(unsigned char));
    reader.read(reinterpret_cast<void*>(&hiddenValues[0]), hiddenValues.size() * sizeof(float));

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
        s.hiddenValuesPrev.resize(numHiddenColumns);

        reader.read(reinterpret_cast<void*>(&s.hiddenTargetCsPrev[0]), s.hiddenTargetCsPrev.size() * sizeof(unsigned char));
        reader.read(reinterpret_cast<void*>(&s.hiddenValuesPrev[0]), s.hiddenValuesPrev.size() * sizeof(float));

        reader.read(reinterpret_cast<void*>(&s.reward), sizeof(float));
    }
}
