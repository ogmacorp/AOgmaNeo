// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Actor.h"

using namespace aon;

void Actor::forward(
    const Int2 &columnPos,
    const Array<const IntBuffer*> &inputCIs
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        float sum = 0.0f;

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int diam = vld.radius * 2 + 1;

            // Projection
            Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
                static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));

            Int2 visibleCenter = project(columnPos, hToV);

            // Lower corner
            Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

            // Bounds of receptive field, clamped to input size
            Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
            Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                    int inCI = (*inputCIs[vli])[visibleColumnIndex];

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    sum += vl.weightsAdv[inCI + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex))];
                }
        }

        if (sum > maxActivation || maxIndex == -1) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    hiddenCIs[hiddenColumnIndex] = maxIndex;
}

void Actor::learn(
    const Int2 &columnPos,
    int t
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    float valueNext = 0.0f;
    int count = 0;

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        int diam = vld.radius * 2 + 1;

        // Projection
        Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
            static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));

        Int2 visibleCenter = project(columnPos, hToV);

        // Lower corner
        Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

        // Bounds of receptive field, clamped to input size
        Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
        Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

        count += (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1);

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int inCI = historySamples[t - nSteps].inputCIs[vli][visibleColumnIndex];

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                valueNext += vl.weightsV[inCI + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenColumnIndex))];
            }
    }

    valueNext /= count;

    float minAdvNext = 999999.0f;
    float maxAdvNext = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        float adv = 0.0f;

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int diam = vld.radius * 2 + 1;

            // Projection
            Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
                static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));

            Int2 visibleCenter = project(columnPos, hToV);

            // Lower corner
            Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

            // Bounds of receptive field, clamped to input size
            Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
            Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                    int inCI = historySamples[t - nSteps].inputCIs[vli][visibleColumnIndex];

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    adv += vl.weightsAdv[inCI + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex))];
                }
        }

        adv /= count;

        minAdvNext = min(minAdvNext, adv);
        maxAdvNext = max(maxAdvNext, adv);
    }

    float qNext = valueNext + maxAdvNext - minAdvNext;

    float tdError;
    float minAdv;
    float maxAdv;
    float gae = 0.0f;
    float div = 0.0f;

    for (int n = nSteps - 1; n >= 0; n--) {
        float value = 0.0f;

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int diam = vld.radius * 2 + 1;

            // Projection
            Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
                static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));

            Int2 visibleCenter = project(columnPos, hToV);

            // Lower corner
            Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

            // Bounds of receptive field, clamped to input size
            Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
            Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                    int inCI = historySamples[t - n].inputCIs[vli][visibleColumnIndex];

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    value += vl.weightsV[inCI + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenColumnIndex))];
                }
        }

        value /= count;

        minAdv = 999999.0f;
        maxAdv = -999999.0f;

        for (int hc = 0; hc < hiddenSize.z; hc++) {
            int hiddenCellIndex = hc + hiddenCellsStart;

            float adv = 0.0f;

            for (int vli = 0; vli < visibleLayers.size(); vli++) {
                VisibleLayer &vl = visibleLayers[vli];
                const VisibleLayerDesc &vld = visibleLayerDescs[vli];

                int diam = vld.radius * 2 + 1;

                // Projection
                Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
                    static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));

                Int2 visibleCenter = project(columnPos, hToV);

                // Lower corner
                Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

                // Bounds of receptive field, clamped to input size
                Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
                Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

                for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                    for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                        int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                        int inCI = historySamples[t - n].inputCIs[vli][visibleColumnIndex];

                        Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                        adv += vl.weightsAdv[inCI + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex))];
                    }
            }

            adv /= count;

            hiddenAdvsTemp[hiddenCellIndex] = adv;

            minAdv = min(minAdv, adv);
            maxAdv = max(maxAdv, adv);
        }

        int targetCI = historySamples[t - n - 1].hiddenTargetCIsPrev[hiddenColumnIndex];

        float q = value + (n == 0 ? hiddenAdvsTemp[targetCI + hiddenCellsStart] : maxAdv) - minAdv;

        tdError = historySamples[t - n - 1].reward + discount * qNext - q;

        float weight = discount * decay;

        float weighting = powf(discount * decay, n);

        gae += tdError * weighting;
        div += weighting;

        qNext = q;
    }

    gae /= div;

    float deltaV = vlr * tdError;

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        int diam = vld.radius * 2 + 1;

        // Projection
        Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
            static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));

        Int2 visibleCenter = project(columnPos, hToV);

        // Lower corner
        Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

        // Bounds of receptive field, clamped to input size
        Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
        Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int inCIPrev = historySamples[t].inputCIs[vli][visibleColumnIndex];

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                vl.weightsV[inCIPrev + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenColumnIndex))] += deltaV;
            }
    }

    int targetCIPrev = historySamples[t - 1].hiddenTargetCIsPrev[hiddenColumnIndex];

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        float deltaAdv = (hc == targetCIPrev ? alr * gae : -force * hiddenAdvsTemp[hiddenCellIndex]) - reg * minAdv;

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int diam = vld.radius * 2 + 1;

            // Projection
            Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
                static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));

            Int2 visibleCenter = project(columnPos, hToV);

            // Lower corner
            Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

            // Bounds of receptive field, clamped to input size
            Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
            Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                    int inCIPrev = historySamples[t].inputCIs[vli][visibleColumnIndex];

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    vl.weightsAdv[inCIPrev + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex))] += deltaAdv;
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
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    // Create layers
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisibleCells = numVisibleColumns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        // Create weight matrix for this visible layer
        vl.weightsV.resize(numHiddenColumns * area * vld.size.z);

        for (int i = 0; i < vl.weightsV.size(); i++)
            vl.weightsV[i] = randf(-0.0001f, 0.0001f);

        vl.weightsAdv.resize(numHiddenCells * area * vld.size.z);

        for (int i = 0; i < vl.weightsAdv.size(); i++)
            vl.weightsAdv[i] = randf(-0.0001f, 0.0001f);
    }

    hiddenCIs = IntBuffer(numHiddenColumns, 0);

    hiddenAdvsTemp = FloatBuffer(numHiddenCells, 0.0f);

    // Create (pre-allocated) history samples
    historySize = 0;
    historySamples.resize(historyCapacity);

    for (int i = 0; i < historySamples.size(); i++) {
        historySamples[i].inputCIs.resize(visibleLayers.size());

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

            int numVisibleColumns = vld.size.x * vld.size.y;

            historySamples[i].inputCIs[vli] = IntBuffer(numVisibleColumns);
        }

        historySamples[i].hiddenTargetCIsPrev = IntBuffer(numHiddenColumns);
    }
}

void Actor::step(
    const Array<const IntBuffer*> &inputCIs,
    const IntBuffer* hiddenTargetCIsPrev,
    float reward,
    bool learnEnabled
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCIs);

    historySamples.pushFront();

    // If not at cap, increment
    if (historySize < historySamples.size())
        historySize++;
    
    // Add new sample
    {
        HistorySample &s = historySamples[0];

        for (int vli = 0; vli < visibleLayers.size(); vli++)
            s.inputCIs[vli] = *inputCIs[vli];

        // Copy hidden CIs
        s.hiddenTargetCIsPrev = *hiddenTargetCIsPrev;

        s.reward = reward;
    }

    // Learn (if have sufficient samples)
    if (learnEnabled && historySize > nSteps) {
        for (int it = 0; it < historyIters; it++) {
            int t = rand() % (historySize - nSteps) + nSteps;

            #pragma omp parallel for
            for (int i = 0; i < numHiddenColumns; i++)
                learn(Int2(i / hiddenSize.y, i % hiddenSize.y), t);
        }
    }
}

int Actor::size() const {
    int size = sizeof(Int3) + 5 * sizeof(float) + 2 * sizeof(int) + hiddenCIs.size() * sizeof(int) + sizeof(int);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        size += sizeof(VisibleLayerDesc) + vl.weightsV.size() * sizeof(float) + vl.weightsAdv.size() * sizeof(float);
    }

    size += 3 * sizeof(int);

    int sampleSize = 0;

    const HistorySample &s = historySamples[0];

    for (int vli = 0; vli < visibleLayers.size(); vli++)
        sampleSize += s.inputCIs[vli].size() * sizeof(int);

    sampleSize += s.hiddenTargetCIsPrev.size() * sizeof(int) + sizeof(float);

    size += historySamples.size() * sampleSize;

    return size;
}

int Actor::stateSize() const {
    int size = hiddenCIs.size() * sizeof(int) + sizeof(int);

    int sampleSize = 0;

    const HistorySample &s = historySamples[0];

    for (int vli = 0; vli < visibleLayers.size(); vli++)
        sampleSize += s.inputCIs[vli].size() * sizeof(int);

    sampleSize += s.hiddenTargetCIsPrev.size() * sizeof(int) + sizeof(float);

    size += historySamples.size() * sampleSize;

    return size;
}

void Actor::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&vlr), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&alr), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&force), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&reg), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&discount), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&decay), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&nSteps), sizeof(int));
    writer.write(reinterpret_cast<const void*>(&historyIters), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    int numVisibleCellsLayers = visibleLayers.size();

    writer.write(reinterpret_cast<const void*>(&numVisibleCellsLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

        writer.write(reinterpret_cast<const void*>(&vl.weightsV[0]), vl.weightsV.size() * sizeof(float));
        writer.write(reinterpret_cast<const void*>(&vl.weightsAdv[0]), vl.weightsAdv.size() * sizeof(float));
    }

    writer.write(reinterpret_cast<const void*>(&historySize), sizeof(int));

    int numHistorySamples = historySamples.size();

    writer.write(reinterpret_cast<const void*>(&numHistorySamples), sizeof(int));

    int historyStart = historySamples.start;

    writer.write(reinterpret_cast<const void*>(&historyStart), sizeof(int));

    for (int t = 0; t < historySamples.size(); t++) {
        const HistorySample &s = historySamples[t];

        for (int vli = 0; vli < visibleLayers.size(); vli++)
            writer.write(reinterpret_cast<const void*>(&s.inputCIs[vli][0]), s.inputCIs[vli].size() * sizeof(int));

        writer.write(reinterpret_cast<const void*>(&s.hiddenTargetCIsPrev[0]), s.hiddenTargetCIsPrev.size() * sizeof(int));

        writer.write(reinterpret_cast<const void*>(&s.reward), sizeof(float));
    }
}

void Actor::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;
    
    reader.read(reinterpret_cast<void*>(&vlr), sizeof(float));
    reader.read(reinterpret_cast<void*>(&alr), sizeof(float));
    reader.read(reinterpret_cast<void*>(&force), sizeof(float));
    reader.read(reinterpret_cast<void*>(&reg), sizeof(float));
    reader.read(reinterpret_cast<void*>(&discount), sizeof(float));
    reader.read(reinterpret_cast<void*>(&decay), sizeof(float));
    reader.read(reinterpret_cast<void*>(&nSteps), sizeof(int));
    reader.read(reinterpret_cast<void*>(&historyIters), sizeof(int));

    hiddenCIs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    hiddenAdvsTemp = FloatBuffer(numHiddenCells, 0.0f);

    int numVisibleCellsLayers = visibleLayers.size();

    reader.read(reinterpret_cast<void*>(&numVisibleCellsLayers), sizeof(int));

    visibleLayers.resize(numVisibleCellsLayers);
    visibleLayerDescs.resize(numVisibleCellsLayers);
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = visibleLayerDescs[vli];

        reader.read(reinterpret_cast<void*>(&vld), sizeof(VisibleLayerDesc));

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisibleCells = numVisibleColumns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weightsV.resize(numHiddenColumns * area * vld.size.z);
        vl.weightsAdv.resize(numHiddenCells * area * vld.size.z);

        reader.read(reinterpret_cast<void*>(&vl.weightsV[0]), vl.weightsV.size() * sizeof(float));
        reader.read(reinterpret_cast<void*>(&vl.weightsAdv[0]), vl.weightsAdv.size() * sizeof(float));
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

        s.inputCIs.resize(numVisibleCellsLayers);

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int numVisibleColumns = vld.size.x * vld.size.y;

            s.inputCIs[vli].resize(numVisibleColumns);

            reader.read(reinterpret_cast<void*>(&s.inputCIs[vli][0]), s.inputCIs[vli].size() * sizeof(int));
        }

        s.hiddenTargetCIsPrev.resize(numHiddenColumns);

        reader.read(reinterpret_cast<void*>(&s.hiddenTargetCIsPrev[0]), s.hiddenTargetCIsPrev.size() * sizeof(int));

        reader.read(reinterpret_cast<void*>(&s.reward), sizeof(float));
    }
}

void Actor::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    int historyStart = historySamples.start;

    writer.write(reinterpret_cast<const void*>(&historyStart), sizeof(int));

    for (int t = 0; t < historySamples.size(); t++) {
        const HistorySample &s = historySamples[t];

        for (int vli = 0; vli < visibleLayers.size(); vli++)
            writer.write(reinterpret_cast<const void*>(&s.inputCIs[vli][0]), s.inputCIs[vli].size() * sizeof(int));

        writer.write(reinterpret_cast<const void*>(&s.hiddenTargetCIsPrev[0]), s.hiddenTargetCIsPrev.size() * sizeof(int));

        writer.write(reinterpret_cast<const void*>(&s.reward), sizeof(float));
    }
}

void Actor::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    int historyStart;

    reader.read(reinterpret_cast<void*>(&historyStart), sizeof(int));

    historySamples.start = historyStart;

    for (int t = 0; t < historySamples.size(); t++) {
        HistorySample &s = historySamples[t];

        for (int vli = 0; vli < visibleLayers.size(); vli++)
            reader.read(reinterpret_cast<void*>(&s.inputCIs[vli][0]), s.inputCIs[vli].size() * sizeof(int));

        reader.read(reinterpret_cast<void*>(&s.hiddenTargetCIsPrev[0]), s.hiddenTargetCIsPrev.size() * sizeof(int));

        reader.read(reinterpret_cast<void*>(&s.reward), sizeof(float));
    }
}

