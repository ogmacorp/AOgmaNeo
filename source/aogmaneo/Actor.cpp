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
    const IntBuffer* nextCIs,
    const IntBuffer* inputCIs,
    unsigned int* state
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

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

    int count = (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1);

    // --- Action ---

    if (temperature > 0.0f) {
        float maxActivation = -999999.0f;

        for (int hc = 0; hc < hiddenSize.z; hc++) {
            int hiddenCellIndex = hc + hiddenCellsStart;

            float sum = 0.0f;

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                    if (vld.hasFeedBack) {
                        assert(nextCIs != nullptr);

                        int nextCI = (*nextCIs)[visibleColumnIndex];

                        sum += vl.actionWeightsNext[nextCI + wiStart];
                    }

                    int inCI = (*inputCIs)[visibleColumnIndex];

                    sum += vl.actionWeights[inCI + wiStart];
                }

            sum /= count;

            hiddenActs[hiddenCellIndex] = sum;

            maxActivation = max(maxActivation, sum);
        }

        float total = 0.0f;

        for (int hc = 0; hc < hiddenSize.z; hc++) {
            int hiddenCellIndex = hc + hiddenCellsStart;

            hiddenActs[hiddenCellIndex] = expf((hiddenActs[hiddenCellIndex] - maxActivation) / temperature);
            
            total += hiddenActs[hiddenCellIndex];
        }

        float cusp = randf(state) * total;

        int selectIndex = 0;
        float sumSoFar = 0.0f;

        for (int hc = 0; hc < hiddenSize.z; hc++) {
            int hiddenCellIndex = hc + hiddenCellsStart;

            sumSoFar += hiddenActs[hiddenCellIndex];

            if (sumSoFar >= cusp) {
                selectIndex = hc;

                break;
            }
        }
        
        hiddenCIs[hiddenColumnIndex] = selectIndex;
    }
    else { // Deterministic
        int maxIndex = -1;
        float maxActivation = -999999.0f;

        for (int hc = 0; hc < hiddenSize.z; hc++) {
            int hiddenCellIndex = hc + hiddenCellsStart;

            float sum = 0.0f;

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                    if (vld.hasFeedBack) {
                        assert(nextCIs != nullptr);

                        int nextCI = (*nextCIs)[visibleColumnIndex];

                        sum += vl.actionWeightsNext[nextCI + wiStart];
                    }

                    int inCI = (*inputCIs)[visibleColumnIndex];

                    sum += vl.actionWeights[inCI + wiStart];
                }

            if (sum > maxActivation || maxIndex == -1) {
                maxActivation = sum;
                maxIndex = hc;
            }
        }

        hiddenCIs[hiddenColumnIndex] = maxIndex;
    }
}

void Actor::learn(
    const Int2 &columnPos,
    int t,
    float r,
    float d,
    bool mimic
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    int targetCI = historySamples[t - 1].hiddenTargetCIsPrev[hiddenColumnIndex];

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

    int count = (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1);

    // --- Value ---
    
    float valueNext = 0.0f;

    for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
        for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
            int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

            Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

            int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenColumnIndex));

            if (vld.hasFeedBack) {
                int nextCI = historySamples[0].nextCIs[visibleColumnIndex];

                valueNext += vl.valueWeightsNext[nextCI + wiStart];
            }

            int inCI = historySamples[0].inputCIs[visibleColumnIndex];

            valueNext += vl.valueWeights[inCI + wiStart];
        }

    valueNext /= count;

    // --- Value Prev ---

    float newValue = r + d * valueNext;

    float valuePrev = 0.0f;

    for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
        for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
            int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

            Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

            int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenColumnIndex));

            if (vld.hasFeedBack) {
                int nextCI = historySamples[t].nextCIs[visibleColumnIndex];

                valuePrev += vl.valueWeightsNext[nextCI + wiStart];
            }

            int inCI = historySamples[t].inputCIs[visibleColumnIndex];

            valuePrev += vl.valueWeights[inCI + wiStart];
        }

    valuePrev /= count;

    float tdErrorValue = newValue - valuePrev;

    float deltaValue = vlr * tdErrorValue;

    for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
        for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
            int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

            Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

            int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenColumnIndex));

            if (vld.hasFeedBack) {
                int nextCI = historySamples[t].nextCIs[visibleColumnIndex];

                vl.valueWeightsNext[nextCI + wiStart] += deltaValue;
            }

            int inCI = historySamples[t].inputCIs[visibleColumnIndex];

            vl.valueWeights[inCI + wiStart] += deltaValue;
        }

    // --- Action ---

    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        float sum = 0.0f;

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                if (vld.hasFeedBack) {
                    int nextCI = historySamples[t].nextCIs[visibleColumnIndex];

                    sum += vl.actionWeightsNext[nextCI + wiStart];
                }

                int inCI = historySamples[t].inputCIs[visibleColumnIndex];

                sum += vl.actionWeights[inCI + wiStart];
            }

        sum /= count;

        hiddenActs[hiddenCellIndex] = sum;

        maxActivation = max(maxActivation, sum);
    }

    float total = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        hiddenActs[hiddenCellIndex] = expf(hiddenActs[hiddenCellIndex] - maxActivation);

        total += hiddenActs[hiddenCellIndex];
    }

    float scale = 1.0f / max(0.0001f, total);

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        hiddenActs[hiddenCellIndex] *= scale;

        float deltaAction = (mimic ? alr : alr * tanhf(tdErrorValue)) * ((hc == targetCI) - hiddenActs[hiddenCellIndex]);

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                if (vld.hasFeedBack) {
                    int nextCI = historySamples[t].nextCIs[visibleColumnIndex];

                    vl.actionWeightsNext[nextCI + wiStart] += deltaAction;
                }

                int inCI = historySamples[t].inputCIs[visibleColumnIndex];

                vl.actionWeights[inCI + wiStart] += deltaAction;
            }
    }
}

void Actor::initRandom(
    const Int3 &hiddenSize,
    int historyCapacity,
    const VisibleLayerDesc &vld
) {
    this->vld = vld;

    this->hiddenSize = hiddenSize;

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    int numVisibleColumns = vld.size.x * vld.size.y;
    int numVisibleCells = numVisibleColumns * vld.size.z;

    int diam = vld.radius * 2 + 1;
    int area = diam * diam;

    // Create weight matrix for this visible layer and initialize randomly
    vl.valueWeights.resize(numHiddenColumns * area * vld.size.z);

    for (int i = 0; i < vl.valueWeights.size(); i++)
        vl.valueWeights[i] = randf(-0.01f, 0.01f);

    if (vld.hasFeedBack) {
        vl.valueWeightsNext.resize(vl.valueWeights.size());

        for (int i = 0; i < vl.valueWeightsNext.size(); i++)
            vl.valueWeightsNext[i] = randf(-0.01f, 0.01f);
    }

    vl.actionWeights.resize(numHiddenCells * area * vld.size.z);

    for (int i = 0; i < vl.actionWeights.size(); i++)
        vl.actionWeights[i] = randf(-0.01f, 0.01f);

    if (vld.hasFeedBack) {
        vl.actionWeightsNext.resize(vl.actionWeights.size());

        for (int i = 0; i < vl.actionWeightsNext.size(); i++)
            vl.actionWeightsNext[i] = randf(-0.01f, 0.01f);
    }

    hiddenCIs = IntBuffer(numHiddenColumns, 0);

    hiddenActs = FloatBuffer(numHiddenCells, 0.0f);

    // Create (pre-allocated) history samples
    historySize = 0;
    historySamples.resize(historyCapacity);

    for (int i = 0; i < historySamples.size(); i++) {
        if (vld.hasFeedBack)
            historySamples[i].nextCIs = IntBuffer(numVisibleColumns);

        historySamples[i].inputCIs = IntBuffer(numVisibleColumns);

        historySamples[i].hiddenTargetCIsPrev = IntBuffer(numHiddenColumns);
    }
}

void Actor::step(
    const IntBuffer* nextCIs,
    const IntBuffer* inputCIs,
    const IntBuffer* hiddenTargetCIsPrev,
    float reward,
    bool learnEnabled,
    bool mimic
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    // Forward kernel
    unsigned int baseState = rand();

    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++) {
        unsigned int state = baseState + i * 12345;

        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), nextCIs, inputCIs, &state);
    }

    historySamples.pushFront();

    // If not at cap, increment
    if (historySize < historySamples.size())
        historySize++;
    
    // Add new sample
    {
        HistorySample &s = historySamples[0];

        if (vld.hasFeedBack)
            s.nextCIs = *nextCIs;

        s.inputCIs = *inputCIs;

        s.hiddenTargetCIsPrev = *hiddenTargetCIsPrev;

        s.reward = reward;
    }

    // Learn (if have sufficient samples)
    if (learnEnabled && historySize > minSteps) {
        for (int it = 0; it < historyIters; it++) {
            int t = rand() % (historySize - minSteps) + minSteps;

            // Compute (partial) values, rest is completed in the kernel
            float r = 0.0f;
            float d = 1.0f;

            for (int t2 = t - 1; t2 >= 0; t2--) {
                r += historySamples[t2].reward * d;

                d *= discount;
            }

            #pragma omp parallel for
            for (int i = 0; i < numHiddenColumns; i++)
                learn(Int2(i / hiddenSize.y, i % hiddenSize.y), t, r, d, mimic);
        }
    }
}

void Actor::clearState() {
    hiddenCIs.fill(0);

    historySize = 0;
}

int Actor::size() const {
    int size = sizeof(Int3) + 4 * sizeof(float) + 2 * sizeof(int) + hiddenCIs.size() * sizeof(int);

    size += sizeof(VisibleLayerDesc) + (vld.hasFeedBack ? 2 : 1) * (vl.valueWeights.size() * sizeof(float) + vl.actionWeights.size() * sizeof(float));

    size += 3 * sizeof(int);

    const HistorySample &s = historySamples[0];

    int sampleSize = (vld.hasFeedBack ? 2 : 1) * s.inputCIs.size() * sizeof(int) + s.hiddenTargetCIsPrev.size() * sizeof(int) + sizeof(float);

    size += historySamples.size() * sampleSize;

    return size;
}

int Actor::stateSize() const {
    int size = hiddenCIs.size() * sizeof(int) + 2 * sizeof(int);

    const HistorySample &s = historySamples[0];

    int sampleSize = (vld.hasFeedBack ? 2 : 1) * s.inputCIs.size() * sizeof(int) + s.hiddenTargetCIsPrev.size() * sizeof(int) + sizeof(float);

    size += historySamples.size() * sampleSize;

    return size;
}

void Actor::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&vlr), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&alr), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&discount), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&temperature), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&minSteps), sizeof(int));
    writer.write(reinterpret_cast<const void*>(&historyIters), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

    writer.write(reinterpret_cast<const void*>(&vl.valueWeights[0]), vl.valueWeights.size() * sizeof(float));

    if (vld.hasFeedBack)
        writer.write(reinterpret_cast<const void*>(&vl.valueWeightsNext[0]), vl.valueWeightsNext.size() * sizeof(float));

    writer.write(reinterpret_cast<const void*>(&vl.actionWeights[0]), vl.actionWeights.size() * sizeof(float));

    if (vld.hasFeedBack)
        writer.write(reinterpret_cast<const void*>(&vl.actionWeightsNext[0]), vl.actionWeightsNext.size() * sizeof(float));

    writer.write(reinterpret_cast<const void*>(&historySize), sizeof(int));

    int numHistorySamples = historySamples.size();

    writer.write(reinterpret_cast<const void*>(&numHistorySamples), sizeof(int));

    int historyStart = historySamples.start;

    writer.write(reinterpret_cast<const void*>(&historyStart), sizeof(int));

    for (int t = 0; t < historySamples.size(); t++) {
        const HistorySample &s = historySamples[t];

        if (vld.hasFeedBack)
            writer.write(reinterpret_cast<const void*>(&s.nextCIs[0]), s.nextCIs.size() * sizeof(int));

        writer.write(reinterpret_cast<const void*>(&s.inputCIs[0]), s.inputCIs.size() * sizeof(int));

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
    reader.read(reinterpret_cast<void*>(&discount), sizeof(float));
    reader.read(reinterpret_cast<void*>(&temperature), sizeof(float));
    reader.read(reinterpret_cast<void*>(&minSteps), sizeof(int));
    reader.read(reinterpret_cast<void*>(&historyIters), sizeof(int));

    hiddenCIs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    hiddenActs = FloatBuffer(numHiddenCells, 0.0f);

    reader.read(reinterpret_cast<void*>(&vld), sizeof(VisibleLayerDesc));

    int numVisibleColumns = vld.size.x * vld.size.y;
    int numVisibleCells = numVisibleColumns * vld.size.z;

    int diam = vld.radius * 2 + 1;
    int area = diam * diam;

    vl.valueWeights.resize(numHiddenColumns * area * vld.size.z);
    vl.actionWeights.resize(numHiddenCells * area * vld.size.z);

    reader.read(reinterpret_cast<void*>(&vl.valueWeights[0]), vl.valueWeights.size() * sizeof(float));

    if (vld.hasFeedBack) {
        vl.valueWeightsNext.resize(vl.valueWeights.size());

        reader.read(reinterpret_cast<void*>(&vl.valueWeightsNext[0]), vl.valueWeightsNext.size() * sizeof(float));
    }

    reader.read(reinterpret_cast<void*>(&vl.actionWeights[0]), vl.actionWeights.size() * sizeof(float));

    if (vld.hasFeedBack) {
        vl.actionWeightsNext.resize(vl.actionWeights.size());

        reader.read(reinterpret_cast<void*>(&vl.actionWeightsNext[0]), vl.actionWeightsNext.size() * sizeof(float));
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

        if (vld.hasFeedBack) {
            s.nextCIs.resize(numVisibleColumns);

            reader.read(reinterpret_cast<void*>(&s.nextCIs[0]), s.nextCIs.size() * sizeof(int));
        }

        s.inputCIs.resize(numVisibleColumns);

        reader.read(reinterpret_cast<void*>(&s.inputCIs[0]), s.inputCIs.size() * sizeof(int));

        s.hiddenTargetCIsPrev.resize(numHiddenColumns);

        reader.read(reinterpret_cast<void*>(&s.hiddenTargetCIsPrev[0]), s.hiddenTargetCIsPrev.size() * sizeof(int));

        reader.read(reinterpret_cast<void*>(&s.reward), sizeof(float));
    }
}

void Actor::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    writer.write(reinterpret_cast<const void*>(&historySize), sizeof(int));

    int historyStart = historySamples.start;

    writer.write(reinterpret_cast<const void*>(&historyStart), sizeof(int));

    for (int t = 0; t < historySamples.size(); t++) {
        const HistorySample &s = historySamples[t];

        if (vld.hasFeedBack)
            writer.write(reinterpret_cast<const void*>(&s.nextCIs[0]), s.nextCIs.size() * sizeof(int));

        writer.write(reinterpret_cast<const void*>(&s.inputCIs[0]), s.inputCIs.size() * sizeof(int));

        writer.write(reinterpret_cast<const void*>(&s.hiddenTargetCIsPrev[0]), s.hiddenTargetCIsPrev.size() * sizeof(int));

        writer.write(reinterpret_cast<const void*>(&s.reward), sizeof(float));
    }
}

void Actor::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    reader.read(reinterpret_cast<void*>(&historySize), sizeof(int));

    int historyStart;

    reader.read(reinterpret_cast<void*>(&historyStart), sizeof(int));

    historySamples.start = historyStart;

    for (int t = 0; t < historySamples.size(); t++) {
        HistorySample &s = historySamples[t];

        if (vld.hasFeedBack)
            reader.read(reinterpret_cast<void*>(&s.nextCIs[0]), s.nextCIs.size() * sizeof(int));

        reader.read(reinterpret_cast<void*>(&s.inputCIs[0]), s.inputCIs.size() * sizeof(int));

        reader.read(reinterpret_cast<void*>(&s.hiddenTargetCIsPrev[0]), s.hiddenTargetCIsPrev.size() * sizeof(int));

        reader.read(reinterpret_cast<void*>(&s.reward), sizeof(float));
    }
}
