// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Actor.h"

using namespace aon;

void Actor::activate(
    const Int2 &columnPos,
    const Array<const IntBuffer*> &inputCIs,
    const Array<const FloatBuffer*> &inputActs,
    const IntBuffer* hiddenTargetCIsPrev,
    float reward,
    bool learnEnabled
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    int targetCI = (*hiddenTargetCIsPrev)[hiddenColumnIndex];

    float valuePrev = hiddenQs[targetCI + hiddenCellsStart];

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        float sum = 0.0f;
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

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                    if (inputActs[vli] == nullptr) {
                        int inCI = (*inputCIs[vli])[visibleColumnIndex];

                        int wi = inCI + wiStart;

                        sum += vl.weightsEval[wi];
                    }
                    else {
                        int visibleCellsStart = visibleColumnIndex * vld.size.z;

                        for (int vc = 0; vc < vld.size.z; vc++) {
                            int visibleCellIndex = vc + visibleCellsStart;

                            float inAct = (*inputActs[vli])[visibleCellIndex];

                            int wi = vc + wiStart;

                            sum += vl.weightsEval[wi] * inAct;
                        }
                    }

                    if (learnEnabled) {
                        for (int vc = 0; vc < vld.size.z; vc++) {
                            int wi = vc + wiStart;

                            vl.weightsEval[wi] += drift * (vl.weightsLearn[wi] - vl.weightsEval[wi]);
                        }
                    }
                }
        }

        sum /= count;

        hiddenQs[hiddenCellIndex] = sum;

        if (sum > maxActivation || maxIndex == -1) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    hiddenTDErrors[hiddenColumnIndex] = reward + discount * maxActivation - valuePrev;

    hiddenCIs[hiddenColumnIndex] = maxIndex;
}

void Actor::learn(
    const Int2 &columnPos,
    int t,
    float r,
    float d
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        float sum = 0.0f;
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

                    int visibleCellsStart = visibleColumnIndex * vld.size.z;

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                    if (historySamples[t - nSteps].inputActs[vli][visibleCellsStart] == -1.0f) {
                        int inCI = historySamples[t - nSteps].inputCIs[vli][visibleColumnIndex];

                        int wi = inCI + wiStart;

                        sum += vl.weightsEval[wi];
                    }
                    else {
                        for (int vc = 0; vc < vld.size.z; vc++) {
                            int visibleCellIndex = vc + visibleCellsStart;

                            float inAct = historySamples[t - nSteps].inputActs[vli][visibleCellIndex];

                            int wi = vc + wiStart;

                            sum += vl.weightsEval[wi] * inAct;
                        }
                    }
                }
        }

        sum /= count;

        maxActivation = max(maxActivation, sum);
    }

    int targetCI = historySamples[t - 1].hiddenTargetCIsPrev[hiddenColumnIndex];

    int hiddenCellIndexTarget = targetCI + hiddenCellsStart;

    float sumPrev = 0.0f;
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

                int visibleCellsStart = visibleColumnIndex * vld.size.z;

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndexTarget));

                if (historySamples[t].inputActs[vli][visibleCellsStart] == -1.0f) {
                    int inCI = historySamples[t].inputCIs[vli][visibleColumnIndex];

                    int wi = inCI + wiStart;

                    sumPrev += vl.weightsLearn[wi];
                }
                else {
                    for (int vc = 0; vc < vld.size.z; vc++) {
                        int visibleCellIndex = vc + visibleCellsStart;

                        float inAct = historySamples[t].inputActs[vli][visibleCellIndex];

                        int wi = vc + wiStart;

                        sumPrev += vl.weightsLearn[wi] * inAct;
                    }
                }
            }
    }

    sumPrev /= count;

    float newValue = r + d * maxActivation;

    float delta = lr * (newValue - sumPrev);

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

                int visibleCellsStart = visibleColumnIndex * vld.size.z;

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndexTarget));

                if (historySamples[t].inputActs[vli][visibleCellsStart] == -1.0f) {
                    int inCI = historySamples[t].inputCIs[vli][visibleColumnIndex];

                    int wi = inCI + wiStart;

                    vl.weightsLearn[wi] += delta;
                }
                else {
                    for (int vc = 0; vc < vld.size.z; vc++) {
                        int visibleCellIndex = vc + visibleCellsStart;

                        float inAct = historySamples[t].inputActs[vli][visibleCellIndex];

                        int wi = vc + wiStart;

                        vl.weightsLearn[wi] += delta * inAct;
                    }
                }
            }
    }
}

void Actor::generateErrors(
    const Int2 &columnPos,
    FloatBuffer* visibleErrors,
    int t,
    int vli
) {
    VisibleLayer &vl = visibleLayers[vli];
    VisibleLayerDesc &vld = visibleLayerDescs[vli];

    int diam = vld.radius * 2 + 1;

    int visibleColumnIndex = address2(columnPos, Int2(vld.size.x, vld.size.y));

    int visibleCellsStart = visibleColumnIndex * vld.size.z;

    // Projection
    Float2 vToH = Float2(static_cast<float>(hiddenSize.x) / static_cast<float>(vld.size.x),
        static_cast<float>(hiddenSize.y) / static_cast<float>(vld.size.y));

    Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
        static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));
                
    Int2 hiddenCenter = project(columnPos, vToH);

    Int2 reverseRadii(ceilf(vToH.x * (vld.radius * 2 + 1) * 0.5f), ceilf(vToH.y * (vld.radius * 2 + 1) * 0.5f));

    // Lower corner
    Int2 fieldLowerBound(hiddenCenter.x - reverseRadii.x, hiddenCenter.y - reverseRadii.y);

    // Bounds of receptive field, clamped to input size
    Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
    Int2 iterUpperBound(min(hiddenSize.x - 1, hiddenCenter.x + reverseRadii.x), min(hiddenSize.y - 1, hiddenCenter.y + reverseRadii.y));

    if (historySamples[t].inputActs[vli][visibleCellsStart] == -1.0f) {
        int inCIPrev = historySamples[t].inputCIs[vli][visibleColumnIndex];

        int visibleCellIndex = inCIPrev + visibleCellsStart;

        float sum = 0.0f;
        int count = 0;

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                Int2 hiddenPos = Int2(ix, iy);

                int hiddenColumnIndex = address2(hiddenPos, Int2(hiddenSize.x, hiddenSize.y));

                Int2 visibleCenter = project(hiddenPos, hToV);

                if (inBounds(columnPos, Int2(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius), Int2(visibleCenter.x + vld.radius + 1, visibleCenter.y + vld.radius + 1))) {
                    Int2 offset(columnPos.x - visibleCenter.x + vld.radius, columnPos.y - visibleCenter.y + vld.radius);

                    int hiddenCellIndexTarget = historySamples[t - 1].hiddenTargetCIsPrev[hiddenColumnIndex] + hiddenColumnIndex * hiddenSize.z;

                    sum += hiddenTDErrors[hiddenColumnIndex] * vl.weightsLearn[inCIPrev + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndexTarget))];
                    count++;
                }
            }

        sum /= max(1, count);

        (*visibleErrors)[visibleCellIndex] += sum;
    }
    else {
        int inCIPrev = historySamples[t].inputCIs[vli][visibleColumnIndex];

        for (int vc = 0; vc < vld.size.z; vc++) {
            int visibleCellIndex = vc + visibleCellsStart;

            if (historySamples[t].inputActs[vli][visibleCellIndex] <= 0.0f)
                continue;

            float sum = 0.0f;
            int count = 0;

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    Int2 hiddenPos = Int2(ix, iy);

                    int hiddenColumnIndex = address2(hiddenPos, Int2(hiddenSize.x, hiddenSize.y));

                    Int2 visibleCenter = project(hiddenPos, hToV);

                    if (inBounds(columnPos, Int2(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius), Int2(visibleCenter.x + vld.radius + 1, visibleCenter.y + vld.radius + 1))) {
                        Int2 offset(columnPos.x - visibleCenter.x + vld.radius, columnPos.y - visibleCenter.y + vld.radius);

                        int hiddenCellIndexTarget = historySamples[t - 1].hiddenTargetCIsPrev[hiddenColumnIndex] + hiddenColumnIndex * hiddenSize.z;

                        sum += hiddenTDErrors[hiddenColumnIndex] * vl.weightsLearn[vc + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndexTarget))];
                        count++;
                    }
                }

            sum /= max(1, count);

            (*visibleErrors)[visibleCellIndex] += sum;
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
        vl.weightsEval.resize(numHiddenCells * area * vld.size.z);

        for (int i = 0; i < vl.weightsEval.size(); i++)
            vl.weightsEval[i] = randf(-0.01f, 0.01f);

        vl.weightsLearn = vl.weightsEval;
    }

    hiddenCIs = IntBuffer(numHiddenColumns, 0);

    hiddenQs = FloatBuffer(numHiddenCells, 0.0f);
    hiddenTDErrors = FloatBuffer(numHiddenColumns, 0.0f);

    // Create (pre-allocated) history samples
    historySize = 0;
    historySamples.resize(historyCapacity);

    for (int i = 0; i < historySamples.size(); i++) {
        historySamples[i].inputCIs.resize(visibleLayers.size());
        historySamples[i].inputActs.resize(visibleLayers.size());

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

            int numVisibleColumns = vld.size.x * vld.size.y;
            int numVisibleCells = numVisibleColumns * vld.size.z;

            historySamples[i].inputCIs[vli] = IntBuffer(numVisibleColumns);
            historySamples[i].inputActs[vli] = FloatBuffer(numVisibleCells, -1.0f);
        }

        historySamples[i].hiddenTargetCIsPrev = IntBuffer(numHiddenColumns);
    }
}

void Actor::step(
    const Array<const IntBuffer*> &inputCIs,
    const Array<const FloatBuffer*> &inputActs,
    const IntBuffer* hiddenTargetCIsPrev,
    float reward,
    bool learnEnabled
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        activate(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCIs, inputActs, hiddenTargetCIsPrev, reward, learnEnabled);

    historySamples.pushFront();

    // If not at cap, increment
    if (historySize < historySamples.size())
        historySize++;
    
    // Add new sample
    {
        HistorySample &s = historySamples[0];

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            s.inputCIs[vli] = *inputCIs[vli];

            if (inputActs[vli] == nullptr)
                s.inputActs[vli].fill(-1.0f); // Flag
            else
                s.inputActs[vli] = *inputActs[vli];
        }

        // Copy hidden CIs
        s.hiddenTargetCIsPrev = *hiddenTargetCIsPrev;

        s.reward = reward;
    }

    // Learn (if have sufficient samples)
    if (learnEnabled && historySize > nSteps) {
        for (int it = 0; it < historyIters; it++) {
            int t = rand() % (historySize - nSteps) + nSteps;

            // Compute (partial) values, rest is completed in the kernel
            float r = 0.0f;
            float d = 1.0f;

            for (int t2 = t - 1; t2 >= t - nSteps; t2--) {
                r += historySamples[t2].reward * d;

                d *= discount;
            }

            #pragma omp parallel for
            for (int i = 0; i < numHiddenColumns; i++)
                learn(Int2(i / hiddenSize.y, i % hiddenSize.y), t, r, d);
        }
    }
}

void Actor::generateErrors(
    FloatBuffer* visibleErrors,
    int vli
) {
    if (historySize < 2)
        return;

    const VisibleLayer &vl = visibleLayers[vli];
    const VisibleLayerDesc &vld = visibleLayerDescs[vli];

    int numVisibleColumns = vld.size.x * vld.size.y;

    #pragma omp parallel for
    for (int i = 0; i < numVisibleColumns; i++)
        generateErrors(Int2(i / vld.size.y, i % vld.size.y), visibleErrors, 1, vli);
}

int Actor::size() const {
    int size = sizeof(Int3) + 3 * sizeof(float) + 2 * sizeof(int) + hiddenCIs.size() * sizeof(int) + hiddenQs.size() * sizeof(float) + hiddenTDErrors.size() * sizeof(float) + sizeof(int);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        size += sizeof(VisibleLayerDesc) + 2 * vl.weightsEval.size() * sizeof(float);
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
    int size = hiddenCIs.size() * sizeof(int) + hiddenQs.size() * sizeof(float) + hiddenTDErrors.size() * sizeof(float) + sizeof(int);

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

    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&drift), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&discount), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&nSteps), sizeof(int));
    writer.write(reinterpret_cast<const void*>(&historyIters), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hiddenQs[0]), hiddenQs.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&hiddenTDErrors[0]), hiddenTDErrors.size() * sizeof(float));

    int numVisibleCellsLayers = visibleLayers.size();

    writer.write(reinterpret_cast<const void*>(&numVisibleCellsLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

        writer.write(reinterpret_cast<const void*>(&vl.weightsEval[0]), vl.weightsEval.size() * sizeof(float));
        writer.write(reinterpret_cast<const void*>(&vl.weightsLearn[0]), vl.weightsLearn.size() * sizeof(float));
    }

    writer.write(reinterpret_cast<const void*>(&historySize), sizeof(int));

    int numHistorySamples = historySamples.size();

    writer.write(reinterpret_cast<const void*>(&numHistorySamples), sizeof(int));

    int historyStart = historySamples.start;

    writer.write(reinterpret_cast<const void*>(&historyStart), sizeof(int));

    for (int t = 0; t < historySamples.size(); t++) {
        const HistorySample &s = historySamples[t];

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            writer.write(reinterpret_cast<const void*>(&s.inputCIs[vli][0]), s.inputCIs[vli].size() * sizeof(int));
            writer.write(reinterpret_cast<const void*>(&s.inputActs[vli][0]), s.inputActs[vli].size() * sizeof(float));
        }

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
    
    reader.read(reinterpret_cast<void*>(&lr), sizeof(float));
    reader.read(reinterpret_cast<void*>(&drift), sizeof(float));
    reader.read(reinterpret_cast<void*>(&discount), sizeof(float));
    reader.read(reinterpret_cast<void*>(&nSteps), sizeof(int));
    reader.read(reinterpret_cast<void*>(&historyIters), sizeof(int));

    hiddenCIs.resize(numHiddenColumns);
    hiddenQs.resize(numHiddenCells);
    hiddenTDErrors.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hiddenQs[0]), hiddenQs.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&hiddenTDErrors[0]), hiddenTDErrors.size() * sizeof(float));

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

        vl.weightsEval.resize(numHiddenCells * area * vld.size.z);
        vl.weightsLearn.resize(vl.weightsEval.size());

        reader.read(reinterpret_cast<void*>(&vl.weightsEval[0]), vl.weightsEval.size() * sizeof(float));
        reader.read(reinterpret_cast<void*>(&vl.weightsLearn[0]), vl.weightsLearn.size() * sizeof(float));
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
        s.inputActs.resize(numVisibleCellsLayers);

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int numVisibleColumns = vld.size.x * vld.size.y;
            int numVisibleCells = numVisibleColumns * vld.size.z;

            s.inputCIs[vli].resize(numVisibleColumns);
            s.inputActs[vli].resize(numVisibleCells);

            reader.read(reinterpret_cast<void*>(&s.inputCIs[vli][0]), s.inputCIs[vli].size() * sizeof(int));
            reader.read(reinterpret_cast<void*>(&s.inputActs[vli][0]), s.inputActs[vli].size() * sizeof(float));
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
    writer.write(reinterpret_cast<const void*>(&hiddenQs[0]), hiddenQs.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&hiddenTDErrors[0]), hiddenTDErrors.size() * sizeof(float));

    int historyStart = historySamples.start;

    writer.write(reinterpret_cast<const void*>(&historyStart), sizeof(int));

    for (int t = 0; t < historySamples.size(); t++) {
        const HistorySample &s = historySamples[t];

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            writer.write(reinterpret_cast<const void*>(&s.inputCIs[vli][0]), s.inputCIs[vli].size() * sizeof(int));
            writer.write(reinterpret_cast<const void*>(&s.inputActs[vli][0]), s.inputActs[vli].size() * sizeof(float));
        }

        writer.write(reinterpret_cast<const void*>(&s.hiddenTargetCIsPrev[0]), s.hiddenTargetCIsPrev.size() * sizeof(int));

        writer.write(reinterpret_cast<const void*>(&s.reward), sizeof(float));
    }
}

void Actor::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hiddenQs[0]), hiddenQs.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&hiddenTDErrors[0]), hiddenTDErrors.size() * sizeof(float));

    int historyStart;

    reader.read(reinterpret_cast<void*>(&historyStart), sizeof(int));

    historySamples.start = historyStart;

    for (int t = 0; t < historySamples.size(); t++) {
        HistorySample &s = historySamples[t];

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            reader.read(reinterpret_cast<void*>(&s.inputCIs[vli][0]), s.inputCIs[vli].size() * sizeof(int));
            reader.read(reinterpret_cast<void*>(&s.inputActs[vli][0]), s.inputActs[vli].size() * sizeof(float));
        }

        reader.read(reinterpret_cast<void*>(&s.hiddenTargetCIsPrev[0]), s.hiddenTargetCIsPrev.size() * sizeof(int));

        reader.read(reinterpret_cast<void*>(&s.reward), sizeof(float));
    }
}
