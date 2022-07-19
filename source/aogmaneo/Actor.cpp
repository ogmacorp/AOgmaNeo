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
    const IntBuffer* hiddenTargetCIs,
    float reward
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    int targetCI = (*hiddenTargetCIs)[hiddenColumnIndex];

    float valuePrev = hiddenValues[targetCI + hiddenCellsStart];

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

                    if (inputActs[vli] == nullptr) {
                        int inCI = (*inputCIs[vli])[visibleColumnIndex];

                        int wi = inCI + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                        sum += vl.weights[wi];
                    }
                    else {
                        int visibleCellsStart = visibleColumnIndex * vld.size.z;

                        int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                        for (int vc = 0; vc < vld.size.z; vc++) {
                            int visibleCellIndex = vc + visibleCellsStart;

                            float inAct = (*inputActs[vli])[visibleCellIndex];

                            int wi = vc + wiStart;

                            sum += vl.weights[wi] * inAct;
                        }
                    }
                }
        }

        sum /= count;

        hiddenValues[hiddenCellIndex] = sum;

        if (sum > maxActivation || maxIndex == -1) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    hiddenCIs[hiddenColumnIndex] = maxIndex;

    hiddenTDErrors[hiddenColumnIndex] = min(1.0f, max(-1.0f, reward + discount * maxActivation - valuePrev));
}

void Actor::learn(
    const Int2 &columnPos,
    const IntBuffer* hiddenTargetCIs
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    int targetCI = (*hiddenTargetCIs)[hiddenColumnIndex];

    float delta = lr * hiddenTDErrors[hiddenColumnIndex];

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

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

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    int visibleCellsStart = visibleColumnIndex * vld.size.z;

                    if (vl.inputActsPrev[visibleCellsStart] == -1.0f) {
                        int inCIPrev = vl.inputCIsPrev[visibleColumnIndex];

                        int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                        for (int vc = 0; vc < vld.size.z; vc++) {
                            int visibleCellIndex = vc + visibleCellsStart;

                            int wi = vc + wiStart;

                            vl.traces[wi] *= traceDecay;

                            if (vc == inCIPrev && hc == targetCI)
                                vl.traces[wi] = 1.0f;

                            vl.weights[wi] += delta * vl.traces[wi];
                        }
                    }
                    else {
                        int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                        for (int vc = 0; vc < vld.size.z; vc++) {
                            int visibleCellIndex = vc + visibleCellsStart;

                            int wi = vc + wiStart;

                            vl.traces[wi] *= traceDecay;

                            if (hc == targetCI)
                                vl.traces[wi] = max(vl.traces[wi], vl.inputActsPrev[visibleCellIndex]);

                            vl.weights[wi] += delta * vl.traces[wi];
                        }
                    }
                }
        }
    }
}

void Actor::generateErrors(
    const Int2 &columnPos,
    const IntBuffer* hiddenTargetCIs,
    FloatBuffer* visibleErrors,
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

    if (vl.inputActsPrev[visibleCellsStart] == -1.0f) {
        int inCIPrev = vl.inputCIsPrev[visibleColumnIndex];

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

                    int hiddenCellIndexTarget = (*hiddenTargetCIs)[hiddenColumnIndex] + hiddenColumnIndex * hiddenSize.z;

                    sum += hiddenTDErrors[hiddenColumnIndex] * vl.weights[inCIPrev + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndexTarget))];
                    count++;
                }
            }

        sum /= max(1, count);

        (*visibleErrors)[visibleCellIndex] += sum;
    }
    else {
        for (int vc = 0; vc < vld.size.z; vc++) {
            int visibleCellIndex = vc + visibleCellsStart;

            if (vl.inputActsPrev[visibleCellIndex] == 0.0f)
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

                        int hiddenCellIndexTarget = (*hiddenTargetCIs)[hiddenColumnIndex] + hiddenColumnIndex * hiddenSize.z;

                        sum += hiddenTDErrors[hiddenColumnIndex] * vl.weights[vc + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndexTarget))];
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

        // Create weight matrix for this visible layer and initialize randomly
        vl.weights.resize(numHiddenCells * area * vld.size.z);

        for (int i = 0; i < vl.weights.size(); i++)
            vl.weights[i] = randf(-0.01f, 0.01f);

        vl.traces = FloatBuffer(vl.weights.size(), 0.0f);

        vl.inputCIsPrev = IntBuffer(numVisibleColumns, 0);
        vl.inputActsPrev = FloatBuffer(numVisibleCells, -1.0f); // Flag
    }

    hiddenValues = FloatBuffer(numHiddenCells, 0.0f);

    hiddenTDErrors = FloatBuffer(numHiddenColumns, 0.0f);

    hiddenCIs = IntBuffer(numHiddenColumns, 0);
}

void Actor::activate(
    const Array<const IntBuffer*> &inputCIs,
    const Array<const FloatBuffer*> &inputActs,
    const IntBuffer* hiddenTargetCIs,
    float reward
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        activate(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCIs, inputActs, hiddenTargetCIs, reward);
}

void Actor::learn(
    const IntBuffer* hiddenTargetCIs
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        learn(Int2(i / hiddenSize.y, i % hiddenSize.y), hiddenTargetCIs);
}

void Actor::stepEnd(
    const Array<const IntBuffer*> &inputCIs,
    const Array<const FloatBuffer*> &inputActs
) {
    // Copy to prevs
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];

        vl.inputCIsPrev = *inputCIs[vli];
        
        if (inputActs[vli] == nullptr)
            vl.inputActsPrev.fill(-1.0f); // Flag
        else
            vl.inputActsPrev = *inputActs[vli];
    }
}

void Actor::generateErrors(
    const IntBuffer* hiddenTargetCIs,
    FloatBuffer* visibleErrors,
    int vli
) {
    const VisibleLayer &vl = visibleLayers[vli];
    const VisibleLayerDesc &vld = visibleLayerDescs[vli];

    int numVisibleColumns = vld.size.x * vld.size.y;

    #pragma omp parallel for
    for (int i = 0; i < numVisibleColumns; i++)
        generateErrors(Int2(i / vld.size.y, i % vld.size.y), hiddenTargetCIs, visibleErrors, vli);
}

int Actor::size() const {
    int size = sizeof(Int3) + 3 * sizeof(float) + hiddenValues.size() * sizeof(float) + hiddenTDErrors.size() * sizeof(float) + hiddenCIs.size() * sizeof(int) + sizeof(int);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        size += sizeof(VisibleLayerDesc) + 2 * vl.weights.size() * sizeof(float) + vl.inputCIsPrev.size() * sizeof(int) + vl.inputActsPrev.size() * sizeof(float);
    }

    return size;
}

int Actor::stateSize() const {
    int size = hiddenValues.size() * sizeof(float) + hiddenTDErrors.size() * sizeof(float) + hiddenCIs.size() * sizeof(int);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        size += vl.traces.size() * sizeof(float) + vl.inputCIsPrev.size() * sizeof(int) + vl.inputActsPrev.size() * sizeof(float);
    }

    return size;
}

void Actor::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&discount), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&traceDecay), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenValues[0]), hiddenValues.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&hiddenTDErrors[0]), hiddenTDErrors.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    int numVisibleCellsLayers = visibleLayers.size();

    writer.write(reinterpret_cast<const void*>(&numVisibleCellsLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

        writer.write(reinterpret_cast<const void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));
        writer.write(reinterpret_cast<const void*>(&vl.traces[0]), vl.traces.size() * sizeof(float));

        writer.write(reinterpret_cast<const void*>(&vl.inputCIsPrev[0]), vl.inputCIsPrev.size() * sizeof(int));
        writer.write(reinterpret_cast<const void*>(&vl.inputActsPrev[0]), vl.inputActsPrev.size() * sizeof(float));
    }
}

void Actor::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;
    
    reader.read(reinterpret_cast<void*>(&lr), sizeof(float));
    reader.read(reinterpret_cast<void*>(&discount), sizeof(float));
    reader.read(reinterpret_cast<void*>(&traceDecay), sizeof(float));

    hiddenValues.resize(numHiddenCells);
    hiddenTDErrors.resize(numHiddenColumns);
    hiddenCIs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenValues[0]), hiddenValues.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&hiddenTDErrors[0]), hiddenTDErrors.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

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

        vl.weights.resize(numHiddenCells * area * vld.size.z);
        vl.traces.resize(vl.weights.size());

        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));
        reader.read(reinterpret_cast<void*>(&vl.traces[0]), vl.traces.size() * sizeof(float));

        vl.inputCIsPrev.resize(numVisibleColumns);
        vl.inputActsPrev.resize(numVisibleCells);

        reader.read(reinterpret_cast<void*>(&vl.inputCIsPrev[0]), vl.inputCIsPrev.size() * sizeof(int));
        reader.read(reinterpret_cast<void*>(&vl.inputActsPrev[0]), vl.inputActsPrev.size() * sizeof(float));
    }
}

void Actor::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenValues[0]), hiddenValues.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&hiddenTDErrors[0]), hiddenTDErrors.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];

        writer.write(reinterpret_cast<const void*>(&vl.traces[0]), vl.traces.size() * sizeof(float));

        writer.write(reinterpret_cast<const void*>(&vl.inputCIsPrev[0]), vl.inputCIsPrev.size() * sizeof(int));
        writer.write(reinterpret_cast<const void*>(&vl.inputActsPrev[0]), vl.inputActsPrev.size() * sizeof(float));
    }
}

void Actor::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenValues[0]), hiddenValues.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&hiddenTDErrors[0]), hiddenTDErrors.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];

        reader.read(reinterpret_cast<void*>(&vl.traces[0]), vl.traces.size() * sizeof(float));

        reader.read(reinterpret_cast<void*>(&vl.inputCIsPrev[0]), vl.inputCIsPrev.size() * sizeof(int));
        reader.read(reinterpret_cast<void*>(&vl.inputActsPrev[0]), vl.inputActsPrev.size() * sizeof(float));
    }
}
