// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Encoder.h"

using namespace aon;

void Encoder::activate(
    const Int2 &columnPos
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

            if (!vl.useInput)
                continue;

            if (vl.needsUpdate) {
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

                float subSum = 0.0f;
                int subCount = (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1);

                for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                    for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                        int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                        Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                        int inCI = vl.inputCIs[visibleColumnIndex];

                        int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                        subSum += vl.weights[inCI + wiStart];
                    }

                subSum /= subCount;

                vl.partialActs[hiddenCellIndex] = subSum;

                vl.needsUpdate = false;
            }

            sum += vl.partialActs[hiddenCellIndex];
        }

        if (sum > maxActivation || maxIndex == -1) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    hiddenCIs[hiddenColumnIndex] = maxIndex;
}

void Encoder::learn(
    const Int2 &columnPos,
    int vli
) {
    VisibleLayer &vl = visibleLayers[vli];
    VisibleLayerDesc &vld = visibleLayerDescs[vli];

    if (!vl.useInput)
        return;

    int diam = vld.radius * 2 + 1;

    int visibleColumnIndex = address2(columnPos, Int2(vld.size.x, vld.size.y));

    int visibleCellsStart = visibleColumnIndex * vld.size.z;

    int targetCI = vl.inputCIs[visibleColumnIndex];

    // Projection
    Float2 vToH = Float2(static_cast<float>(hiddenSize.x) / static_cast<float>(vld.size.x),
        static_cast<float>(hiddenSize.y) / static_cast<float>(vld.size.y));

    Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
        static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));

    Int2 reverseRadii(ceilf(vToH.x * (vld.radius * 2 + 1) * 0.5f), ceilf(vToH.y * (vld.radius * 2 + 1) * 0.5f));

    Int2 hiddenCenter = project(columnPos, vToH);

    // Lower corner
    Int2 fieldLowerBound(hiddenCenter.x - reverseRadii.x, hiddenCenter.y - reverseRadii.y);

    // Bounds of receptive field, clamped to input size
    Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
    Int2 iterUpperBound(min(hiddenSize.x - 1, hiddenCenter.x + reverseRadii.x), min(hiddenSize.y - 1, hiddenCenter.y + reverseRadii.y));

    for (int vc = 0; vc < vld.size.z; vc++) {
        int visibleCellIndex = vc + visibleCellsStart;

        float sum = 0.0f;
        int count = 0;

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                Int2 hiddenPos = Int2(ix, iy);

                int hiddenColumnIndex = address2(hiddenPos, Int2(hiddenSize.x, hiddenSize.y));

                Int2 visibleCenter = project(hiddenPos, hToV);

                if (inBounds(columnPos, Int2(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius), Int2(visibleCenter.x + vld.radius + 1, visibleCenter.y + vld.radius + 1))) {
                    Int2 offset(columnPos.x - visibleCenter.x + vld.radius, columnPos.y - visibleCenter.y + vld.radius);

                    int hiddenCellIndexMax = hiddenCIs[hiddenColumnIndex] + hiddenColumnIndex * hiddenSize.z;

                    int wi = vc + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndexMax));

                    sum += vl.weights[wi];
                    count++;
                }
            }

        sum /= max(1, count);

        float delta = lr * ((vc == targetCI) - expf(sum));
  
        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                Int2 hiddenPos = Int2(ix, iy);

                int hiddenColumnIndex = address2(hiddenPos, Int2(hiddenSize.x, hiddenSize.y));

                Int2 visibleCenter = project(hiddenPos, hToV);

                if (inBounds(columnPos, Int2(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius), Int2(visibleCenter.x + vld.radius + 1, visibleCenter.y + vld.radius + 1))) {
                    Int2 offset(columnPos.x - visibleCenter.x + vld.radius, columnPos.y - visibleCenter.y + vld.radius);

                    int hiddenCellIndexMax = hiddenCIs[hiddenColumnIndex] + hiddenColumnIndex * hiddenSize.z;

                    int wi = vc + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndexMax));

                    vl.weights[wi] += delta;
                }
            }
    }
}

void Encoder::reconstruct(
    const Int2 &columnPos,
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

    Int2 reverseRadii(ceilf(vToH.x * (vld.radius * 2 + 1) * 0.5f), ceilf(vToH.y * (vld.radius * 2 + 1) * 0.5f));

    Int2 hiddenCenter = project(columnPos, vToH);

    // Lower corner
    Int2 fieldLowerBound(hiddenCenter.x - reverseRadii.x, hiddenCenter.y - reverseRadii.y);

    // Bounds of receptive field, clamped to input size
    Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
    Int2 iterUpperBound(min(hiddenSize.x - 1, hiddenCenter.x + reverseRadii.x), min(hiddenSize.y - 1, hiddenCenter.y + reverseRadii.y));

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int vc = 0; vc < vld.size.z; vc++) {
        int visibleCellIndex = vc + visibleCellsStart;

        float sum = 0.0f;

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                Int2 hiddenPos = Int2(ix, iy);

                int hiddenColumnIndex = address2(hiddenPos, Int2(hiddenSize.x, hiddenSize.y));

                Int2 visibleCenter = project(hiddenPos, hToV);

                if (inBounds(columnPos, Int2(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius), Int2(visibleCenter.x + vld.radius + 1, visibleCenter.y + vld.radius + 1))) {
                    Int2 offset(columnPos.x - visibleCenter.x + vld.radius, columnPos.y - visibleCenter.y + vld.radius);

                    int hiddenCellIndexMax = hiddenCIs[hiddenColumnIndex] + hiddenColumnIndex * hiddenSize.z;

                    int wi = vc + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndexMax));

                    sum += vl.weights[wi];
                }
            }

        if (sum > maxActivation || maxIndex == -1) {
            maxActivation = sum;
            maxIndex = vc;
        }
    }

    vl.reconCIs[visibleColumnIndex] = maxIndex;
}

void Encoder::initRandom(
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
        const VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisibleCells = numVisibleColumns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(numHiddenCells * area * vld.size.z);

        for (int i = 0; i < vl.weights.size(); i++)
            vl.weights[i] = randf(-1.0f, 0.0f);

        vl.partialActs = FloatBuffer(numHiddenCells, 0.0f);

        vl.inputCIs = IntBuffer(numVisibleColumns, 0);
        vl.reconCIs = IntBuffer(numVisibleColumns, 0);
    }

    hiddenCIs = IntBuffer(numHiddenColumns, 0);
}

void Encoder::step(
    bool learnEnabled
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        activate(Int2(i / hiddenSize.y, i % hiddenSize.y));

    if (learnEnabled) {
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int numVisibleColumns = vld.size.x * vld.size.y;
        
            #pragma omp parallel for
            for (int i = 0; i < numVisibleColumns; i++)
                learn(Int2(i / vld.size.y, i % vld.size.y), vli);
        }
    }
}

void Encoder::reconstruct(
    int vli
) {
    const VisibleLayerDesc &vld = visibleLayerDescs[vli];

    int numVisibleColumns = vld.size.x * vld.size.y;

    #pragma omp parallel for
    for (int i = 0; i < numVisibleColumns; i++)
        reconstruct(Int2(i / vld.size.y, i % vld.size.y), vli);
}

int Encoder::size() const {
    int size = sizeof(Int3) + sizeof(float) + hiddenCIs.size() * sizeof(int) + sizeof(int);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];

        size += sizeof(VisibleLayerDesc) + vl.weights.size() * sizeof(float) + vl.partialActs.size() * sizeof(float) + 2 * vl.inputCIs.size() * sizeof(int) + 2 * sizeof(Byte) + sizeof(float);
    }

    return size;
}

int Encoder::stateSize() const {
    int size = hiddenCIs.size() * sizeof(float);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];

        size += vl.partialActs.size() * sizeof(float) + 2 * vl.inputCIs.size() * sizeof(int) + 2 * sizeof(Byte);
    }

    return size;
}

void Encoder::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    int numVisibleLayers = visibleLayers.size();

    writer.write(reinterpret_cast<const void*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

        writer.write(reinterpret_cast<const void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));

        writer.write(reinterpret_cast<const void*>(&vl.partialActs[0]), vl.partialActs.size() * sizeof(float));

        writer.write(reinterpret_cast<const void*>(&vl.inputCIs[0]), vl.inputCIs.size() * sizeof(int));
        writer.write(reinterpret_cast<const void*>(&vl.reconCIs[0]), vl.reconCIs.size() * sizeof(int));

        writer.write(reinterpret_cast<const void*>(&vl.useInput), sizeof(Byte));
        writer.write(reinterpret_cast<const void*>(&vl.needsUpdate), sizeof(Byte));

        writer.write(reinterpret_cast<const void*>(&vl.importance), sizeof(float));
    }
}

void Encoder::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&lr), sizeof(float));

    hiddenCIs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    int numVisibleLayers = visibleLayers.size();

    reader.read(reinterpret_cast<void*>(&numVisibleLayers), sizeof(int));

    visibleLayers.resize(numVisibleLayers);
    visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = visibleLayerDescs[vli];

        reader.read(reinterpret_cast<void*>(&vld), sizeof(VisibleLayerDesc));

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisibleCells = numVisibleColumns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(numHiddenCells * area * vld.size.z);

        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));

        vl.partialActs.resize(numHiddenCells);

        reader.read(reinterpret_cast<void*>(&vl.partialActs[0]), vl.partialActs.size() * sizeof(float));

        vl.inputCIs.resize(numVisibleColumns);
        vl.reconCIs.resize(numVisibleColumns);

        reader.read(reinterpret_cast<void*>(&vl.inputCIs[0]), vl.inputCIs.size() * sizeof(int));
        reader.read(reinterpret_cast<void*>(&vl.reconCIs[0]), vl.reconCIs.size() * sizeof(int));

        reader.read(reinterpret_cast<void*>(&vl.useInput), sizeof(Byte));
        reader.read(reinterpret_cast<void*>(&vl.needsUpdate), sizeof(Byte));

        reader.read(reinterpret_cast<void*>(&vl.importance), sizeof(float));
    }
}

void Encoder::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];

        writer.write(reinterpret_cast<const void*>(&vl.partialActs[0]), vl.partialActs.size() * sizeof(float));

        writer.write(reinterpret_cast<const void*>(&vl.inputCIs[0]), vl.inputCIs.size() * sizeof(int));
        writer.write(reinterpret_cast<const void*>(&vl.reconCIs[0]), vl.reconCIs.size() * sizeof(int));

        writer.write(reinterpret_cast<const void*>(&vl.useInput), sizeof(Byte));
        writer.write(reinterpret_cast<const void*>(&vl.needsUpdate), sizeof(Byte));
    }
}

void Encoder::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];

        reader.read(reinterpret_cast<void*>(&vl.partialActs[0]), vl.partialActs.size() * sizeof(float));

        reader.read(reinterpret_cast<void*>(&vl.inputCIs[0]), vl.inputCIs.size() * sizeof(int));
        reader.read(reinterpret_cast<void*>(&vl.reconCIs[0]), vl.reconCIs.size() * sizeof(int));

        reader.read(reinterpret_cast<void*>(&vl.useInput), sizeof(Byte));
        reader.read(reinterpret_cast<void*>(&vl.needsUpdate), sizeof(Byte));
    }
}

