// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Encoder.h"

using namespace aon;

void Encoder::forward(
    const Int2 &columnPos
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    int maxIndex = -1;
    float maxActivation = 0.0f;

    int maxBackupIndex = -1;
    float maxBackupActivation = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        float sum = 0.0f;
        float totalImportance = 0.0f;

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];

            if (!vl.useInputs)
                continue;

            if (vl.needsUpdate) {
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

                float subSum = 0.0f;
                int subCount = (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1);

                for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                    for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                        int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                        int inCI = vl.inputCIs[visibleColumnIndex];

                        Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                        int wi = inCI + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                        subSum += vl.weights[wi];
                    }

                subSum /= subCount;

                vl.partialActs[hiddenCellIndex] = subSum;
            }

            sum += vl.partialActs[hiddenCellIndex] * vl.importance;
            totalImportance += vl.importance;
        }

        sum /= max(0.0001f, totalImportance);

        float activation = sum / (gap + hiddenTotals[hiddenCellIndex]);

        if (sum >= vigilance) {
            if (activation > maxActivation || maxIndex == -1) {
                maxActivation = activation;
                maxIndex = hc;
            }
        }

        if (activation > maxBackupActivation || maxBackupIndex == -1) {
            maxBackupActivation = activation;
            maxBackupIndex = hc;
        }
    }

    learnCIs[hiddenColumnIndex] = maxIndex;

    hiddenCIs[hiddenColumnIndex] = maxBackupIndex;

    hiddenMaxActs[hiddenColumnIndex] = maxBackupActivation;
}

void Encoder::learn(
    const Int2 &columnPos
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    if (learnCIs[hiddenColumnIndex] == -1)
        return;

    float maxActivation = hiddenMaxActs[hiddenColumnIndex];

    for (int dcx = -lRadius; dcx <= lRadius; dcx++)
        for (int dcy = -lRadius; dcy <= lRadius; dcy++) {
            Int2 otherColumnPos(columnPos.x + dcx, columnPos.y + dcy);

            if (inBounds0(otherColumnPos, Int2(hiddenSize.x, hiddenSize.y))) {
                int otherHiddenColumnIndex = address2(otherColumnPos, Int2(hiddenSize.x, hiddenSize.y));

                if (hiddenMaxActs[otherHiddenColumnIndex] > maxActivation)
                    return;
            }
        }

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    int hiddenCellIndexMax = learnCIs[hiddenColumnIndex] + hiddenCellsStart;

    float total = 0.0f;
    float totalImportance = 0.0f;

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        assert(vl.useInputs);

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

        float subTotal = 0.0f;
        int subCount = (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1) * vld.size.z;

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int inCI = vl.inputCIs[visibleColumnIndex];

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndexMax));

                for (int vc = 0; vc < vld.size.z; vc++) {
                    int wi = vc + wiStart;

                    if (vc != inCI)
                        vl.weights[wi] -= lr * vl.weights[wi];

                    subTotal += vl.weights[wi];
                }
            }

        subTotal /= subCount;

        total += subTotal * vl.importance;
        totalImportance += vl.importance;
    }

    total /= max(0.0001f, totalImportance);

    hiddenTotals[hiddenCellIndexMax] = sqrtf(total);
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
    float maxActivation = 0.0f;

    for (int vc = 0; vc < vld.size.z; vc++) {
        float sum = 0.0f;
        int count = 0;

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                Int2 hiddenPos = Int2(ix, iy);

                int hiddenColumnIndex = address2(hiddenPos, Int2(hiddenSize.x, hiddenSize.y));

                int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

                int hiddenCellIndex = hiddenCIs[hiddenColumnIndex] + hiddenCellsStart;

                Int2 visibleCenter = project(hiddenPos, hToV);

                if (inBounds(columnPos, Int2(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius), Int2(visibleCenter.x + vld.radius + 1, visibleCenter.y + vld.radius + 1))) {
                    Int2 offset(columnPos.x - visibleCenter.x + vld.radius, columnPos.y - visibleCenter.y + vld.radius);

                    int wi = vc + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                    sum += vl.weights[wi];
                    count++;
                }
            }

        sum /= count;

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

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(numHiddenCells * area * vld.size.z);

        // Initialize to random values
        for (int i = 0; i < vl.weights.size(); i++)
            vl.weights[i] = randf(0.0f, 1.0f);

        vl.partialActs = FloatBuffer(numHiddenCells, 0.0f);

        vl.inputCIs = IntBuffer(numVisibleColumns, 0);
        vl.reconCIs = IntBuffer(numVisibleColumns, 0);
    }

    hiddenTotals = FloatBuffer(numHiddenCells, 1.0f);
    hiddenMaxActs = FloatBuffer(numHiddenColumns);

    hiddenCIs = IntBuffer(numHiddenColumns, 0);
    learnCIs = IntBuffer(numHiddenColumns, -1);
}

void Encoder::setInputCIs(
    const IntBuffer* inputCIs,
    int vli
) {
    if (inputCIs != nullptr) {
        visibleLayers[vli].inputCIs = *inputCIs;
        visibleLayers[vli].useInputs = true;
        visibleLayers[vli].needsUpdate = true;
    }
    else
        visibleLayers[vli].useInputs = false;
}

void Encoder::activate() {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    
    // Activate
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        forward(Int2(i / hiddenSize.y, i % hiddenSize.y));

    // Clear updated layers
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];

        if (vl.useInputs)
            vl.needsUpdate = false;
    }
}

void Encoder::learn() {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        learn(Int2(i / hiddenSize.y, i % hiddenSize.y));
}

void Encoder::reconstruct(
    int vli
) {
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;

        #pragma omp parallel for
        for (int i = 0; i < numVisibleColumns; i++)
            reconstruct(Int2(i / vld.size.y, i % vld.size.y), vli);
    }
}

int Encoder::size() const {
    int size = sizeof(Int3) + sizeof(float) + hiddenTotals.size() * sizeof(float) + hiddenCIs.size() * sizeof(int) + sizeof(int);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];

        size += sizeof(VisibleLayerDesc) + vl.weights.size() * sizeof(float) + 2 * vl.inputCIs.size() * sizeof(int) + sizeof(float);
    }

    return size;
}

int Encoder::stateSize() const {
    int size = hiddenCIs.size() * sizeof(int);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];

        size += 2 * vl.inputCIs.size() * sizeof(int);
    }

    return size;
}

void Encoder::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&gap), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&vigilance), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&lRadius), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&hiddenTotals[0]), hiddenTotals.size() * sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    
    int numVisibleLayers = visibleLayers.size();

    writer.write(reinterpret_cast<const void*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

        writer.write(reinterpret_cast<const void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));

        writer.write(reinterpret_cast<const void*>(&vl.inputCIs[0]), vl.inputCIs.size() * sizeof(int));
        writer.write(reinterpret_cast<const void*>(&vl.reconCIs[0]), vl.reconCIs.size() * sizeof(int));

        writer.write(reinterpret_cast<const void*>(&vl.importance), sizeof(float));
    }
}

void Encoder::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&gap), sizeof(float));
    reader.read(reinterpret_cast<void*>(&vigilance), sizeof(float));
    reader.read(reinterpret_cast<void*>(&lr), sizeof(float));
    reader.read(reinterpret_cast<void*>(&lRadius), sizeof(int));

    hiddenTotals.resize(numHiddenCells);

    reader.read(reinterpret_cast<void*>(&hiddenTotals[0]), hiddenTotals.size() * sizeof(float));

    hiddenMaxActs = FloatBuffer(numHiddenColumns);

    hiddenCIs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    learnCIs = IntBuffer(numHiddenColumns, -1);

    int numVisibleLayers;

    reader.read(reinterpret_cast<void*>(&numVisibleLayers), sizeof(int));

    visibleLayers.resize(numVisibleLayers);
    visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = visibleLayerDescs[vli];

        reader.read(reinterpret_cast<void*>(&vld), sizeof(VisibleLayerDesc));

        int numVisibleColumns = vld.size.x * vld.size.y;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(numHiddenCells * area * vld.size.z);

        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));

        vl.partialActs = FloatBuffer(numHiddenCells, 0.0f);

        vl.inputCIs.resize(numVisibleColumns);
        vl.reconCIs.resize(numVisibleColumns);

        reader.read(reinterpret_cast<void*>(&vl.inputCIs[0]), vl.inputCIs.size() * sizeof(int));
        reader.read(reinterpret_cast<void*>(&vl.reconCIs[0]), vl.reconCIs.size() * sizeof(int));

        reader.read(reinterpret_cast<void*>(&vl.importance), sizeof(float));

        vl.useInputs = false;
        vl.needsUpdate = true;
    }
}

void Encoder::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];

        writer.write(reinterpret_cast<const void*>(&vl.inputCIs[0]), vl.inputCIs.size() * sizeof(int));
        writer.write(reinterpret_cast<const void*>(&vl.reconCIs[0]), vl.reconCIs.size() * sizeof(int));
    }
}

void Encoder::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;

        vl.inputCIs.resize(numVisibleColumns);
        vl.reconCIs.resize(numVisibleColumns);

        reader.read(reinterpret_cast<void*>(&vl.inputCIs[0]), vl.inputCIs.size() * sizeof(int));
        reader.read(reinterpret_cast<void*>(&vl.reconCIs[0]), vl.reconCIs.size() * sizeof(int));

        vl.useInputs = false;
        vl.needsUpdate = true;
    }
}
