// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Decoder.h"

using namespace aon;

void Decoder::forward(
    const Int2 &columnPos,
    const IntBuffer* nextCIs,
    const IntBuffer* inputCIs
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

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

        float sum = 0.0f;

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                if (nextCIs != nullptr) {
                    int nextCI = (*nextCIs)[visibleColumnIndex];

                    sum += vl.weights[nextCI + wiStart];
                }

                int inCI = (*inputCIs)[visibleColumnIndex];

                sum += vl.weightsPrev[inCI + wiStart];
            }

        if (sum > maxActivation || maxIndex == -1) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    hiddenCIs[hiddenColumnIndex] = maxIndex;
}

void Decoder::learn(
    const Int2 &columnPos,
    const IntBuffer* hiddenTargetCIs,
    const IntBuffer* inputCIs,
    const IntBuffer* inputCIsPrev
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    int targetCI = (*hiddenTargetCIs)[hiddenColumnIndex];

    int maxIndex = -1;
    float maxActivation = -999999.0f;

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

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        float sum = 0.0f;
        int count = (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1);

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                int inCI = (*inputCIs)[visibleColumnIndex];
                int inCIPrev = (*inputCIsPrev)[visibleColumnIndex];

                sum += vl.weights[inCI + wiStart] + vl.weightsPrev[inCIPrev + wiStart];
            }

        sum /= count;

        hiddenActs[hiddenCellIndex] = sum;

        if (sum > maxActivation || maxIndex == -1) {
            maxActivation = sum;
            maxIndex = hc;
        }
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

        float delta = lr * ((hc == targetCI) - hiddenActs[hiddenCellIndex]);

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                int inCI = (*inputCIs)[visibleColumnIndex];
                int inCIPrev = (*inputCIsPrev)[visibleColumnIndex];

                vl.weights[inCI + wiStart] += delta;
                vl.weightsPrev[inCIPrev + wiStart] += delta;
            }
    }
}

void Decoder::initRandom(
    const Int3 &hiddenSize,
    const VisibleLayerDesc &vld
) {
    this->vld = vld; 

    this->hiddenSize = hiddenSize;

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;
    
    int numVisibleColumns = vld.size.x * vld.size.y;

    int diam = vld.radius * 2 + 1;
    int area = diam * diam;

    vl.weights.resize(numHiddenCells * area * vld.size.z);
    vl.weightsPrev.resize(vl.weights.size());

    for (int i = 0; i < vl.weights.size(); i++) {
        vl.weights[i] = randf(-0.01f, 0.01f);
        vl.weightsPrev[i] = randf(-0.01f, 0.01f);
    }

    hiddenActs = FloatBuffer(numHiddenCells, 0.0f);

    hiddenCIs = IntBuffer(numHiddenColumns, 0);
}

void Decoder::activate(
    const IntBuffer* nextCIs,
    const IntBuffer* inputCIs
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    // Forward kernel
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), nextCIs, inputCIs);
}

void Decoder::learn(
    const IntBuffer* hiddenTargetCIs,
    const IntBuffer* inputCIs,
    const IntBuffer* inputCIsPrev
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    
    // Learn kernel
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        learn(Int2(i / hiddenSize.y, i % hiddenSize.y), hiddenTargetCIs, inputCIs, inputCIsPrev);
}

void Decoder::clearState() {
    hiddenCIs.fill(0);
}

int Decoder::size() const {
    int size = sizeof(Int3) + sizeof(float) + hiddenCIs.size() * sizeof(int) + sizeof(int);

    size += sizeof(VisibleLayerDesc) + 2 * vl.weights.size() * sizeof(float);

    return size;
}

int Decoder::stateSize() const {
    return hiddenCIs.size() * sizeof(int);
}

void Decoder::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    
    writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

    writer.write(reinterpret_cast<const void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&vl.weightsPrev[0]), vl.weightsPrev.size() * sizeof(float));
}

void Decoder::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&lr), sizeof(float));

    hiddenActs = FloatBuffer(numHiddenCells, 0.0f);

    hiddenCIs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    reader.read(reinterpret_cast<void*>(&vld), sizeof(VisibleLayerDesc));

    int numVisibleColumns = vld.size.x * vld.size.y;

    int diam = vld.radius * 2 + 1;
    int area = diam * diam;

    vl.weights.resize(numHiddenCells * area * vld.size.z);
    vl.weightsPrev.resize(vl.weights.size());

    reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&vl.weightsPrev[0]), vl.weightsPrev.size() * sizeof(float));
}

void Decoder::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
}

void Decoder::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
}
