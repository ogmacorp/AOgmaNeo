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
    const IntBuffer* goalCIs,
    const FloatBuffer* inputActs
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        float sum = 0.0f;

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
                int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                {
                    int inCI = (*goalCIs)[visibleColumnIndex];

                    int wi = inCI + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                    sum += vl.weights[wi];
                }

                {
                    int visibleCellsStart = visibleColumnIndex * vld.size.z;

                    int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                    for (int vc = 0; vc < vld.size.z; vc++) {
                        int visibleCellIndex = vc + visibleCellsStart;

                        float inAct = (*inputActs)[visibleCellIndex];

                        int wi = vc + wiStart;

                        sum += vl.weightsPrev[wi] * inAct;
                    }
                }
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
    const FloatBuffer* inputActsPrev
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    int targetCI = (*hiddenTargetCIs)[hiddenColumnIndex];

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        float sum = 0.0f;

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

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                {
                    int inCI = (*inputCIs)[visibleColumnIndex];

                    int wi = inCI + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                    sum += vl.weights[wi];
                }

                {
                    int visibleCellsStart = visibleColumnIndex * vld.size.z;

                    int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                    for (int vc = 0; vc < vld.size.z; vc++) {
                        int visibleCellIndex = vc + visibleCellsStart;

                        float inAct = (*inputActsPrev)[visibleCellIndex];

                        int wi = vc + wiStart;

                        sum += vl.weightsPrev[wi] * inAct;
                    }
                }
            }

        sum /= count;

        hiddenActivations[hiddenCellIndex] = min(1.0f, max(0.0f, sum));

        float delta = lr * ((hc == targetCI) - hiddenActivations[hiddenCellIndex]);

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                {
                    int inCI = (*inputCIs)[visibleColumnIndex];

                    int wi = inCI + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                    vl.weights[wi] += delta;
                }

                {
                    int visibleCellsStart = visibleColumnIndex * vld.size.z;

                    int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                    for (int vc = 0; vc < vld.size.z; vc++) {
                        int visibleCellIndex = vc + visibleCellsStart;

                        float inAct = (*inputActsPrev)[visibleCellIndex];

                        int wi = vc + wiStart;

                        vl.weightsPrev[wi] += delta * inAct;
                    }
                }
            }
    }
}

void Decoder::generateErrors(
    const Int2 &columnPos,
    const IntBuffer* hiddenTargetCIs,
    FloatBuffer* visibleErrors
) {
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

                    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

                    for (int hc = 0; hc < hiddenSize.z; hc++) {
                        int hiddenCellIndex = hc + hiddenCellsStart;

                        float error = (hc == (*hiddenTargetCIs)[hiddenColumnIndex]) - hiddenActivations[hiddenCellIndex];

                        sum += error * vl.weightsPrev[vc + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex))];
                    }

                    count++;
                }
            }

        sum /= max(1, count);

        (*visibleErrors)[visibleCellIndex] += sum;
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
    
    // Create layers
    int numVisibleColumns = vld.size.x * vld.size.y;
    int numVisibleCells = numVisibleColumns * vld.size.z;

    int diam = vld.radius * 2 + 1;
    int area = diam * diam;

    vl.weights.resize(numHiddenCells * area * vld.size.z);
    vl.weightsPrev.resize(vl.weights.size());

    for (int i = 0; i < vl.weights.size(); i++) {
        vl.weights[i] = randf(0.0f, 0.01f);
        vl.weightsPrev[i] = randf(0.0f, 0.01f);
    }

    hiddenActivations = FloatBuffer(numHiddenCells, 0.0f);

    // Hidden CIs
    hiddenCIs = IntBuffer(numHiddenColumns, 0);
}

void Decoder::activate(
    const IntBuffer* goalCIs,
    const FloatBuffer* inputActs
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    // Forward kernel
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), goalCIs, inputActs);
}

void Decoder::learn(
    const IntBuffer* hiddenTargetCIs,
    const IntBuffer* inputCIs,
    const FloatBuffer* inputActsPrev
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    // Learn kernel
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        learn(Int2(i / hiddenSize.y, i % hiddenSize.y), hiddenTargetCIs, inputCIs, inputActsPrev);
}

void Decoder::generateErrors(
    const IntBuffer* hiddenTargetCIs,
    FloatBuffer* visibleErrors,
    int vli
) {
    int numVisibleColumns = vld.size.x * vld.size.y;

    #pragma omp parallel for
    for (int i = 0; i < numVisibleColumns; i++)
        generateErrors(Int2(i / vld.size.y, i % vld.size.y), hiddenTargetCIs, visibleErrors);
}

int Decoder::size() const {
    int size = sizeof(Int3) + sizeof(float) + hiddenActivations.size() * sizeof(float) + hiddenCIs.size() * sizeof(int);

    size += sizeof(VisibleLayerDesc) + vl.weights.size() * sizeof(float);

    return size;
}

int Decoder::stateSize() const {
    return hiddenActivations.size() * sizeof(float) + hiddenCIs.size() * sizeof(int);
}

void Decoder::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenActivations[0]), hiddenActivations.size() * sizeof(float));
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

    hiddenActivations.resize(numHiddenCells);
    hiddenCIs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenActivations[0]), hiddenActivations.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    reader.read(reinterpret_cast<void*>(&vld), sizeof(VisibleLayerDesc));

    int numVisibleColumns = vld.size.x * vld.size.y;
    int numVisibleCells = numVisibleColumns * vld.size.z;

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
    writer.write(reinterpret_cast<const void*>(&hiddenActivations[0]), hiddenActivations.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
}

void Decoder::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenActivations[0]), hiddenActivations.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
}
