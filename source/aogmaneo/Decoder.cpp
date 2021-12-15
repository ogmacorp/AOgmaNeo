// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Decoder.h"

using namespace aon;

void Decoder::forward(
    const Int2 &columnPos,
    const IntBuffer* progCIs,
    const IntBuffer* inputCIs
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    int diam = vld.radius * 2 + 1;

    // Projection
    Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
        static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));

    Int2 visibleCenter = project(columnPos, hToV);

    visibleCenter = minOverhang(visibleCenter, Int2(vld.size.x, vld.size.y), vld.radius);

    // Lower corner
    Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

    // Bounds of receptive field, clamped to input size
    Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
    Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        float sum = 0.0f;

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                int inCI = (*progCIs)[visibleColumnIndex];
                int inCIPrev = (*inputCIs)[visibleColumnIndex];

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int wi = inCI + vld.size.z * (inCIPrev + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex)));

                sum += vl.weights[wi];
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
    int t1,
    int t2
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    int targetCI = history[t1 - 1].hiddenTargetCIs[hiddenColumnIndex];

    // Pre-count
    int diam = vld.radius * 2 + 1;

    // Projection
    Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
        static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));

    Int2 visibleCenter = project(columnPos, hToV);

    visibleCenter = minOverhang(visibleCenter, Int2(vld.size.x, vld.size.y), vld.radius);

    // Lower corner
    Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

    // Bounds of receptive field, clamped to input size
    Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
    Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

    int count = (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1);

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        float sum = 0.0f;

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int inCI = history[t2].inputCIs[visibleColumnIndex];
                int inCIPrev = history[t1].inputCIs[visibleColumnIndex];

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int wi = inCI + vld.size.z * (inCIPrev + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex)));

                sum += vl.weights[wi];
            }

        sum /= count;

        hiddenActivations[hiddenCellIndex] = sum;

        if (sum > maxActivation || maxIndex == -1) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }
    
    if (maxIndex == targetCI)
        return;

    float total = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        hiddenActivations[hiddenCellIndex] = expf(hiddenActivations[hiddenCellIndex] - maxActivation);

        total += hiddenActivations[hiddenCellIndex];
    }

    float scale = 1.0f / max(0.0001f, total);

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        hiddenActivations[hiddenCellIndex] *= scale;
    }

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        float delta = lr * ((hc == targetCI) - hiddenActivations[hiddenCellIndex]);
            
        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int inCI = history[t2].inputCIs[visibleColumnIndex];
                int inCIPrev = history[t1].inputCIs[visibleColumnIndex];

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int wi = inCI + vld.size.z * (inCIPrev + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex)));

                vl.weights[wi] += delta;
            }
    }
}

void Decoder::initRandom(
    const Int3 &hiddenSize,
    int historyCapacity,
    const VisibleLayerDesc &vld
) {
    this->vld = vld; 

    this->hiddenSize = hiddenSize;

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;
    
    // Create layers
    int diam = vld.radius * 2 + 1;
    int area = diam * diam;

    vl.weights.resize(numHiddenCells * area * vld.size.z * vld.size.z);

    for (int i = 0; i < vl.weights.size(); i++)
        vl.weights[i] = randf(-0.01f, 0.01f);

    // Hidden CIs
    hiddenCIs = IntBuffer(numHiddenColumns, hiddenSize.z / 2);

    hiddenActivations = FloatBuffer(numHiddenCells, 0.0f);

    int numVisibleColumns = vld.size.x * vld.size.y;

    historySize = 0;
    history.resize(historyCapacity);

    for (int i = 0; i < history.size(); i++) {
        history[i].inputCIs = IntBuffer(numVisibleColumns, vld.size.z / 2);
        history[i].hiddenTargetCIs = IntBuffer(numHiddenColumns, hiddenSize.z / 2);
    }
}

void Decoder::step(
    const IntBuffer* progCIs,
    const IntBuffer* inputCIs,
    const IntBuffer* hiddenTargetCIs,
    bool learnEnabled,
    bool stateUpdate
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    if (stateUpdate) {
        history.pushFront();

        // If not at cap, increment
        if (historySize < history.size())
            historySize++;
    
        history[0].inputCIs = *inputCIs;
        history[0].hiddenTargetCIs = *hiddenTargetCIs;

        if (learnEnabled && historySize > 2) {
            for (int it = 0; it < iters; it++) {
                int t1 = rand() % (historySize - 1) + 1;
                int t2 = rand() % t1;

                // Learn kernel
                #pragma omp parallel for
                for (int i = 0; i < numHiddenColumns; i++)
                    learn(Int2(i / hiddenSize.y, i % hiddenSize.y), t1, t2);
            }
        }
    }

    // Forward kernel
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), progCIs, inputCIs);
}

int Decoder::size() const {
    int size = sizeof(Int3) + sizeof(float) + sizeof(int) + hiddenCIs.size() * sizeof(int);

    size += sizeof(VisibleLayerDesc) + vl.weights.size() * sizeof(float);

    size += 3 * sizeof(int) + history.size() * (history[0].inputCIs.size() * sizeof(int) + history[0].hiddenTargetCIs.size() * sizeof(int));

    return size;
}

int Decoder::stateSize() const {
    int size = hiddenCIs.size() * sizeof(int);

    size += sizeof(int) + history.size() * (history[0].inputCIs.size() * sizeof(int) + history[0].hiddenTargetCIs.size() * sizeof(int));

    return size;
}

void Decoder::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&iters), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    
    writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

    writer.write(reinterpret_cast<const void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));

    writer.write(reinterpret_cast<const void*>(&historySize), sizeof(int));

    int numHistory = history.size();

    writer.write(reinterpret_cast<const void*>(&numHistory), sizeof(int));

    int historyStart = history.start;

    writer.write(reinterpret_cast<const void*>(&historyStart), sizeof(int));

    for (int t = 0; t < history.size(); t++) {
        writer.write(reinterpret_cast<const void*>(&history[t].inputCIs[0]), history[t].inputCIs.size() * sizeof(int));
        writer.write(reinterpret_cast<const void*>(&history[t].hiddenTargetCIs[0]), history[t].hiddenTargetCIs.size() * sizeof(int));
    }
}

void Decoder::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&lr), sizeof(float));
    reader.read(reinterpret_cast<void*>(&iters), sizeof(int));

    hiddenCIs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    hiddenActivations = FloatBuffer(numHiddenCells, 0.0f);

    reader.read(reinterpret_cast<void*>(&vld), sizeof(VisibleLayerDesc));

    int diam = vld.radius * 2 + 1;
    int area = diam * diam;

    vl.weights.resize(numHiddenCells * area * vld.size.z * vld.size.z);

    reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));

    reader.read(reinterpret_cast<void*>(&historySize), sizeof(int));

    int numHistory;

    reader.read(reinterpret_cast<void*>(&numHistory), sizeof(int));

    int historyStart;

    reader.read(reinterpret_cast<void*>(&historyStart), sizeof(int));

    history.resize(numHistory);
    history.start = historyStart;

    int numVisibleColumns = vld.size.x * vld.size.y;

    for (int t = 0; t < history.size(); t++) {
        history[t].inputCIs.resize(numVisibleColumns);
        history[t].hiddenTargetCIs.resize(numHiddenColumns);

        reader.read(reinterpret_cast<void*>(&history[t].inputCIs[0]), history[t].inputCIs.size() * sizeof(int));
        reader.read(reinterpret_cast<void*>(&history[t].hiddenTargetCIs[0]), history[t].hiddenTargetCIs.size() * sizeof(int));
    }
}

void Decoder::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    
    int historyStart = history.start;

    writer.write(reinterpret_cast<const void*>(&historyStart), sizeof(int));

    for (int t = 0; t < history.size(); t++) {
        writer.write(reinterpret_cast<const void*>(&history[t].inputCIs[0]), history[t].inputCIs.size() * sizeof(int));
        writer.write(reinterpret_cast<const void*>(&history[t].hiddenTargetCIs[0]), history[t].hiddenTargetCIs.size() * sizeof(int));
    }
}

void Decoder::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    int historyStart;

    reader.read(reinterpret_cast<void*>(&historyStart), sizeof(int));

    history.start = historyStart;

    for (int t = 0; t < history.size(); t++) {
        reader.read(reinterpret_cast<void*>(&history[t].inputCIs[0]), history[t].inputCIs.size() * sizeof(int));
        reader.read(reinterpret_cast<void*>(&history[t].hiddenTargetCIs[0]), history[t].hiddenTargetCIs.size() * sizeof(int));
    }
}
