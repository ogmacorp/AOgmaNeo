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
    const IntBuffer* goalCIs,
    const IntBuffer* inputCIs
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    // Pre-count
    int diam = visibleLayerDesc.radius * 2 + 1;

    // Projection
    Float2 hToV = Float2(static_cast<float>(visibleLayerDesc.size.x) / static_cast<float>(hiddenSize.x),
        static_cast<float>(visibleLayerDesc.size.y) / static_cast<float>(hiddenSize.y));

    Int2 visibleCenter = project(columnPos, hToV);

    visibleCenter = minOverhang(visibleCenter, Int2(visibleLayerDesc.size.x, visibleLayerDesc.size.y), visibleLayerDesc.radius);

    // Lower corner
    Int2 fieldLowerBound(visibleCenter.x - visibleLayerDesc.radius, visibleCenter.y - visibleLayerDesc.radius);

    // Bounds of receptive field, clamped to input size
    Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
    Int2 iterUpperBound(min(visibleLayerDesc.size.x - 1, visibleCenter.x + visibleLayerDesc.radius), min(visibleLayerDesc.size.y - 1, visibleCenter.y + visibleLayerDesc.radius));

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        float sum = 0.0f;

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int visibleColumnIndex = address2(Int2(ix, iy), Int2(visibleLayerDesc.size.x,  visibleLayerDesc.size.y));

                int goalCI = (*goalCIs)[visibleColumnIndex];
                int inCIPrev = (*inputCIs)[visibleColumnIndex];

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int wiStart = visibleLayerDesc.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                sum += visibleLayer.weights[goalCI + visibleLayerDesc.size.z * (inCIPrev + wiStart)];
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
    int t2,
    float reward
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    int targetCI = history[t1 - 1].hiddenTargetCIs[hiddenColumnIndex];

    // Pre-count
    int diam = visibleLayerDesc.radius * 2 + 1;

    // Projection
    Float2 hToV = Float2(static_cast<float>(visibleLayerDesc.size.x) / static_cast<float>(hiddenSize.x),
        static_cast<float>(visibleLayerDesc.size.y) / static_cast<float>(hiddenSize.y));

    Int2 visibleCenter = project(columnPos, hToV);

    visibleCenter = minOverhang(visibleCenter, Int2(visibleLayerDesc.size.x, visibleLayerDesc.size.y), visibleLayerDesc.radius);

    // Lower corner
    Int2 fieldLowerBound(visibleCenter.x - visibleLayerDesc.radius, visibleCenter.y - visibleLayerDesc.radius);

    // Bounds of receptive field, clamped to input size
    Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
    Int2 iterUpperBound(min(visibleLayerDesc.size.x - 1, visibleCenter.x + visibleLayerDesc.radius), min(visibleLayerDesc.size.y - 1, visibleCenter.y + visibleLayerDesc.radius));

    int count = (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1);

    float maxActivation = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        float sum = 0.0f;

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int visibleColumnIndex = address2(Int2(ix, iy), Int2(visibleLayerDesc.size.x,  visibleLayerDesc.size.y));

                int actualCI = history[t2].actualCIs[visibleColumnIndex];
                int inCINext = history[t1 - 1].inputCIs[visibleColumnIndex];

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int wiStart = visibleLayerDesc.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                sum += visibleLayer.weights[actualCI + visibleLayerDesc.size.z * (inCINext + wiStart)];
            }

        sum /= count;

        maxActivation = max(maxActivation, sum);
    }

    int hiddenCellIndexTarget = targetCI + hiddenCellsStart;

    float sumPrev = 0.0f;

    for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
        for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
            int visibleColumnIndex = address2(Int2(ix, iy), Int2(visibleLayerDesc.size.x,  visibleLayerDesc.size.y));

            int actualCI = history[t2].actualCIs[visibleColumnIndex];
            int inCIPrev = history[t1].inputCIs[visibleColumnIndex];

            Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

            int wiStart = visibleLayerDesc.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndexTarget));

            sumPrev += visibleLayer.weights[actualCI + visibleLayerDesc.size.z * (inCIPrev + wiStart)];
        }

    sumPrev /= count;

    float delta = lr * (max(reward, discount * min(1.0f, maxActivation)) - sumPrev);

    for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
        for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
            int visibleColumnIndex = address2(Int2(ix, iy), Int2(visibleLayerDesc.size.x,  visibleLayerDesc.size.y));

            int actualCI = history[t2].actualCIs[visibleColumnIndex];
            int inCIPrev = history[t1].inputCIs[visibleColumnIndex];

            Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

            int wiStart = visibleLayerDesc.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndexTarget));

            visibleLayer.weights[actualCI + visibleLayerDesc.size.z * (inCIPrev + wiStart)] += delta;
        }
}

void Decoder::initRandom(
    const Int3 &hiddenSize,
    int historyCapacity,
    const VisibleLayerDesc &visibleLayerDesc
) {
    this->visibleLayerDesc = visibleLayerDesc; 

    this->hiddenSize = hiddenSize;

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;
    
    // Create layers
    int diam = visibleLayerDesc.radius * 2 + 1;
    int area = diam * diam;

    visibleLayer.weights.resize(numHiddenCells * area * visibleLayerDesc.size.z * visibleLayerDesc.size.z);

    for (int i = 0; i < visibleLayer.weights.size(); i++)
        visibleLayer.weights[i] = randf(0.0f, 0.001f);

    // Hidden CIs
    hiddenCIs = IntBuffer(numHiddenColumns, 0);

    int numVisibleColumns = visibleLayerDesc.size.x * visibleLayerDesc.size.y;

    historySize = 0;
    history.resize(historyCapacity);

    for (int i = 0; i < history.size(); i++) {
        history[i].actualCIs = IntBuffer(numVisibleColumns, 0);
        history[i].inputCIs = IntBuffer(numVisibleColumns, 0);
        history[i].hiddenTargetCIs = IntBuffer(numHiddenColumns, 0);
    }
}

void Decoder::step(
    const IntBuffer* goalCIs,
    const IntBuffer* actualCIs,
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
    
        history[0].actualCIs = *actualCIs;
        history[0].inputCIs = *inputCIs;
        history[0].hiddenTargetCIs = *hiddenTargetCIs;

        if (learnEnabled && historySize > 1) {
            for (int it = 0; it < historyIters; it++) {
                int t1 = rand() % (historySize - 1) + 1;
                int t2 = rand() % t1;

                int power = t1 - 1 - t2;

                float reward = 1.0f;

                for (int p = 0; p < power; p++)
                    reward *= discount;

                // Learn under goal
                #pragma omp parallel for
                for (int i = 0; i < numHiddenColumns; i++)
                    learn(Int2(i / hiddenSize.y, i % hiddenSize.y), t1, t2, reward);
            }
        }
    }

    // Forward kernel
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), goalCIs, inputCIs);
}

int Decoder::size() const {
    int size = sizeof(Int3) + 2 * sizeof(float) + sizeof(int) + hiddenCIs.size() * sizeof(int);

    size += sizeof(VisibleLayerDesc) + visibleLayer.weights.size() * sizeof(float);

    size += 3 * sizeof(int) + history.size() * (2 * history[0].inputCIs.size() * sizeof(int) + history[0].hiddenTargetCIs.size() * sizeof(int));

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
    writer.write(reinterpret_cast<const void*>(&discount), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&historyIters), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    
    writer.write(reinterpret_cast<const void*>(&visibleLayerDesc), sizeof(VisibleLayerDesc));

    writer.write(reinterpret_cast<const void*>(&visibleLayer.weights[0]), visibleLayer.weights.size() * sizeof(float));

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
    reader.read(reinterpret_cast<void*>(&discount), sizeof(float));
    reader.read(reinterpret_cast<void*>(&historyIters), sizeof(int));

    hiddenCIs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    reader.read(reinterpret_cast<void*>(&visibleLayerDesc), sizeof(VisibleLayerDesc));

    int diam = visibleLayerDesc.radius * 2 + 1;
    int area = diam * diam;

    visibleLayer.weights.resize(numHiddenCells * area * visibleLayerDesc.size.z * visibleLayerDesc.size.z);

    reader.read(reinterpret_cast<void*>(&visibleLayer.weights[0]), visibleLayer.weights.size() * sizeof(float));

    reader.read(reinterpret_cast<void*>(&historySize), sizeof(int));

    int numHistory;

    reader.read(reinterpret_cast<void*>(&numHistory), sizeof(int));

    int historyStart;

    reader.read(reinterpret_cast<void*>(&historyStart), sizeof(int));

    history.resize(numHistory);
    history.start = historyStart;

    int numVisibleColumns = visibleLayerDesc.size.x * visibleLayerDesc.size.y;

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
        writer.write(reinterpret_cast<const void*>(&history[t].actualCIs[0]), history[t].actualCIs.size() * sizeof(int));
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
        reader.read(reinterpret_cast<void*>(&history[t].actualCIs[0]), history[t].actualCIs.size() * sizeof(int));
        reader.read(reinterpret_cast<void*>(&history[t].inputCIs[0]), history[t].inputCIs.size() * sizeof(int));
        reader.read(reinterpret_cast<void*>(&history[t].hiddenTargetCIs[0]), history[t].hiddenTargetCIs.size() * sizeof(int));
    }
}
