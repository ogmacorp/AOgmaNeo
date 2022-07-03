// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "StateAdapter.h"

using namespace aon;

void StateAdapter::forward(
    const Int2 &columnPos,
    const IntBuffer* goalCIs,
    const IntBuffer* hiddenCIs
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    int diam = radius * 2 + 1;

    // Lower corner
    Int2 fieldLowerBound(columnPos.x - radius, columnPos.y - radius);

    // Bounds of receptive field, clamped to input size
    Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
    Int2 iterUpperBound(min(hiddenSize.x - 1, columnPos.x + radius), min(hiddenSize.y - 1, columnPos.y + radius));

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        float sum = 0.0f;

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int otherHiddenColumnIndex = address2(Int2(ix, iy), Int2(hiddenSize.x, hiddenSize.y));

                int inCI = (*goalCIs)[otherHiddenColumnIndex];
                int inCIPrev = (*hiddenCIs)[otherHiddenColumnIndex];

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int wi = inCI + hiddenSize.z * (inCIPrev + hiddenSize.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex)));

                sum += weights[wi];
            }

        if (sum > maxActivation || maxIndex == -1) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    this->goalCIs[hiddenColumnIndex] = maxIndex;
}

void StateAdapter::learn(
    const Int2 &columnPos,
    int t1,
    int t2,
    float reward
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    int diam = radius * 2 + 1;

    // Lower corner
    Int2 fieldLowerBound(columnPos.x - radius, columnPos.y - radius);

    // Bounds of receptive field, clamped to input size
    Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
    Int2 iterUpperBound(min(hiddenSize.x - 1, columnPos.x + radius), min(hiddenSize.y - 1, columnPos.y + radius));

    int count = (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1);

    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        float sum = 0.0f;

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int otherHiddenColumnIndex = address2(Int2(ix, iy), Int2(hiddenSize.x, hiddenSize.y));

                int inCI = history[t2].hiddenCIs[otherHiddenColumnIndex];
                int inCIPrev = history[t1 - 1].hiddenCIs[otherHiddenColumnIndex];

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int wi = inCI + hiddenSize.z * (inCIPrev + hiddenSize.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex)));

                sum += weights[wi];
            }

        sum /= count;

        maxActivation = max(maxActivation, sum);
    }

    int targetCI = history[t1 - 1].hiddenCIs[hiddenColumnIndex];

    int hiddenCellIndexTarget = targetCI + hiddenCellsStart;

    float sumPrev = 0.0f;

    for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
        for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
            int otherHiddenColumnIndex = address2(Int2(ix, iy), Int2(hiddenSize.x, hiddenSize.y));

            int inCI = history[t2].hiddenCIs[otherHiddenColumnIndex];
            int inCIPrev = history[t1].hiddenCIs[otherHiddenColumnIndex];

            Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

            int wi = inCI + hiddenSize.z * (inCIPrev + hiddenSize.z * (offset.y + diam * (offset.x + diam * hiddenCellIndexTarget)));

            sumPrev += weights[wi];
        }

    sumPrev /= count;

    float delta = lr * (max(reward, discount * maxActivation) - sumPrev);

    for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
        for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
            int otherHiddenColumnIndex = address2(Int2(ix, iy), Int2(hiddenSize.x, hiddenSize.y));

            int inCI = history[t2].hiddenCIs[otherHiddenColumnIndex];
            int inCIPrev = history[t1].hiddenCIs[otherHiddenColumnIndex];

            Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

            int wi = inCI + hiddenSize.z * (inCIPrev + hiddenSize.z * (offset.y + diam * (offset.x + diam * hiddenCellIndexTarget)));

            weights[wi] += delta;
        }
}

void StateAdapter::initRandom(
    const Int3 &hiddenSize,
    int radius,
    int historyCapacity
) {
    this->hiddenSize = hiddenSize;
    this->radius = radius;

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    int diam = radius * 2 + 1;
    int area = diam * diam;

    weights.resize(numHiddenCells * area * hiddenSize.z * hiddenSize.z);

    for (int i = 0; i < weights.size(); i++)
        weights[i] = randf(-0.01f, 0.0f);

    goalCIs = IntBuffer(numHiddenColumns, 0);

    historySize = 0;
    history.resize(historyCapacity);

    for (int i = 0; i < history.size(); i++)
        history[i].hiddenCIs = IntBuffer(numHiddenColumns, 0);
}

void StateAdapter::step(
    const IntBuffer* goalCIs,
    const IntBuffer* hiddenCIs,
    bool learnEnabled
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    history.pushFront();

    // If not at cap, increment
    if (historySize < history.size())
        historySize++;

    history[0].hiddenCIs = *hiddenCIs;

    if (learnEnabled && historySize > 1) {
        for (int it = 0; it < historyIters; it++) {
            int t1 = rand() % (historySize - 1) + 1;
            int t2 = rand() % t1;

            int power = t1 - 1 - t2;

            float reward = 1.0f;

            for (int p = 0; p < power; p++)
                reward *= discount;

            // Learn kernel
            #pragma omp parallel for
            for (int i = 0; i < numHiddenColumns; i++)
                learn(Int2(i / hiddenSize.y, i % hiddenSize.y), t1, t2, reward);
        }
    }
    
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), goalCIs, hiddenCIs);
}

int StateAdapter::size() const {
    int size = sizeof(Int3) + sizeof(int) + 2 * sizeof(float) + sizeof(int) + goalCIs.size() * sizeof(int);

    size += weights.size() * sizeof(float);

    size += 3 * sizeof(int) + history.size() * (history[0].hiddenCIs.size() * sizeof(int));

    return size;
}

int StateAdapter::stateSize() const {
    int size = goalCIs.size() * sizeof(int);

    size += sizeof(int) + history.size() * (history[0].hiddenCIs.size() * sizeof(int));

    return size;
}

void StateAdapter::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));
    writer.write(reinterpret_cast<const void*>(&radius), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&discount), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&historyIters), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&goalCIs[0]), goalCIs.size() * sizeof(int));

    writer.write(reinterpret_cast<const void*>(&weights[0]), weights.size() * sizeof(float));

    writer.write(reinterpret_cast<const void*>(&historySize), sizeof(int));

    int numHistory = history.size();

    writer.write(reinterpret_cast<const void*>(&numHistory), sizeof(int));

    int historyStart = history.start;

    writer.write(reinterpret_cast<const void*>(&historyStart), sizeof(int));

    for (int t = 0; t < history.size(); t++)
        writer.write(reinterpret_cast<const void*>(&history[t].hiddenCIs[0]), history[t].hiddenCIs.size() * sizeof(int));
}

void StateAdapter::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));
    reader.read(reinterpret_cast<void*>(&radius), sizeof(int));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&lr), sizeof(float));
    reader.read(reinterpret_cast<void*>(&discount), sizeof(float));
    reader.read(reinterpret_cast<void*>(&historyIters), sizeof(int));

    goalCIs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&goalCIs[0]), goalCIs.size() * sizeof(int));

    int diam = radius * 2 + 1;
    int area = diam * diam;

    weights.resize(numHiddenCells * area * hiddenSize.z * hiddenSize.z);

    reader.read(reinterpret_cast<void*>(&weights[0]), weights.size() * sizeof(float));

    reader.read(reinterpret_cast<void*>(&historySize), sizeof(int));

    int numHistory;

    reader.read(reinterpret_cast<void*>(&numHistory), sizeof(int));

    int historyStart;

    reader.read(reinterpret_cast<void*>(&historyStart), sizeof(int));

    history.resize(numHistory);
    history.start = historyStart;

    for (int t = 0; t < history.size(); t++) {
        history[t].hiddenCIs.resize(numHiddenColumns);

        reader.read(reinterpret_cast<void*>(&history[t].hiddenCIs[0]), history[t].hiddenCIs.size() * sizeof(int));
    }
}

void StateAdapter::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&goalCIs[0]), goalCIs.size() * sizeof(int));

    int historyStart = history.start;

    writer.write(reinterpret_cast<const void*>(&historyStart), sizeof(int));

    for (int t = 0; t < history.size(); t++)
        writer.write(reinterpret_cast<const void*>(&history[t].hiddenCIs[0]), history[t].hiddenCIs.size() * sizeof(int));
}

void StateAdapter::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&goalCIs[0]), goalCIs.size() * sizeof(int));

    int historyStart;

    reader.read(reinterpret_cast<void*>(&historyStart), sizeof(int));

    history.start = historyStart;

    for (int t = 0; t < history.size(); t++)
        reader.read(reinterpret_cast<void*>(&history[t].hiddenCIs[0]), history[t].hiddenCIs.size() * sizeof(int));
}
