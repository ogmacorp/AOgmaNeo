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
    const IntBuffer* progCIs,
    const IntBuffer* inputCIs
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
                int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int progCI = (*progCIs)[visibleColumnIndex];
                int inCI = (*inputCIs)[visibleColumnIndex];

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int wi = progCI + vld.size.z * (inCI + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex)));

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
    int t2,
    float minQ
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    int targetCI = history[t1 - 1].hiddenTargetCIsPrev[hiddenColumnIndex];

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

    float maxActivation = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        float sum = 0.0f;

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int progCI = history[t1 - 1].inputCIs[visibleColumnIndex];
                int inCI = history[t1].inputCIs[visibleColumnIndex];

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int wi = progCI + vld.size.z * (inCI + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex)));

                sum += vl.weights[wi];
            }

        sum /= count;

        maxActivation = max(maxActivation, sum);
    }

    int hiddenCellIndexTarget = targetCI + hiddenCellsStart;

    float sum = 0.0f;

    for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
        for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
            int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

            int progCI = history[t1 - 1].inputCIs[visibleColumnIndex];
            int inCI = history[t1].inputCIs[visibleColumnIndex];

            Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

            int wi = progCI + vld.size.z * (inCI + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndexTarget)));

            sum += vl.weights[wi];
        }

    sum /= count;

    float delta = lr * (max(minQ, discount * maxActivation) - sum);

    for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
        for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
            int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

            int progCI = history[t1 - 1].inputCIs[visibleColumnIndex];
            int inCI = history[t1].inputCIs[visibleColumnIndex];

            Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

            int wi = progCI + vld.size.z * (inCI + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndexTarget)));

            vl.weights[wi] += delta;
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
    
    int numVisibleColumns = vld.size.x * vld.size.y;

    int diam = vld.radius * 2 + 1;
    int area = diam * diam;

    vl.weights.resize(numHiddenCells * area * vld.size.z * vld.size.z);

    for (int i = 0; i < vl.weights.size(); i++)
        vl.weights[i] = randf(0.0f, 0.01f);

    hiddenCIs = IntBuffer(numHiddenColumns, 0);

    // Create (pre-allocated) history samples
    historySize = 0;
    history.resize(historyCapacity);

    for (int i = 0; i < history.size(); i++) {
        history[i].inputCIs = IntBuffer(numVisibleColumns, 0);
        history[i].hiddenTargetCIsPrev = IntBuffer(numHiddenColumns, 0);
    }
}

void Decoder::activate(
    const IntBuffer* progCIs,
    const IntBuffer* inputCIs
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    // Forward kernel
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), progCIs, inputCIs);
}

void Decoder::learn(
    const IntBuffer* hiddenTargetCIsPrev,
    const IntBuffer* inputCIs
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    
    history.pushFront();

    // If not at cap, increment
    if (historySize < history.size())
        historySize++;
    
    // Add new sample
    {
        HistorySample &s = history[0];

        // Copy
        s.inputCIs = *inputCIs;
        s.hiddenTargetCIsPrev = *hiddenTargetCIsPrev;
    }

    if (historySize == history.size()) {
        int t1 = historySize - 1;

        float strength = 1.0f;

        for (int t2 = 0; t2 < historySize - 1; t2++) {
            float minQ = powf(discount, t1 - 1 - t2);

            // Learn kernel
            #pragma omp parallel for
            for (int i = 0; i < numHiddenColumns; i++)
                learn(Int2(i / hiddenSize.y, i % hiddenSize.y), t1, t2, minQ);
        }
    }
}

int Decoder::size() const {
    int size = sizeof(Int3) + 2 * sizeof(float) + hiddenCIs.size() * sizeof(int);

    size += sizeof(VisibleLayerDesc) + vl.weights.size() * sizeof(float);

    size += 3 * sizeof(int);

    const HistorySample &s = history[0];

    int sampleSize = s.inputCIs.size() * sizeof(int) + s.hiddenTargetCIsPrev.size() * sizeof(int);

    size += history.size() * sampleSize;

    return size;
}

int Decoder::stateSize() const {
    int size = hiddenCIs.size() * sizeof(int);

    const HistorySample &s = history[0];

    int sampleSize = s.inputCIs.size() * sizeof(int) + s.hiddenTargetCIsPrev.size() * sizeof(int);

    size += history.size() * sampleSize;

    return size;
}

void Decoder::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&discount), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    
    writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

    writer.write(reinterpret_cast<const void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));

    writer.write(reinterpret_cast<const void*>(&historySize), sizeof(int));

    int numHistorySamples = history.size();

    writer.write(reinterpret_cast<const void*>(&numHistorySamples), sizeof(int));

    int historyStart = history.start;

    writer.write(reinterpret_cast<const void*>(&historyStart), sizeof(int));

    for (int t = 0; t < history.size(); t++) {
        const HistorySample &s = history[t];

        writer.write(reinterpret_cast<const void*>(&s.inputCIs[0]), s.inputCIs.size() * sizeof(int));
        writer.write(reinterpret_cast<const void*>(&s.hiddenTargetCIsPrev[0]), s.hiddenTargetCIsPrev.size() * sizeof(int));
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

    hiddenCIs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    reader.read(reinterpret_cast<void*>(&vld), sizeof(VisibleLayerDesc));

    int numVisibleColumns = vld.size.x * vld.size.y;

    int diam = vld.radius * 2 + 1;
    int area = diam * diam;

    vl.weights.resize(numHiddenCells * area * vld.size.z * vld.size.z);

    reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));

    reader.read(reinterpret_cast<void*>(&historySize), sizeof(int));

    int numHistorySamples;

    reader.read(reinterpret_cast<void*>(&numHistorySamples), sizeof(int));

    int historyStart;

    reader.read(reinterpret_cast<void*>(&historyStart), sizeof(int));

    history.resize(numHistorySamples);
    history.start = historyStart;

    for (int t = 0; t < history.size(); t++) {
        HistorySample &s = history[t];

        s.inputCIs.resize(numVisibleColumns);

        reader.read(reinterpret_cast<void*>(&s.inputCIs[0]), s.inputCIs.size() * sizeof(int));

        s.hiddenTargetCIsPrev.resize(numHiddenColumns);

        reader.read(reinterpret_cast<void*>(&s.hiddenTargetCIsPrev[0]), s.hiddenTargetCIsPrev.size() * sizeof(int));
    }
}

void Decoder::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    int historyStart = history.start;

    writer.write(reinterpret_cast<const void*>(&historyStart), sizeof(int));

    for (int t = 0; t < history.size(); t++) {
        const HistorySample &s = history[t];

        writer.write(reinterpret_cast<const void*>(&s.inputCIs[0]), s.inputCIs.size() * sizeof(int));
        writer.write(reinterpret_cast<const void*>(&s.hiddenTargetCIsPrev[0]), s.hiddenTargetCIsPrev.size() * sizeof(int));
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
        HistorySample &s = history[t];

        reader.read(reinterpret_cast<void*>(&s.inputCIs[0]), s.inputCIs.size() * sizeof(int));
        reader.read(reinterpret_cast<void*>(&s.hiddenTargetCIsPrev[0]), s.hiddenTargetCIsPrev.size() * sizeof(int));
    }
}
