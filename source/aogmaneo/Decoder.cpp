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

    int numCellsPerColumn = hiddenSize.z * numDendrites;
    int hiddenCellsStart = hiddenColumnIndex * numCellsPerColumn;

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < numCellsPerColumn; hc++) {
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

                int progCI = (*progCIs)[visibleColumnIndex];
                int inCI = (*inputCIs)[visibleColumnIndex];

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                sum += vl.weightsInfer[progCI + wiStart] + vl.weightsInferPrev[inCI + wiStart];
            }

        if (sum > maxActivation || maxIndex == -1) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    hiddenCIs[hiddenColumnIndex] = maxIndex / numDendrites;
}

void Decoder::learn(
    const Int2 &columnPos,
    const IntBuffer* hiddenTargetCIsPrev,
    const IntBuffer* inputCIs,
    const IntBuffer* inputCIsPrev,
    float strength
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int numCellsPerColumn = hiddenSize.z * numDendrites;
    int hiddenCellsStart = hiddenColumnIndex * numCellsPerColumn;

    int targetCI = (*hiddenTargetCIsPrev)[hiddenColumnIndex];

    int maxDendriteIndex = -1;
    float maxDendriteActivation = -999999.0f;

    for (int di = 0; di < numDendrites; di++) {
        int hiddenCellIndex = (targetCI * numDendrites + di) + hiddenCellsStart;

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

                int inCI = (*inputCIs)[visibleColumnIndex];
                int inCIPrev = (*inputCIsPrev)[visibleColumnIndex];

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                sum += vl.weightsInfer[inCI + wiStart] + vl.weightsInferPrev[inCIPrev + wiStart];
            }

        if (sum > maxDendriteActivation || maxDendriteIndex == -1) {
            maxDendriteActivation = sum;
            maxDendriteIndex = di;
        }
    }

    for (int di = 0; di < numDendrites; di++) {
        int hiddenCellIndex = (targetCI * numDendrites + di) + hiddenCellsStart;

        float rate = (di == maxDendriteIndex ? lr : boost) * strength;

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

                int inCI = (*inputCIs)[visibleColumnIndex];
                int inCIPrev = (*inputCIsPrev)[visibleColumnIndex];

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                for (int vc = 0; vc < vld.size.z; vc++) {
                    int wi = vc + wiStart;

                    vl.weightsLearn[wi] += rate * ((vc == inCI) - vl.weightsLearn[wi]);
                    vl.weightsLearnPrev[wi] += rate * ((vc == inCIPrev) - vl.weightsLearnPrev[wi]);

                    vl.weightsInfer[wi] = logf(max(0.0001f, vl.weightsLearn[wi]));
                    vl.weightsInferPrev[wi] = logf(max(0.0001f, vl.weightsLearnPrev[wi]));
                }
            }
    }
}

void Decoder::initRandom(
    const Int3 &hiddenSize,
    int numDendrites,
    int historyCapacity,
    const VisibleLayerDesc &vld
) {
    this->vld = vld; 

    this->hiddenSize = hiddenSize;
    this->numDendrites = numDendrites;

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numCellsPerColumn = hiddenSize.z * numDendrites;
    int numHiddenCells = numHiddenColumns * numCellsPerColumn;
    
    int numVisibleColumns = vld.size.x * vld.size.y;

    int diam = vld.radius * 2 + 1;
    int area = diam * diam;

    vl.weightsInfer.resize(numHiddenCells * area * vld.size.z);
    vl.weightsInferPrev.resize(vl.weightsInfer.size());
    vl.weightsLearn.resize(vl.weightsInfer.size());
    vl.weightsLearnPrev.resize(vl.weightsInfer.size());

    for (int i = 0; i < vl.weightsInfer.size(); i++) {
        vl.weightsLearn[i] = randf(0.0f, 0.01f);
        vl.weightsLearnPrev[i] = randf(0.0f, 0.01f);

        vl.weightsInfer[i] = logf(max(0.0001f, vl.weightsLearn[i]));
        vl.weightsInferPrev[i] = logf(max(0.0001f, vl.weightsLearnPrev[i]));
    }

    hiddenCIs = IntBuffer(numHiddenColumns, 0);

    // Create (pre-allocated) history samples
    historySize = 0;
    history.resize(historyCapacity);

    for (int i = 0; i < history.size(); i++) {
        history[i].inputCIs.resize(numVisibleColumns, 0);
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
        HistorySample &s = history[0];

        for (int t = 1; t < historySize; t++) {
            HistorySample &sPrev = history[t];
            HistorySample &sPrevNext = history[t - 1];

            float strength = powf(discount, historySize - 1 - t);

            // Learn kernel
            #pragma omp parallel for
            for (int i = 0; i < numHiddenColumns; i++)
                learn(Int2(i / hiddenSize.y, i % hiddenSize.y), &sPrevNext.hiddenTargetCIsPrev, &s.inputCIs, &sPrev.inputCIs, strength);
        }
    }
}

int Decoder::size() const {
    int size = sizeof(Int3) + sizeof(int) + 3 * sizeof(float) + hiddenCIs.size() * sizeof(int) + sizeof(int);

    size += sizeof(VisibleLayerDesc) + 4 * vl.weightsInfer.size() * sizeof(float);

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
    writer.write(reinterpret_cast<const void*>(&numDendrites), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&boost), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&discount), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    
    writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

    writer.write(reinterpret_cast<const void*>(&vl.weightsInfer[0]), vl.weightsInfer.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&vl.weightsInferPrev[0]), vl.weightsInferPrev.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&vl.weightsLearn[0]), vl.weightsLearn.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&vl.weightsLearnPrev[0]), vl.weightsLearnPrev.size() * sizeof(float));

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
    reader.read(reinterpret_cast<void*>(&numDendrites), sizeof(int));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numCellsPerColumn = hiddenSize.z * numDendrites;
    int numHiddenCells = numHiddenColumns * numCellsPerColumn;

    reader.read(reinterpret_cast<void*>(&lr), sizeof(float));
    reader.read(reinterpret_cast<void*>(&boost), sizeof(float));
    reader.read(reinterpret_cast<void*>(&discount), sizeof(float));

    hiddenCIs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    reader.read(reinterpret_cast<void*>(&vld), sizeof(VisibleLayerDesc));

    int numVisibleColumns = vld.size.x * vld.size.y;

    int diam = vld.radius * 2 + 1;
    int area = diam * diam;

    vl.weightsInfer.resize(numHiddenCells * area * vld.size.z);
    vl.weightsInferPrev.resize(vl.weightsInfer.size());
    vl.weightsLearn.resize(vl.weightsInfer.size());
    vl.weightsLearnPrev.resize(vl.weightsInfer.size());

    reader.read(reinterpret_cast<void*>(&vl.weightsInfer[0]), vl.weightsInfer.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&vl.weightsInferPrev[0]), vl.weightsInferPrev.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&vl.weightsLearn[0]), vl.weightsLearn.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&vl.weightsLearnPrev[0]), vl.weightsLearnPrev.size() * sizeof(float));

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
