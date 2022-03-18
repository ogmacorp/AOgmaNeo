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
    const IntBuffer* inputCIs,
    const IntBuffer* feedBackCIs
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    // Pre-count
    int diam = visibleLayerDesc.radius * 2 + 1;

    // Projection
    Float2 hToV = Float2(static_cast<float>(visibleLayerDesc.size.x) / static_cast<float>(hiddenSize.x),
        static_cast<float>(visibleLayerDesc.size.y) / static_cast<float>(hiddenSize.y));

    Int2 visibleCenter = project(columnPos, hToV);

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
                int visibleColumnIndex = address2(Int2(ix, iy), Int2(visibleLayerDesc.size.x, visibleLayerDesc.size.y));

                int goalCI = (*goalCIs)[visibleColumnIndex];
                int inCI = (*inputCIs)[visibleColumnIndex];

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int wiStart = visibleLayerDesc.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                sum += visibleLayer.iWeights[goalCI + visibleLayerDesc.gSizeZ * (inCI + wiStart)];

                if (hasFeedBack) {
                    int feedBackCI = (*feedBackCIs)[visibleColumnIndex];

                    sum += visibleLayer.fbWeights[goalCI + visibleLayerDesc.gSizeZ * (feedBackCI + wiStart)];
                }
            }

        if (sum > maxActivation || maxIndex == -1) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    hiddenCIs[hiddenColumnIndex] = maxIndex;
}

void Decoder::backward(
    const Int2 &columnPos,
    int t1,
    int t2
) {
    int diam = visibleLayerDesc.radius * 2 + 1;

    int visibleColumnIndex = address2(columnPos, Int2(visibleLayerDesc.size.x, visibleLayerDesc.size.y));

    int visibleCellsStart = visibleColumnIndex * visibleLayerDesc.size.z;

    int actualCI = history[t2].actualCIs[visibleColumnIndex];
    int inCI = history[t1].inputCIs[visibleColumnIndex];
    int feedBackCI = history[t1 - 1].inputCIs[visibleColumnIndex];

    // Projection
    Float2 vToH = Float2(static_cast<float>(hiddenSize.x) / static_cast<float>(visibleLayerDesc.size.x),
        static_cast<float>(hiddenSize.y) / static_cast<float>(visibleLayerDesc.size.y));

    Float2 hToV = Float2(static_cast<float>(visibleLayerDesc.size.x) / static_cast<float>(hiddenSize.x),
        static_cast<float>(visibleLayerDesc.size.y) / static_cast<float>(hiddenSize.y));

    Int2 reverseRadii(ceilf(vToH.x * (visibleLayerDesc.radius * 2 + 1) * 0.5f), ceilf(vToH.y * (visibleLayerDesc.radius * 2 + 1) * 0.5f));

    Int2 hiddenCenter = project(columnPos, vToH);

    // Lower corner
    Int2 fieldLowerBound(hiddenCenter.x - reverseRadii.x, hiddenCenter.y - reverseRadii.y);

    // Bounds of receptive field, clamped to input size
    Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
    Int2 iterUpperBound(min(hiddenSize.x - 1, hiddenCenter.x + reverseRadii.x), min(hiddenSize.y - 1, hiddenCenter.y + reverseRadii.y));

    float im = 1.0f;
    float fbm = 1.0f;

    for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
        for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
            Int2 hiddenPos = Int2(ix, iy);

            int hiddenColumnIndex = address2(hiddenPos, Int2(hiddenSize.x, hiddenSize.y));

            Int2 visibleCenter = project(hiddenPos, hToV);

            if (inBounds(columnPos, Int2(visibleCenter.x - visibleLayerDesc.radius, visibleCenter.y - visibleLayerDesc.radius), Int2(visibleCenter.x + visibleLayerDesc.radius + 1, visibleCenter.y + visibleLayerDesc.radius + 1))) {
                Int2 offset(columnPos.x - visibleCenter.x + visibleLayerDesc.radius, columnPos.y - visibleCenter.y + visibleLayerDesc.radius);

                int wiStart = visibleLayerDesc.size.z * (offset.y + diam * (offset.x + diam * hiddenColumnIndex));

                im = min(im, visibleLayer.iRates[actualCI + visibleLayerDesc.gSizeZ * (inCI + wiStart)]);

                if (hasFeedBack)
                    fbm = min(fbm, visibleLayer.fbRates[actualCI + visibleLayerDesc.gSizeZ * (feedBackCI + wiStart)]);
            }
        }

    visibleLayer.iGates[visibleColumnIndex] = im;
    visibleLayer.fbGates[visibleColumnIndex] = fbm;
}

void Decoder::learn(
    const Int2 &columnPos,
    int t1,
    int t2,
    float modulation
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

    // Lower corner
    Int2 fieldLowerBound(visibleCenter.x - visibleLayerDesc.radius, visibleCenter.y - visibleLayerDesc.radius);

    // Bounds of receptive field, clamped to input size
    Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
    Int2 iterUpperBound(min(visibleLayerDesc.size.x - 1, visibleCenter.x + visibleLayerDesc.radius), min(visibleLayerDesc.size.y - 1, visibleCenter.y + visibleLayerDesc.radius));

    int count = (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1);

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        float sum = 0.0f;

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int visibleColumnIndex = address2(Int2(ix, iy), Int2(visibleLayerDesc.size.x, visibleLayerDesc.size.y));

                int actualCI = history[t2].actualCIs[visibleColumnIndex];

                int inCI = history[t1].inputCIs[visibleColumnIndex];
                int feedBackCI = history[t1 - 1].inputCIs[visibleColumnIndex];

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int wiStart = visibleLayerDesc.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                sum += visibleLayer.iWeights[actualCI + visibleLayerDesc.gSizeZ * (inCI + wiStart)];

                if (hasFeedBack)
                    sum += visibleLayer.fbWeights[actualCI + visibleLayerDesc.gSizeZ * (feedBackCI + wiStart)];
            }

        sum /= count;

        hiddenActivations[hiddenCellIndex] = sum;

        if (sum > maxActivation || maxIndex == -1) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    if (maxIndex != targetCI) {
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

            float delta = lr * modulation * ((hc == targetCI) - hiddenActivations[hiddenCellIndex]);

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(visibleLayerDesc.size.x, visibleLayerDesc.size.y));

                    int actualCI = history[t2].actualCIs[visibleColumnIndex];

                    int inCI = history[t1].inputCIs[visibleColumnIndex];
                    int feedBackCI = history[t1 - 1].inputCIs[visibleColumnIndex];

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    int wiStart = visibleLayerDesc.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                    visibleLayer.iWeights[actualCI + visibleLayerDesc.gSizeZ * (inCI + wiStart)] += delta * visibleLayer.iGates[visibleColumnIndex];

                    if (hasFeedBack)
                        visibleLayer.fbWeights[actualCI + visibleLayerDesc.gSizeZ * (feedBackCI + wiStart)] += delta * visibleLayer.fbGates[visibleColumnIndex];
                }
        }
    }

    int hiddenCellIndexTarget = targetCI + hiddenCellsStart;

    float mult = (maxIndex == targetCI ? 1.0f - decay : 1.0f + decay);

    for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
        for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
            int visibleColumnIndex = address2(Int2(ix, iy), Int2(visibleLayerDesc.size.x, visibleLayerDesc.size.y));

            int actualCI = history[t2].actualCIs[visibleColumnIndex];

            int inCI = history[t1].inputCIs[visibleColumnIndex];
            int feedBackCI = history[t1 - 1].inputCIs[visibleColumnIndex];

            Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

            int wiRate = actualCI + visibleLayerDesc.gSizeZ * (inCI + visibleLayerDesc.size.z * (offset.y + diam * (offset.x + diam * hiddenColumnIndex)));

            visibleLayer.iRates[wiRate] = min(1.0f, visibleLayer.iRates[wiRate] * mult);

            if (hasFeedBack) {
                int fbwiRate = actualCI + visibleLayerDesc.gSizeZ * (feedBackCI + visibleLayerDesc.size.z * (offset.y + diam * (offset.x + diam * hiddenColumnIndex)));

                visibleLayer.fbRates[fbwiRate] = min(1.0f, visibleLayer.fbRates[fbwiRate] * mult);
            }
        }
}

void Decoder::initRandom(
    const Int3 &hiddenSize,
    int historyCapacity,
    const VisibleLayerDesc &visibleLayerDesc,
    bool hasFeedBack
) {
    this->visibleLayerDesc = visibleLayerDesc; 

    this->hiddenSize = hiddenSize;
    this->hasFeedBack = hasFeedBack;

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;
    
    int numVisibleColumns = visibleLayerDesc.size.x * visibleLayerDesc.size.y;

    // Create layers
    int diam = visibleLayerDesc.radius * 2 + 1;
    int area = diam * diam;

    visibleLayer.iWeights.resize(numHiddenCells * area * visibleLayerDesc.size.z * visibleLayerDesc.gSizeZ);
    visibleLayer.iRates = FloatBuffer(numHiddenColumns * area * visibleLayerDesc.size.z * visibleLayerDesc.gSizeZ, 1.0f);

    for (int i = 0; i < visibleLayer.iWeights.size(); i++)
        visibleLayer.iWeights[i] = randf(-0.01f, 0.01f);

    if (hasFeedBack) {
        visibleLayer.fbWeights.resize(visibleLayer.iWeights.size());
        visibleLayer.fbRates = FloatBuffer(visibleLayer.iRates.size(), 1.0f);

        for (int i = 0; i < visibleLayer.iWeights.size(); i++)
            visibleLayer.fbWeights[i] = randf(-0.01f, 0.01f);
    }
    else {
        visibleLayer.fbWeights.resize(0); // No feed back
        visibleLayer.fbRates.resize(0);
    }

    visibleLayer.iGates = FloatBuffer(numVisibleColumns, 1.0f);
    visibleLayer.fbGates = FloatBuffer(numVisibleColumns, 1.0f);

    hiddenActivations = FloatBuffer(numHiddenCells, 0.0f);
    hiddenCIs = IntBuffer(numHiddenColumns, 0);

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
    const IntBuffer* feedBackCIs,
    const IntBuffer* hiddenTargetCIs,
    bool learnEnabled,
    bool stateUpdate
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    int numVisibleColumns = visibleLayerDesc.size.x * visibleLayerDesc.size.y;

    if (stateUpdate) {
        history.pushFront();

        // If not at cap, increment
        if (historySize < history.size())
            historySize++;

        history[0].actualCIs = *actualCIs;
        history[0].inputCIs = *inputCIs;
        history[0].hiddenTargetCIs = *hiddenTargetCIs;

        if (learnEnabled && historySize > maxSteps + 1) {
            for (int it = 0; it < historyIters; it++) {
                int t1 = rand() % (historySize - 1 - maxSteps) + maxSteps + 1;

                int t2 = t1 - 1 - (rand() % maxSteps);

                int power = t1 - 1 - t2;

                float modulation = 1.0f;

                for (int p = 0; p < power; p++)
                    modulation *= discount;
                
                #pragma omp parallel for
                for (int i = 0; i < numVisibleColumns; i++)
                    backward(Int2(i / visibleLayerDesc.size.y, i % visibleLayerDesc.size.y), t1, t2);

                // Learn under goal
                #pragma omp parallel for
                for (int i = 0; i < numHiddenColumns; i++)
                    learn(Int2(i / hiddenSize.y, i % hiddenSize.y), t1, t2, modulation);
            }
        }
    }

    // Forward kernel
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), goalCIs, inputCIs, feedBackCIs);
}

int Decoder::size() const {
    int size = sizeof(Int3) + sizeof(Byte) + 3 * sizeof(float) + 2 * sizeof(int) + hiddenCIs.size() * sizeof(int);

    size += sizeof(VisibleLayerDesc) + (hasFeedBack ? 2 : 1) * (visibleLayer.iWeights.size() * sizeof(float) + visibleLayer.iRates.size() * sizeof(float));

    size += 3 * sizeof(int) + history.size() * (2 * history[0].inputCIs.size() * sizeof(int) + history[0].hiddenTargetCIs.size() * sizeof(int));

    return size;
}

int Decoder::stateSize() const {
    int size = hiddenCIs.size() * sizeof(int);

    size += sizeof(int) + history.size() * (2 * history[0].inputCIs.size() * sizeof(int) + history[0].hiddenTargetCIs.size() * sizeof(int));

    return size;
}

void Decoder::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));
    writer.write(reinterpret_cast<const void*>(&hasFeedBack), sizeof(Byte));

    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&decay), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&discount), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&historyIters), sizeof(int));
    writer.write(reinterpret_cast<const void*>(&maxSteps), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    
    writer.write(reinterpret_cast<const void*>(&visibleLayerDesc), sizeof(VisibleLayerDesc));

    writer.write(reinterpret_cast<const void*>(&visibleLayer.iWeights[0]), visibleLayer.iWeights.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&visibleLayer.iRates[0]), visibleLayer.iRates.size() * sizeof(float));

    if (hasFeedBack) {
        writer.write(reinterpret_cast<const void*>(&visibleLayer.fbWeights[0]), visibleLayer.fbWeights.size() * sizeof(float));
        writer.write(reinterpret_cast<const void*>(&visibleLayer.fbRates[0]), visibleLayer.fbRates.size() * sizeof(float));
    }

    writer.write(reinterpret_cast<const void*>(&historySize), sizeof(int));

    int numHistory = history.size();

    writer.write(reinterpret_cast<const void*>(&numHistory), sizeof(int));

    int historyStart = history.start;

    writer.write(reinterpret_cast<const void*>(&historyStart), sizeof(int));

    for (int t = 0; t < history.size(); t++) {
        writer.write(reinterpret_cast<const void*>(&history[t].actualCIs[0]), history[t].actualCIs.size() * sizeof(int));
        writer.write(reinterpret_cast<const void*>(&history[t].inputCIs[0]), history[t].inputCIs.size() * sizeof(int));
        writer.write(reinterpret_cast<const void*>(&history[t].hiddenTargetCIs[0]), history[t].hiddenTargetCIs.size() * sizeof(int));
    }
}

void Decoder::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));
    reader.read(reinterpret_cast<void*>(&hasFeedBack), sizeof(Byte));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&lr), sizeof(float));
    reader.read(reinterpret_cast<void*>(&decay), sizeof(float));
    reader.read(reinterpret_cast<void*>(&discount), sizeof(float));
    reader.read(reinterpret_cast<void*>(&historyIters), sizeof(int));
    reader.read(reinterpret_cast<void*>(&maxSteps), sizeof(int));

    hiddenActivations = FloatBuffer(numHiddenCells, 0.0f);

    hiddenCIs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    reader.read(reinterpret_cast<void*>(&visibleLayerDesc), sizeof(VisibleLayerDesc));

    int numVisibleColumns = visibleLayerDesc.size.x * visibleLayerDesc.size.y;

    int diam = visibleLayerDesc.radius * 2 + 1;
    int area = diam * diam;

    visibleLayer.iWeights.resize(numHiddenCells * area * visibleLayerDesc.size.z * visibleLayerDesc.gSizeZ);
    visibleLayer.iRates.resize(numHiddenColumns * area * visibleLayerDesc.size.z * visibleLayerDesc.gSizeZ);

    reader.read(reinterpret_cast<void*>(&visibleLayer.iWeights[0]), visibleLayer.iWeights.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&visibleLayer.iRates[0]), visibleLayer.iRates.size() * sizeof(float));

    if (hasFeedBack) {
        visibleLayer.fbWeights.resize(visibleLayer.iWeights.size());
        visibleLayer.fbRates.resize(visibleLayer.iRates.size());

        reader.read(reinterpret_cast<void*>(&visibleLayer.fbWeights[0]), visibleLayer.fbWeights.size() * sizeof(float));
        reader.read(reinterpret_cast<void*>(&visibleLayer.fbRates[0]), visibleLayer.fbRates.size() * sizeof(float));
    }
    else {
        visibleLayer.fbWeights.resize(0); // No feedback
        visibleLayer.fbRates.resize(0);
    }

    visibleLayer.iGates = FloatBuffer(numVisibleColumns, 1.0f);
    visibleLayer.fbGates = FloatBuffer(numVisibleColumns, 1.0f);

    reader.read(reinterpret_cast<void*>(&historySize), sizeof(int));

    int numHistory;

    reader.read(reinterpret_cast<void*>(&numHistory), sizeof(int));

    int historyStart;

    reader.read(reinterpret_cast<void*>(&historyStart), sizeof(int));

    history.resize(numHistory);
    history.start = historyStart;

    for (int t = 0; t < history.size(); t++) {
        history[t].actualCIs.resize(numVisibleColumns);
        history[t].inputCIs.resize(numVisibleColumns);
        history[t].hiddenTargetCIs.resize(numHiddenColumns);

        reader.read(reinterpret_cast<void*>(&history[t].actualCIs[0]), history[t].actualCIs.size() * sizeof(int));
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
