// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "RLAdapter.h"
#include <iostream>

using namespace aon;

void RLAdapter::forward(
    const Int2 &columnPos,
    const IntBuffer* hiddenCIs,
    float reward,
    bool learnEnabled
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    float valuePrev = hiddenValues[address3(Int3(columnPos.x, columnPos.y, (*hiddenCIs)[hiddenColumnIndex]), hiddenSize)];

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    int diam = radius * 2 + 1;

    // Lower corner
    Int2 fieldLowerBound(columnPos.x - radius, columnPos.y - radius);

    // Bounds of receptive field, clamped to input size
    Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
    Int2 iterUpperBound(min(hiddenSize.x - 1, columnPos.x + radius), min(hiddenSize.y - 1, columnPos.y + radius));

    int count = (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1);

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        float sum = 0.0f;

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int otherHiddenColumnIndex = address2(Int2(ix, iy), Int2(hiddenSize.x, hiddenSize.y));

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int hidCI = (*hiddenCIs)[hiddenColumnIndex];

                sum += weights[hidCI + hiddenSize.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex))];
            }

        sum /= max(1, count);

        hiddenValues[hiddenCellIndex] = sum;

        if (sum > maxActivation || maxIndex == -1) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    goalCIs[hiddenColumnIndex] = maxIndex;

    float delta = lr * (reward + discount * maxActivation - valuePrev);
    std::cout << maxActivation << std::endl;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int otherHiddenColumnIndex = address2(Int2(ix, iy), Int2(hiddenSize.x, hiddenSize.y));

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int hidCIPrev = hiddenCIsPrev[hiddenColumnIndex];
                
                int wiStart = hiddenSize.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                for (int ohc = 0; ohc < hiddenSize.z; ohc++) {
                    int wi = ohc + wiStart;

                    traces[wi] = max(traces[wi] * traceDecay, static_cast<float>(hc == (*hiddenCIs)[hiddenColumnIndex] && ohc == hidCIPrev));

                    if (learnEnabled)
                        weights[wi] += delta * traces[wi];
                }
            }
    }
}

void RLAdapter::initRandom(
    const Int3 &hiddenSize,
    int radius
) {
    this->hiddenSize = hiddenSize;
    this->radius = radius;

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    int diam = radius * 2 + 1;
    int area = diam * diam;

    weights.resize(numHiddenCells * area * hiddenSize.z);

    for (int i = 0; i < weights.size(); i++)
        weights[i] = randf(-0.001f, 0.001f);

    traces = FloatBuffer(weights.size(), 0.0f);

    goalCIs = IntBuffer(numHiddenColumns, 0.0f);
    hiddenCIsPrev = IntBuffer(numHiddenColumns, 0.0f);

    hiddenValues = FloatBuffer(numHiddenCells, 0.0f);
}

void RLAdapter::step(
    const IntBuffer* hiddenCIs,
    float reward, 
    bool learnEnabled
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), hiddenCIs, reward, learnEnabled);

    hiddenCIsPrev = *hiddenCIs;
}

int RLAdapter::size() const {
    int size = sizeof(Int3) + sizeof(int) + 3 * sizeof(float) + 2 * goalCIs.size() * sizeof(int) + hiddenValues.size() * sizeof(float);

    size += 2 * weights.size() * sizeof(float);

    return size;
}

int RLAdapter::stateSize() const {
    int size = hiddenCIsPrev.size() * sizeof(int);

    size += traces.size() * sizeof(float);

    return size;
}

void RLAdapter::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));
    writer.write(reinterpret_cast<const void*>(&radius), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&discount), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&traceDecay), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&goalCIs[0]), goalCIs.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hiddenCIsPrev[0]), hiddenCIsPrev.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hiddenValues[0]), hiddenValues.size() * sizeof(float));

    writer.write(reinterpret_cast<const void*>(&weights[0]), weights.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&traces[0]), traces.size() * sizeof(float));
}

void RLAdapter::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));
    reader.read(reinterpret_cast<void*>(&radius), sizeof(int));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&lr), sizeof(float));
    reader.read(reinterpret_cast<void*>(&discount), sizeof(float));
    reader.read(reinterpret_cast<void*>(&traceDecay), sizeof(float));

    goalCIs.resize(numHiddenColumns);
    hiddenCIsPrev.resize(numHiddenColumns);
    hiddenValues.resize(numHiddenCells);

    reader.read(reinterpret_cast<void*>(&goalCIs[0]), goalCIs.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hiddenCIsPrev[0]), hiddenCIsPrev.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hiddenValues[0]), hiddenValues.size() * sizeof(float));

    int diam = radius * 2 + 1;
    int area = diam * diam;

    weights.resize(numHiddenCells * area * hiddenSize.z);
    traces.resize(weights.size());

    reader.read(reinterpret_cast<void*>(&weights[0]), weights.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&traces[0]), traces.size() * sizeof(float));
}

void RLAdapter::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&goalCIs[0]), goalCIs.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hiddenCIsPrev[0]), hiddenCIsPrev.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hiddenValues[0]), hiddenValues.size() * sizeof(float));

    writer.write(reinterpret_cast<const void*>(&traces[0]), traces.size() * sizeof(float));
}

void RLAdapter::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&goalCIs[0]), goalCIs.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hiddenCIsPrev[0]), hiddenCIsPrev.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hiddenValues[0]), hiddenValues.size() * sizeof(float));

    reader.read(reinterpret_cast<void*>(&traces[0]), traces.size() * sizeof(float));
}
