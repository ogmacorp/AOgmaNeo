// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "RLAdapter.h"

using namespace aon;

void RLAdapter::forward(
    const Int2 &columnPos,
    const IntBuffer* hiddenCIs,
    float reward,
    bool learnEnabled
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        float value = weights[hiddenCellIndex];

        if (value > maxActivation || maxIndex == -1) {
            maxActivation = value;
            maxIndex = hc;
        }
    }

    goalCIs[hiddenColumnIndex] = maxIndex;

    float qTarget = reward + discount * maxActivation;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        weights[hiddenCellIndex] += lr * (qTarget - weights[hiddenCellIndex]) * traces[hiddenCellIndex];

        traces[hiddenCellIndex] += (1.0f - traceDecay) * (static_cast<float>(hc == (*hiddenCIs)[hiddenColumnIndex]) - traces[hiddenCellIndex]);
    }
}

void RLAdapter::initRandom(
    const Int3 &hiddenSize
) {
    this->hiddenSize = hiddenSize;

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    weights.resize(numHiddenCells);

    for (int i = 0; i < weights.size(); i++)
        weights[i] = randf(-0.001f, 0.001f);

    traces = FloatBuffer(weights.size(), 0.0f);

    goalCIs = IntBuffer(numHiddenColumns, 0);
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
}

int RLAdapter::size() const {
    int size = sizeof(Int3) + 3 * sizeof(float) + goalCIs.size() * sizeof(int);

    size += 2 * weights.size() * sizeof(float);

    return size;
}

int RLAdapter::stateSize() const {
    return traces.size() * sizeof(float);
}

void RLAdapter::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&discount), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&traceDecay), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&goalCIs[0]), goalCIs.size() * sizeof(int));

    writer.write(reinterpret_cast<const void*>(&weights[0]), weights.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&traces[0]), traces.size() * sizeof(float));
}

void RLAdapter::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&lr), sizeof(float));
    reader.read(reinterpret_cast<void*>(&discount), sizeof(float));
    reader.read(reinterpret_cast<void*>(&traceDecay), sizeof(float));

    goalCIs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&goalCIs[0]), goalCIs.size() * sizeof(int));

    weights.resize(numHiddenCells);
    traces.resize(weights.size());

    reader.read(reinterpret_cast<void*>(&weights[0]), weights.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&traces[0]), traces.size() * sizeof(float));
}

void RLAdapter::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&goalCIs[0]), goalCIs.size() * sizeof(int));

    writer.write(reinterpret_cast<const void*>(&traces[0]), traces.size() * sizeof(float));
}

void RLAdapter::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&goalCIs[0]), goalCIs.size() * sizeof(int));

    reader.read(reinterpret_cast<void*>(&traces[0]), traces.size() * sizeof(float));
}
