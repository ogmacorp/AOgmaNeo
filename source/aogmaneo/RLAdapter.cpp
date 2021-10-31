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

    int hiddenCellIndexMax = address3(Int3(columnPos.x, columnPos.y, (*hiddenCIs)[hiddenColumnIndex]), hiddenSize);

    weights[hiddenCellIndexMax] += lr * (reward - weights[hiddenCellIndexMax]);

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

    progCIs[hiddenColumnIndex] = maxIndex;
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
        weights[i] = randf(-1.01f, -1.0f);

    progCIs = IntBuffer(numHiddenColumns, 0);
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
    int size = sizeof(Int3) + sizeof(float) + progCIs.size() * sizeof(int);

    size += weights.size() * sizeof(float);

    return size;
}

int RLAdapter::stateSize() const {
    return progCIs.size() * sizeof(int);
}

void RLAdapter::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&progCIs[0]), progCIs.size() * sizeof(int));

    writer.write(reinterpret_cast<const void*>(&weights[0]), weights.size() * sizeof(float));
}

void RLAdapter::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&lr), sizeof(float));

    progCIs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&progCIs[0]), progCIs.size() * sizeof(int));

    weights.resize(numHiddenCells);

    reader.read(reinterpret_cast<void*>(&weights[0]), weights.size() * sizeof(float));
}

void RLAdapter::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&progCIs[0]), progCIs.size() * sizeof(int));
}

void RLAdapter::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&progCIs[0]), progCIs.size() * sizeof(int));
}
