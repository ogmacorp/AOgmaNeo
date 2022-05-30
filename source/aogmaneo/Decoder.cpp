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
    const Array<const IntBuffer*> &inputCIs
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    int supportCI = support.getHiddenCIs()[hiddenColumnIndex];

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        int wi = supportCI + support.getHiddenSize().z * hiddenCellIndex;

        float activation = weights[wi];

        if (activation > maxActivation || maxIndex == -1) {
            maxActivation = activation;
            maxIndex = hc;
        }
    }

    hiddenCIs[hiddenColumnIndex] = maxIndex;
}

void Decoder::learn(
    const Int2 &columnPos,
    const IntBuffer* hiddenTargetCIs
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    int supportCI = support.getHiddenCIs()[hiddenColumnIndex];

    int targetCI = (*hiddenTargetCIs)[hiddenColumnIndex];

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        int wi = supportCI + support.getHiddenSize().z * hiddenCellIndex;

        weights[wi] += lr * ((hc == targetCI) - weights[wi]);
    }
}

void Decoder::initRandom(
    const Int3 &hiddenSize,
    int supportSize,
    const Array<Encoder::VisibleLayerDesc> &visibleLayerDescs
) {
    this->hiddenSize = hiddenSize;

    support.initRandom(Int3(hiddenSize.x, hiddenSize.y, supportSize), 1.0f, visibleLayerDescs);

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;
    
    weights.resize(numHiddenCells * supportSize);

    for (int i = 0; i < weights.size(); i++)
        weights[i] = randf(0.0f, 0.01f);

    hiddenCIs = IntBuffer(numHiddenColumns, 0);
}

void Decoder::step(
    const Array<const IntBuffer*> &inputCIs,
    const IntBuffer* hiddenTargetCIs,
    bool learnEnabled 
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    if (learnEnabled) {
        // Forward kernel
        #pragma omp parallel for
        for (int i = 0; i < numHiddenColumns; i++)
            learn(Int2(i / hiddenSize.y, i % hiddenSize.y), hiddenTargetCIs);
    }

    support.step(inputCIs, learnEnabled);

    // Forward kernel
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCIs);
}

int Decoder::size() const {
    int size = sizeof(Int3) + sizeof(float) + hiddenCIs.size() * sizeof(int) + sizeof(int);

    size += support.size();

    size += weights.size() * sizeof(float);

    return size;
}

int Decoder::stateSize() const {
    int size = hiddenCIs.size() * sizeof(int);

    size += support.stateSize();

    return size;
}

void Decoder::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    
    writer.write(reinterpret_cast<const void*>(&weights[0]), weights.size() * sizeof(float));

    support.write(writer);
}

void Decoder::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&lr), sizeof(float));

    hiddenCIs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    support.read(reader);

    weights.resize(numHiddenCells * support.getHiddenSize().z);

    reader.read(reinterpret_cast<void*>(&weights[0]), weights.size() * sizeof(float));
}

void Decoder::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    
    support.writeState(writer);
}

void Decoder::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    support.readState(reader);
}
