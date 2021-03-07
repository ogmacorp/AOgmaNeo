// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Predictor.h"

using namespace aon;

void Predictor::forward(
    const Int2 &columnPos
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    int interCI = sc.getHiddenCIs()[hiddenColumnIndex];

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        hiddenActivations[hiddenCellIndex] = weights[address4(Int4(columnPos.x, columnPos.y, hc, interCI), Int4(hiddenSize.x, hiddenSize.y, hiddenSize.z, sc.getHiddenSize().z))];

        if (hiddenActivations[hiddenCellIndex] > maxActivation || maxIndex == -1) {
            maxActivation = hiddenActivations[hiddenCellIndex];
            maxIndex = hc;
        }
    }

    float total = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        hiddenActivations[hiddenCellIndex] = expf(hiddenActivations[hiddenCellIndex] - maxActivation);

        total += hiddenActivations[hiddenCellIndex];
    }

    float scale = 1.0f / max(0.0001f, total);

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        hiddenActivations[hiddenCellIndex] *= scale;
    }

    hiddenCIs[hiddenColumnIndex] = maxIndex;
}

void Predictor::learn(
    const Int2 &columnPos,
    const IntBuffer* hiddenTargetCIs
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int targetCI = (*hiddenTargetCIs)[hiddenColumnIndex];

    // Missing value handling
    if (targetCI == -1)
        return;

    int interCIPrev = sc.getHiddenCIs()[hiddenColumnIndex];

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        float delta = beta * ((hc == targetCI ? 1.0f : 0.0f) - hiddenActivations[hiddenCellIndex]);
            
        weights[address4(Int4(columnPos.x, columnPos.y, hc, interCIPrev), Int4(hiddenSize.x, hiddenSize.y, hiddenSize.z, sc.getHiddenSize().z))] += delta;
    }
}

void Predictor::initRandom(
    const Int3 &hiddenSize,
    int intermediateSize,
    const Array<SparseCoder::VisibleLayerDesc> &visibleLayerDescs
) {
    this->hiddenSize = hiddenSize;

    sc.initRandom(Int3(hiddenSize.x, hiddenSize.y, intermediateSize), visibleLayerDescs);

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;
    
    weights.resize(numHiddenCells * intermediateSize);

    for (int i = 0; i < weights.size(); i++)
        weights[i] = randf(-0.01f, 0.01f);

    hiddenActivations = FloatBuffer(numHiddenCells, 0.0f);

    hiddenCIs = IntBuffer(numHiddenColumns, 0);
}

void Predictor::step(
    const Array<const IntBuffer*> &inputCIs,
    const IntBuffer* hiddenTargetCIs,
    bool learnEnabled
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    if (learnEnabled) {
        // Learn kernel
        #pragma omp parallel for
        for (int i = 0; i < numHiddenColumns; i++)
            learn(Int2(i / hiddenSize.y, i % hiddenSize.y), hiddenTargetCIs);
    }

    sc.step(inputCIs, learnEnabled);

    // Forward kernel
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        forward(Int2(i / hiddenSize.y, i % hiddenSize.y));
}

int Predictor::size() const {
    int size = sizeof(Int3) + sizeof(float) + hiddenCIs.size() * sizeof(int) + hiddenActivations.size() * sizeof(float) + sizeof(int);

    size += sc.size();

    size += weights.size() * sizeof(float);

    return size;
}

int Predictor::stateSize() const {
    int size = hiddenCIs.size() * sizeof(int) + hiddenActivations.size() * sizeof(float);

    size += sc.stateSize();

    return size;
}

void Predictor::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&beta), sizeof(float));

    sc.write(writer);

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hiddenActivations[0]), hiddenActivations.size() * sizeof(float));

    writer.write(reinterpret_cast<const void*>(&weights[0]), weights.size() * sizeof(float));
}

void Predictor::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&beta), sizeof(float));

    sc.read(reader);

    hiddenCIs.resize(numHiddenColumns);
    hiddenActivations.resize(numHiddenCells);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hiddenActivations[0]), hiddenActivations.size() * sizeof(float));

    weights.resize(numHiddenCells * sc.getHiddenSize().z);

    reader.read(reinterpret_cast<void*>(&weights[0]), weights.size() * sizeof(float));
}

void Predictor::writeState(
    StreamWriter &writer
) const {
    sc.writeState(writer);

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hiddenActivations[0]), hiddenActivations.size() * sizeof(float));
}

void Predictor::readState(
    StreamReader &reader
) {
    sc.readState(reader);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hiddenActivations[0]), hiddenActivations.size() * sizeof(float));
}
