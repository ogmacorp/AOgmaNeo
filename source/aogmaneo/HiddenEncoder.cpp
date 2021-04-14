// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "HiddenEncoder.h"

using namespace aon;

void HiddenEncoder::forward(
    const Int2 &columnPos,
    const IntBuffer* inputCIs,
    int vli
) {
    VisibleLayer &vl = visibleLayers[vli];
    VisibleLayerDesc &vld = visibleLayerDescs[vli];

    int visibleColumnIndex = address2(columnPos, Int2(vld.size.x, vld.size.y));

    vl.weights.reverse(address3(Int3(columnPos.x, columnPos.y, (*inputCIs)[visibleColumnIndex]), vld.size), 1.0f, hiddenActivations);
}

void HiddenEncoder::activate(
    const Int2 &columnPos
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        if (hiddenActivations[hiddenCellIndex] > maxActivation || maxIndex == -1) {
            maxActivation = hiddenActivations[hiddenCellIndex];
            maxIndex = hc;
        }
    }

    hiddenCIs[hiddenColumnIndex] = maxIndex;
}

void HiddenEncoder::learn(
    const Int2 &columnPos,
    const IntBuffer* inputCIs,
    int vli
) {
    VisibleLayer &vl = visibleLayers[vli];
    VisibleLayerDesc &vld = visibleLayerDescs[vli];

    int visibleColumnIndex = address2(columnPos, Int2(vld.size.x, vld.size.y));

    int targetCI = (*inputCIs)[visibleColumnIndex];

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int vc = 0; vc < vld.size.z; vc++) {
        int visibleCellIndex = address3(Int3(columnPos.x, columnPos.y, vc), vld.size);

        float sum = vl.weights.multiplyCIs(visibleCellIndex, hiddenCIs);
        int count = vl.weights.count(visibleCellIndex);

        sum /= max(1, count);

        vl.reconstruction[visibleCellIndex] = sum;

        if (sum > maxActivation || maxIndex == -1) {
            maxActivation = sum;
            maxIndex = vc;
        }
    }

    if (maxIndex != targetCI) {
        for (int vc = 0; vc < vld.size.z; vc++) {
            int visibleCellIndex = address3(Int3(columnPos.x, columnPos.y, vc), vld.size);

            float delta = lr * ((vc == targetCI ? 1.0f : 0.0f) - sigmoid(vl.reconstruction[visibleCellIndex]));
      
            vl.weights.deltaCIs(visibleCellIndex, hiddenCIs, delta);
        }
    }
}

void HiddenEncoder::initRandom(
    const Int3 &hiddenSize,
    const Array<VisibleLayerDesc> &visibleLayerDescs
) {
    this->visibleLayerDescs = visibleLayerDescs;

    this->hiddenSize = hiddenSize;

    visibleLayers.resize(visibleLayerDescs.size());

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    // Create layers
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisibleCells = numVisibleColumns * vld.size.z;

        vl.weights.init(hiddenSize, vld.size, vld.radius);

        for (int i = 0; i < vl.weights.values.size(); i++)
            vl.weights.values[i] = randf(0.0f, 1.0f);

        vl.reconstruction = FloatBuffer(numVisibleCells, 0.0f);
    }

    hiddenActivations = FloatBuffer(numHiddenCells, 0.0f);
    hiddenCIs = IntBuffer(numHiddenColumns, 0);
}

void HiddenEncoder::step(
    const Array<const IntBuffer*> &inputCIs,
    bool learnEnabled
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    hiddenActivations.fill(0.0f);
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;
    
        #pragma omp parallel for
        for (int i = 0; i < numVisibleColumns; i++)
            forward(Int2(i / vld.size.y, i % vld.size.y), inputCIs[vli], vli);
    }

    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        activate(Int2(i / hiddenSize.y, i % hiddenSize.y));

    if (learnEnabled) {
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int numVisibleColumns = vld.size.x * vld.size.y;
        
            #pragma omp parallel for
            for (int i = 0; i < numVisibleColumns; i++)
                learn(Int2(i / vld.size.y, i % vld.size.y), inputCIs[vli], vli);
        }
    }
}

int HiddenEncoder::size() const {
    int size = sizeof(Int3) + sizeof(float) + hiddenCIs.size() * sizeof(int) + sizeof(int);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];

        size += sizeof(VisibleLayerDesc) + vl.weights.size() * sizeof(float);
    }

    return size;
}

int HiddenEncoder::stateSize() const {
    return hiddenCIs.size() * sizeof(int);
}

void HiddenEncoder::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    int numVisibleLayers = visibleLayers.size();

    writer.write(reinterpret_cast<const void*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

        vl.weights.write(writer);
    }
}

void HiddenEncoder::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&lr), sizeof(float));

    hiddenCIs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    hiddenActivations = FloatBuffer(numHiddenCells, 0.0f);

    int numVisibleLayers = visibleLayers.size();

    reader.read(reinterpret_cast<void*>(&numVisibleLayers), sizeof(int));

    visibleLayers.resize(numVisibleLayers);
    visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = visibleLayerDescs[vli];

        reader.read(reinterpret_cast<void*>(&vld), sizeof(VisibleLayerDesc));

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisibleCells = numVisibleColumns * vld.size.z;

        vl.weights.read(reader);

        vl.reconstruction = FloatBuffer(numVisibleCells, 0.0f);
    }
}

void HiddenEncoder::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
}

void HiddenEncoder::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
}
