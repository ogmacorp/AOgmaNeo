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
    const Int2 &columnPos,
    const Array<const ByteBuffer*> &inputCIs,
    const ByteBuffer* hiddenTargetCIs,
    bool learnEnabled
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));
    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    Byte targetCI = (*hiddenTargetCIs)[hiddenColumnIndex];

    // For each visible layer
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        int diam = vld.radius * 2 + 1;

        // Projection
        Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
            static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));

        Int2 visibleCenter = project(columnPos, hToV);

        visibleCenter = minOverhang(visibleCenter, Int2(vld.size.x, vld.size.y), vld.radius);

        // Lower corner
        Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

        // Bounds of receptive field, clamped to input size
        Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
        Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                Byte inCI = (*inputCIs[vli])[visibleColumnIndex];
                Byte inCIPrev = vl.inputCIsPrev[visibleColumnIndex];

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                for (int hc = 0; hc < hiddenSize.z; hc++) {
                    int hiddenCellIndex = hc + hiddenCellsStart;

                    int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                    if (learnEnabled && targetCI != hiddenCIs[hiddenColumnIndex]) {
                        if (hc == targetCI) {
                            int wi = inCIPrev + wiStart;

                            vl.weights[wi] = min<int>(254, vl.weights[wi]) + 1;
                        }
                        else if (hc == hiddenCIs[hiddenColumnIndex]) {
                            int wi = inCIPrev + wiStart;

                            vl.weights[wi] = max<int>(1, vl.weights[wi]) - 1;
                        }
                    }

                    hiddenSums[hiddenCellIndex] += vl.weights[inCI + wiStart];
                }
            }
    }

    Byte maxIndex = 0;
    int maxActivation = -999999;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        if (hiddenSums[hiddenCellIndex] > maxActivation) {
            maxActivation = hiddenSums[hiddenCellIndex];
            maxIndex = hc;
        }

        hiddenSums[hiddenCellIndex] = 0;
    }

    hiddenCIs[hiddenColumnIndex] = maxIndex;
}

void Predictor::initRandom(
    const Int3 &hiddenSize, // Hidden/output/prediction size
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

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(numHiddenCells * area * vld.size.z);

        for (int i = 0; i < vl.weights.size(); i++)
            vl.weights[i] = 123 + rand() % 8;

        vl.inputCIsPrev = ByteBuffer(numVisibleColumns, 0);
    }

    hiddenSums = IntBuffer(numHiddenCells, 0);
    hiddenCIs = ByteBuffer(numHiddenColumns, 0);
}

// Activate the predictor (predict values)
void Predictor::step(
    const Array<const ByteBuffer*> &inputCIs,
    const ByteBuffer* hiddenTargetCIs,
    bool learnEnabled
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    // Forward kernel
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCIs, hiddenTargetCIs, learnEnabled);

    // Copy to prevs
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];

        vl.inputCIsPrev = *inputCIs[vli];
    }
}

int Predictor::size() const {
    int size = sizeof(Int3) + hiddenCIs.size() * sizeof(Byte) + sizeof(int);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        size += sizeof(VisibleLayerDesc) + vl.weights.size() * sizeof(Byte) + vl.inputCIsPrev.size() * sizeof(Byte);
    }

    return size;
}

int Predictor::stateSize() const {
    int size = hiddenCIs.size() * sizeof(Byte);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];

        size += vl.inputCIsPrev.size() * sizeof(Byte);
    }

    return size;
}

void Predictor::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(Byte));
    
    int numVisibleLayers = visibleLayers.size();

    writer.write(reinterpret_cast<const void*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

        writer.write(reinterpret_cast<const void*>(&vl.weights[0]), vl.weights.size() * sizeof(Byte));

        writer.write(reinterpret_cast<const void*>(&vl.inputCIsPrev[0]), vl.inputCIsPrev.size() * sizeof(Byte));
    }
}

void Predictor::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    hiddenCIs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(Byte));

    hiddenSums = IntBuffer(numHiddenCells, 0);

    int numVisibleLayers = visibleLayers.size();

    reader.read(reinterpret_cast<void*>(&numVisibleLayers), sizeof(int));

    visibleLayers.resize(numVisibleLayers);
    visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = visibleLayerDescs[vli];

        reader.read(reinterpret_cast<void*>(&vld), sizeof(VisibleLayerDesc));

        int numVisibleColumns = vld.size.x * vld.size.y;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(numHiddenCells * area * vld.size.z);

        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(Byte));

        vl.inputCIsPrev.resize(numVisibleColumns);

        reader.read(reinterpret_cast<void*>(&vl.inputCIsPrev[0]), vl.inputCIsPrev.size() * sizeof(Byte));
    }
}

void Predictor::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(Byte));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];

        writer.write(reinterpret_cast<const void*>(&vl.inputCIsPrev[0]), vl.inputCIsPrev.size() * sizeof(Byte));
    }
}

void Predictor::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(Byte));

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];

        reader.read(reinterpret_cast<void*>(&vl.inputCIsPrev[0]), vl.inputCIsPrev.size() * sizeof(Byte));
    }
}
