// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Encoder.h"

using namespace aon;

void Encoder::activate(
    const Int2 &columnPos,
    const Array<const IntBuffer*> &inputCIs,
    bool learnEnabled
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    int maxIndex = -1;
    float maxActivation = 0.0f;

    int backupMaxIndex = -1;
    float backupMaxActivation = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        float sum = 0.0f;
        float weightSum = 0.0f;
        int count = 0;

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

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

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    if (vl.mask[offset.y + diam * (offset.x + diam * hiddenColumnIndex)]) {
                        int inCI = (*inputCIs[vli])[visibleColumnIndex];

                        int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                        sum += vl.weights[inCI + wiStart];

                        for (int vc = 0; vc < vld.size.z; vc++) {
                            int wi = vc + wiStart;

                            weightSum += vl.weights[wi];
                        }

                        count++;
                    }
                }
        }

        sum *= byteInv;
        weightSum *= byteInv;

        float activation = sum / (gap + weightSum);
        float match = sum / count;

        if (match >= vigilance) {
            if (activation > maxActivation || maxIndex == -1) {
                maxActivation = activation;
                maxIndex = hc;
            }
        }

        if (activation > backupMaxActivation || backupMaxIndex == -1) {
            backupMaxActivation = activation;
            backupMaxIndex = hc;
        }
    }

    bool found = maxIndex != -1;

    if (!found)
        maxIndex = backupMaxIndex;

    hiddenCIs[hiddenColumnIndex] = maxIndex;

    if (learnEnabled && found) {
        int hiddenCellIndexMax = maxIndex + hiddenCellsStart;

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

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

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    int inCI = (*inputCIs[vli])[visibleColumnIndex];

                    int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndexMax));

                    for (int vc = 0; vc < vld.size.z; vc++) {
                        int wi = vc + wiStart;

                        vl.weights[wi] = max(0, vl.weights[wi] + roundf(lr * min(0.0f, (vc == inCI) * 255.0f - vl.weights[wi])));
                    }
                }
        }
    }
}

void Encoder::initRandom(
    const Int3 &hiddenSize,
    float maskRatio,
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

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(numHiddenCells * area * vld.size.z);

        for (int i = 0; i < vl.weights.size(); i++)
            vl.weights[i] = 255 - rand() % 2;

        vl.mask.resize(numHiddenColumns * area);

        for (int i = 0; i < vl.mask.size(); i++)
            vl.mask[i] = randf() < maskRatio;
    }

    hiddenCIs = IntBuffer(numHiddenColumns, 0);
}

void Encoder::step(
    const Array<const IntBuffer*> &inputCIs,
    bool learnEnabled
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        activate(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCIs, learnEnabled);
}

int Encoder::size() const {
    int size = sizeof(Int3) + 3 * sizeof(float) + hiddenCIs.size() * sizeof(int) + sizeof(int);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];

        size += sizeof(VisibleLayerDesc) + vl.weights.size() * sizeof(Byte) + vl.mask.size() * sizeof(Byte) + sizeof(float);
    }

    return size;
}

int Encoder::stateSize() const {
    return hiddenCIs.size() * sizeof(int);
}

void Encoder::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&gap), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&vigilance), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    int numVisibleLayers = visibleLayers.size();

    writer.write(reinterpret_cast<const void*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

        writer.write(reinterpret_cast<const void*>(&vl.weights[0]), vl.weights.size() * sizeof(Byte));
        writer.write(reinterpret_cast<const void*>(&vl.mask[0]), vl.mask.size() * sizeof(Byte));

        writer.write(reinterpret_cast<const void*>(&vl.importance), sizeof(float));
    }
}

void Encoder::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&gap), sizeof(float));
    reader.read(reinterpret_cast<void*>(&vigilance), sizeof(float));
    reader.read(reinterpret_cast<void*>(&lr), sizeof(float));

    hiddenCIs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

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

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(numHiddenCells * area * vld.size.z);
        vl.mask.resize(numHiddenCells * area);

        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(Byte));
        reader.read(reinterpret_cast<void*>(&vl.mask[0]), vl.mask.size() * sizeof(Byte));

        reader.read(reinterpret_cast<void*>(&vl.importance), sizeof(float));
    }
}

void Encoder::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
}

void Encoder::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
}
