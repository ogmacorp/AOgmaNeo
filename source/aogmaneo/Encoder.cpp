// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Encoder.h"

using namespace aon;

void Encoder::forward(
    const Int2 &columnPos,
    const Array<const ByteBuffer*> &inputCIs
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        hiddenActivations[hiddenCellIndex] = 0.0f;
    }

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

        float scale = vl.importance / ((iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1));

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                float inValue = static_cast<float>((*inputCIs[vli])[visibleColumnIndex]) / static_cast<float>(vld.size.z - 1) * 2.0f - 1.0f;

                for (int hc = 0; hc < hiddenSize.z; hc++) {
                    int hiddenCellIndex = hc + hiddenCellsStart;

                    int wi = offset.y + diam * (offset.x + diam * hiddenCellIndex);

                    float delta = inValue - vl.protos[wi] * halfByteInv;

                    hiddenActivations[hiddenCellIndex] -= delta * delta * scale;
                }
            }
    }

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        if (hiddenActivations[hiddenCellIndex] > maxActivation || maxIndex == -1) {
            maxActivation = hiddenActivations[hiddenCellIndex];
            maxIndex = hc;
        }
    }

    hiddenCIs[hiddenColumnIndex] = maxIndex;
}

void Encoder::inhibit(
    const Int2 &columnPos
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    float activation = hiddenActivations[hiddenCIs[hiddenColumnIndex] + hiddenSize.z * hiddenColumnIndex];

    hiddenSuperWinners[hiddenColumnIndex] = true;

    for (int dx = -neighborRadius; dx <= neighborRadius; dx++)
        for (int dy = -neighborRadius; dy <= neighborRadius; dy++) {
            Int2 neighborPos(columnPos.x + dx, columnPos.y + dy);

            if (inBounds0(neighborPos, Int2(hiddenSize.x, hiddenSize.y))) {
                int neighborColumnIndex = address2(neighborPos, Int2(hiddenSize.x, hiddenSize.y));

                if (hiddenActivations[hiddenCIs[neighborColumnIndex] + hiddenSize.z * neighborColumnIndex] > activation) {
                    hiddenSuperWinners[hiddenColumnIndex] = false;

                    return;
                }
            }
        }
}

void Encoder::learn(
    const Int2 &columnPos,
    const Array<const ByteBuffer*> &inputCIs
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    // Find strength
    float superStrength = 0.0f;

    for (int dx = -learnRadius; dx <= learnRadius; dx++)
        for (int dy = -learnRadius; dy <= learnRadius; dy++) {
            Int2 neighborPos(columnPos.x + dx, columnPos.y + dy);

            if (inBounds0(neighborPos, Int2(hiddenSize.x, hiddenSize.y))) {
                int neighborColumnIndex = address2(neighborPos, Int2(hiddenSize.x, hiddenSize.y));

                if (hiddenSuperWinners[neighborColumnIndex]) {
                    float dist = min(abs(dx), abs(dy));
                    float subStrength = 1.0f - dist / static_cast<float>(learnRadius + 1);

                    superStrength = max(superStrength, subStrength);
                }
            }
        }

    for (int dhc = -1; dhc <= 1; dhc++) {
        int hc = hiddenCIs[hiddenColumnIndex] + dhc;

        if (hc < 0 || hc >= hiddenSize.z)
            continue;

        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        float strength = superStrength * hiddenRates[hiddenCellIndex];

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
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    float inValue = static_cast<float>((*inputCIs[vli])[visibleColumnIndex]) / static_cast<float>(vld.size.z - 1) * 2.0f - 1.0f;

                    int wi = offset.y + diam * (offset.x + diam * hiddenCellIndex);

                    float delta = inValue - vl.protos[wi] * halfByteInv;

                    vl.protos[wi] = min(127, max(-127, roundftoi(vl.protos[wi] + strength * 127.0f * delta)));
                }
        }

        hiddenRates[hiddenCellIndex] -= lr * strength;
    }
}

void Encoder::initRandom(
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

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.protos.resize(numHiddenCells * area);

        // Initialize to random values
        for (int i = 0; i < vl.protos.size(); i++)
            vl.protos[i] = rand() % 8 - 4;
    }

    hiddenActivations = FloatBuffer(numHiddenCells, 0.0f);
    hiddenSuperWinners = ByteBuffer(numHiddenCells, false);

    hiddenCIs = ByteBuffer(numHiddenColumns, hiddenSize.z / 2);

    hiddenRates = FloatBuffer(numHiddenCells, 0.5f);
}

void Encoder::step(
    const Array<const ByteBuffer*> &inputCIs,
    bool learnEnabled
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    // Activate / learn
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCIs);

    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        inhibit(Int2(i / hiddenSize.y, i % hiddenSize.y));

    if (learnEnabled) {
        #pragma omp parallel for
        for (int i = 0; i < numHiddenColumns; i++)
            learn(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCIs);
    }
}

int Encoder::size() const {
    int size = sizeof(Int3) + 2 * sizeof(int) + sizeof(float) + hiddenCIs.size() * sizeof(Byte) + hiddenRates.size() * sizeof(float) + sizeof(int);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];

        size += sizeof(VisibleLayerDesc) + vl.protos.size() * sizeof(SByte);
    }

    return size;
}

int Encoder::stateSize() const {
    return hiddenCIs.size() * sizeof(Byte);
}

void Encoder::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&neighborRadius), sizeof(int));
    writer.write(reinterpret_cast<const void*>(&learnRadius), sizeof(int));
    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(Byte));
    writer.write(reinterpret_cast<const void*>(&hiddenRates[0]), hiddenRates.size() * sizeof(float));
    
    int numVisibleLayers = visibleLayers.size();

    writer.write(reinterpret_cast<const void*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

        writer.write(reinterpret_cast<const void*>(&vl.protos[0]), vl.protos.size() * sizeof(SByte));
    }
}

void Encoder::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&neighborRadius), sizeof(int));
    reader.read(reinterpret_cast<void*>(&learnRadius), sizeof(int));
    reader.read(reinterpret_cast<void*>(&lr), sizeof(float));

    hiddenCIs.resize(numHiddenColumns);
    hiddenRates.resize(numHiddenCells);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(Byte));
    reader.read(reinterpret_cast<void*>(&hiddenRates[0]), hiddenRates.size() * sizeof(float));

    hiddenActivations = FloatBuffer(numHiddenCells, 0.0f);
    hiddenSuperWinners = ByteBuffer(numHiddenCells, false);

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

        vl.protos.resize(numHiddenCells * area);

        reader.read(reinterpret_cast<void*>(&vl.protos[0]), vl.protos.size() * sizeof(SByte));
    }
}

void Encoder::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(Byte));
}

void Encoder::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(Byte));
}
