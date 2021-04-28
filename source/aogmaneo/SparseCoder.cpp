// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseCoder.h"

using namespace aon;

void SparseCoder::resetReconstruction(
    const Int2 &columnPos,
    const IntBuffer* inputCIs,
    int vli
) {
    VisibleLayer &vl = visibleLayers[vli];
    VisibleLayerDesc &vld = visibleLayerDescs[vli];

    int visibleColumnIndex = address2(columnPos, Int2(vld.size.x, vld.size.y));

    vl.reconstruction[visibleColumnIndex] = static_cast<float>((*inputCIs)[visibleColumnIndex]) / static_cast<float>(vld.size.z - 1) * 2.0f - 1.0f;
}

void SparseCoder::forward(
    const Int2 &columnPos,
    int priority,
    bool learnEnabled
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    if (hiddenPriorities[hiddenColumnIndex] != priority)
        return;

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

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

                float inValue = vl.reconstruction[visibleColumnIndex];

                for (int hc = 0; hc < hiddenSize.z; hc++) {
                    int hiddenCellIndex = hc + hiddenCellsStart;

                    int wi = offset.y + diam * (offset.x + diam * hiddenCellIndex);

                    float delta = inValue - vl.protos[wi];

                    hiddenSums[hiddenCellIndex] -= delta * delta;
                }
            }
    }

    int maxIndex = 0;
    float maxActivation = hiddenSums[0 + hiddenCellsStart];
    hiddenSums[0 + hiddenCellsStart] = 0.0f;

    for (int hc = 1; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        if (hiddenSums[hiddenCellIndex] > maxActivation) {
            maxActivation = hiddenSums[hiddenCellIndex];
            maxIndex = hc;
        }

        hiddenSums[hiddenCellIndex] = 0.0f;
    }

    hiddenCIs[hiddenColumnIndex] = maxIndex;

    if (learnEnabled) {
        for (int hc = 0; hc < hiddenSize.z; hc++) {
            int dist = maxIndex - hc;

            int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

            float strength = expf(-dist * dist * gamma / max(0.0001f, hiddenRates[hiddenCellIndex])) * hiddenRates[hiddenCellIndex];

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
                        int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                        Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                        int wi = offset.y + diam * (offset.x + diam * hiddenCellIndex);

                        vl.protos[wi] += strength * (vl.reconstruction[visibleColumnIndex] - vl.protos[wi]);
                    }
            }

            hiddenRates[hiddenCellIndex] -= alpha * strength;
        }
    }
}

void SparseCoder::reconstruct(
    const Int2 &columnPos,
    int vli,
    int priority
) {
    VisibleLayer &vl = visibleLayers[vli];
    VisibleLayerDesc &vld = visibleLayerDescs[vli];

    int diam = vld.radius * 2 + 1;

    int visibleColumnIndex = address2(columnPos, Int2(vld.size.x, vld.size.y));

    // Projection
    Float2 vToH = Float2(static_cast<float>(hiddenSize.x) / static_cast<float>(vld.size.x),
        static_cast<float>(hiddenSize.y) / static_cast<float>(vld.size.y));

    Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
                static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));
                
    Int2 hiddenCenter = project(columnPos, vToH);

    Int2 reverseRadii(ceilf(vToH.x * vld.radius) + 1, ceilf(vToH.y * vld.radius) + 1);
    
    // Lower corner
    Int2 fieldLowerBound(hiddenCenter.x - reverseRadii.x, hiddenCenter.y - reverseRadii.y);

    // Bounds of receptive field, clamped to input size
    Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
    Int2 iterUpperBound(min(hiddenSize.x - 1, hiddenCenter.x + reverseRadii.x), min(hiddenSize.y - 1, hiddenCenter.y + reverseRadii.y));
    
    // Find current max
    float sum = 0.0f;
    float total = 0.0f;

    for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
        for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
            Int2 hiddenPos = Int2(ix, iy);

            int hiddenColumnIndex = address2(hiddenPos, Int2(hiddenSize.x, hiddenSize.y));

            if (hiddenPriorities[hiddenColumnIndex] != priority)
                continue;

            int hiddenCellIndex = address3(Int3(hiddenPos.x, hiddenPos.y, hiddenCIs[hiddenColumnIndex]), hiddenSize);

            Int2 visibleCenter = project(hiddenPos, hToV);

            visibleCenter = minOverhang(visibleCenter, Int2(vld.size.x, vld.size.y), vld.radius);

            if (inBounds(columnPos, Int2(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius), Int2(visibleCenter.x + vld.radius + 1, visibleCenter.y + vld.radius + 1))) {
                Int2 offset(columnPos.x - visibleCenter.x + vld.radius, columnPos.y - visibleCenter.y + vld.radius);

                int wi = offset.y + diam * (offset.x + diam * hiddenCellIndex);

                float distX = static_cast<float>(abs(columnPos.x - visibleCenter.x)) / static_cast<float>(vld.radius + 1);
                float distY = static_cast<float>(abs(columnPos.y - visibleCenter.y)) / static_cast<float>(vld.radius + 1);

                float strength = min(1.0f - distX, 1.0f - distY);

                sum += strength * vl.protos[wi];
                total += strength;
            }
        }

    vl.reconstruction[visibleColumnIndex] -= sum / max(0.0001f, total);
}

void SparseCoder::initRandom(
    const Int3 &hiddenSize,
    int numPriorities,
    const Array<VisibleLayerDesc> &visibleLayerDescs
) {
    this->visibleLayerDescs = visibleLayerDescs;

    this->hiddenSize = hiddenSize;
    this->numPriorities = numPriorities;

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
            vl.protos[i] = randf(-0.01f, 0.01f);

        vl.reconstruction = FloatBuffer(numVisibleColumns, 0.0f);
    }

    hiddenSums = FloatBuffer(numHiddenCells, 0.0f);

    hiddenCIs = IntBuffer(numHiddenColumns, hiddenSize.z / 2);

    hiddenPriorities = IntBuffer(numHiddenColumns);

    for (int i = 0; i < hiddenPriorities.size(); i++)
        hiddenPriorities[i] = rand() % numPriorities;

    hiddenRates = FloatBuffer(numHiddenCells, 0.5f);
}

void SparseCoder::step(
    const Array<const IntBuffer*> &inputCIs,
    bool learnEnabled
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    
    // Reset
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;

        #pragma omp parallel for
        for (int i = 0; i < numVisibleColumns; i++)
            resetReconstruction(Int2(i / vld.size.y, i % vld.size.y), inputCIs[vli], vli);
    }

    // Activate / learn
    for (int p = 0; p < numPriorities; p++) {
        #pragma omp parallel for
        for (int i = 0; i < numHiddenColumns; i++)
            forward(Int2(i / hiddenSize.y, i % hiddenSize.y), p, learnEnabled);

        if (p < numPriorities - 1) {
            for (int vli = 0; vli < visibleLayers.size(); vli++) {
                const VisibleLayerDesc &vld = visibleLayerDescs[vli];

                int numVisibleColumns = vld.size.x * vld.size.y;

                #pragma omp parallel for
                for (int i = 0; i < numVisibleColumns; i++)
                    reconstruct(Int2(i / vld.size.y, i % vld.size.y), vli, p);
            }
        }
    }
}

int SparseCoder::size() const {
    int size = sizeof(Int3) + 2 * sizeof(int) + sizeof(float) + hiddenCIs.size() * sizeof(int) + hiddenPriorities.size() * sizeof(int) + hiddenRates.size() * sizeof(float) + sizeof(int);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];

        size += sizeof(VisibleLayerDesc) + vl.protos.size() * sizeof(float);
    }

    return size;
}

int SparseCoder::stateSize() const {
    return hiddenCIs.size() * sizeof(int);
}

void SparseCoder::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));
    writer.write(reinterpret_cast<const void*>(&numPriorities), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&alpha), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&gamma), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hiddenPriorities[0]), hiddenPriorities.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hiddenRates[0]), hiddenRates.size() * sizeof(float));
    
    int numVisibleLayers = visibleLayers.size();

    writer.write(reinterpret_cast<const void*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

        writer.write(reinterpret_cast<const void*>(&vl.protos[0]), vl.protos.size() * sizeof(float));
    }
}

void SparseCoder::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));
    reader.read(reinterpret_cast<void*>(&numPriorities), sizeof(int));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&alpha), sizeof(float));
    reader.read(reinterpret_cast<void*>(&gamma), sizeof(float));

    hiddenCIs.resize(numHiddenColumns);
    hiddenPriorities.resize(numHiddenColumns);
    hiddenRates.resize(numHiddenCells);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hiddenPriorities[0]), hiddenPriorities.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hiddenRates[0]), hiddenRates.size() * sizeof(float));

    hiddenSums = FloatBuffer(numHiddenCells, 0.0f);

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

        reader.read(reinterpret_cast<void*>(&vl.protos[0]), vl.protos.size() * sizeof(float));

        vl.reconstruction = FloatBuffer(numVisibleColumns, 0);
    }
}

void SparseCoder::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
}

void SparseCoder::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
}
