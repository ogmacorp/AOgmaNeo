// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Encoder.h"

using namespace aon;

void Encoder::resetReconstruction(
    const Int2 &columnPos,
    const IntBuffer* inputCIs,
    int vli
) {
    VisibleLayer &vl = visibleLayers[vli];
    VisibleLayerDesc &vld = visibleLayerDescs[vli];

    int visibleColumnIndex = address2(columnPos, Int2(vld.size.x, vld.size.y));

    vl.reconstruction[visibleColumnIndex] = static_cast<float>((*inputCIs)[visibleColumnIndex]) / static_cast<float>(vld.size.z - 1) * 2.0f - 1.0f;
}

void Encoder::forward(
    const Int2 &clumpPos,
    int priority,
    bool learnEnabled
) {
    Int2 clumpOffset(priority / clumpSize.y, priority % clumpSize.y);
    Int2 columnPos(clumpPos.x * clumpSize.x + clumpOffset.x, clumpPos.y * clumpSize.y + clumpOffset.y);

    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        float sum = 0.0f;

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int diam = vld.radius * 2 + 1;

            // Projection
            Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(numClumps.x),
                static_cast<float>(vld.size.y) / static_cast<float>(numClumps.y));

            Int2 visibleCenter = project(clumpPos, hToV);

            // Lower corner
            Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

            // Bounds of receptive field, clamped to input size
            Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
            Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

            float subSum = 0.0f;
            int subCount = (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1);

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    float inValue = vl.reconstruction[visibleColumnIndex];

                    int wi = offset.y + diam * (offset.x + diam * hiddenCellIndex);

                    float delta = inValue - vl.protos[wi];

                    subSum -= delta * delta;
                }

            subSum /= subCount;

            sum += subSum * vl.importance;
        }

        if (sum > maxActivation || maxIndex == -1) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    hiddenCIs[hiddenColumnIndex] = maxIndex;

    if (learnEnabled) {
        for (int hc = 0; hc < hiddenSize.z; hc++) {
            int hiddenCellIndex = hc + hiddenCellsStart;

            float diff = maxIndex - hc;
            diff /= hiddenSize.z;

            float rate = expf(-falloff * diff * diff / max(0.0001f, hiddenRates[hiddenCellIndex])) * hiddenRates[hiddenCellIndex];

            for (int vli = 0; vli < visibleLayers.size(); vli++) {
                VisibleLayer &vl = visibleLayers[vli];
                const VisibleLayerDesc &vld = visibleLayerDescs[vli];

                int diam = vld.radius * 2 + 1;

                // Projection
                Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(numClumps.x),
                    static_cast<float>(vld.size.y) / static_cast<float>(numClumps.y));

                Int2 visibleCenter = project(clumpPos, hToV);

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

                        vl.protos[wi] += rate * (vl.reconstruction[visibleColumnIndex] - vl.protos[wi]);
                    }
            }

            hiddenRates[hiddenCellIndex] -= lr * rate;
        }
    }
}

void Encoder::reconstruct(
    const Int2 &columnPos,
    int priority,
    int vli
) {
    VisibleLayer &vl = visibleLayers[vli];
    VisibleLayerDesc &vld = visibleLayerDescs[vli];

    int diam = vld.radius * 2 + 1;

    int visibleColumnIndex = address2(columnPos, Int2(vld.size.x, vld.size.y));

    Int2 clumpOffset(priority / clumpSize.y, priority % clumpSize.y);

    // Projection
    Float2 vToH = Float2(static_cast<float>(numClumps.x) / static_cast<float>(vld.size.x),
        static_cast<float>(numClumps.y) / static_cast<float>(vld.size.y));

    Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(numClumps.x),
        static_cast<float>(vld.size.y) / static_cast<float>(numClumps.y));
                
    Int2 clumpCenter = project(columnPos, vToH);

    Int2 reverseRadii(ceilf(vToH.x * vld.radius) + 1, ceilf(vToH.y * vld.radius) + 1);
    
    // Lower corner
    Int2 fieldLowerBound(clumpCenter.x - reverseRadii.x, clumpCenter.y - reverseRadii.y);

    // Bounds of receptive field, clamped to input size
    Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
    Int2 iterUpperBound(min(numClumps.x - 1, clumpCenter.x + reverseRadii.x), min(numClumps.y - 1, clumpCenter.y + reverseRadii.y));

    float sum = 0.0f;
    int count = 0;

    for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
        for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
            Int2 clumpPos(ix, iy);

            Int2 hiddenPos(clumpPos.x * clumpSize.x + clumpOffset.x, clumpPos.y * clumpSize.y + clumpOffset.y);

            int hiddenColumnIndex = address2(hiddenPos, Int2(hiddenSize.x, hiddenSize.y));

            int hiddenCellIndex = hiddenColumnIndex * hiddenSize.z + hiddenCIs[hiddenColumnIndex];

            Int2 visibleCenter = project(clumpPos, hToV);

            if (inBounds(columnPos, Int2(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius), Int2(visibleCenter.x + vld.radius + 1, visibleCenter.y + vld.radius + 1))) {
                Int2 offset(columnPos.x - visibleCenter.x + vld.radius, columnPos.y - visibleCenter.y + vld.radius);

                int wi = offset.y + diam * (offset.x + diam * hiddenCellIndex);

                sum += vl.protos[wi];
                count++;
            }
        }

    sum /= max(1, count);

    vl.reconstruction[visibleColumnIndex] -= sum;
}

void Encoder::initRandom(
    const Int3 &hiddenSize,
    const Int2 &clumpSize,
    const Array<VisibleLayerDesc> &visibleLayerDescs
) {
    this->visibleLayerDescs = visibleLayerDescs;

    this->hiddenSize = hiddenSize;
    this->clumpSize = clumpSize;
    this->numClumps = Int2(hiddenSize.x / clumpSize.x, hiddenSize.y / clumpSize.y);

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
            vl.protos[i] = randf(-0.0001f, 0.0001f);

        vl.reconstruction = FloatBuffer(numVisibleColumns, 0.0f);
    }

    hiddenCIs = IntBuffer(numHiddenColumns, 0);

    hiddenRates = FloatBuffer(numHiddenCells, 1.0f);
}

void Encoder::step(
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
    int numPriorities = clumpSize.x * clumpSize.y;
    int totalClumps = numClumps.x * numClumps.y;

    for (int p = 0; p < numPriorities; p++) {
        #pragma omp parallel for
        for (int i = 0; i < totalClumps; i++)
            forward(Int2(i / numClumps.y, i % numClumps.y), p, learnEnabled);

        if (p < numPriorities - 1) {
            for (int vli = 0; vli < visibleLayers.size(); vli++) {
                const VisibleLayerDesc &vld = visibleLayerDescs[vli];

                int numVisibleColumns = vld.size.x * vld.size.y;

                #pragma omp parallel for
                for (int i = 0; i < numVisibleColumns; i++)
                    reconstruct(Int2(i / vld.size.y, i % vld.size.y), p, vli);
            }
        }
    }
}

int Encoder::size() const {
    int size = sizeof(Int3) + sizeof(Int2) + 2 * sizeof(float) + hiddenCIs.size() * sizeof(int) + hiddenRates.size() * sizeof(float) + sizeof(int);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];

        size += sizeof(VisibleLayerDesc) + vl.protos.size() * sizeof(float);
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
    writer.write(reinterpret_cast<const void*>(&clumpSize), sizeof(Int2));

    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&falloff), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hiddenRates[0]), hiddenRates.size() * sizeof(float));
    
    int numVisibleLayers = visibleLayers.size();

    writer.write(reinterpret_cast<const void*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

        writer.write(reinterpret_cast<const void*>(&vl.protos[0]), vl.protos.size() * sizeof(float));

        writer.write(reinterpret_cast<const void*>(&vl.importance), sizeof(float));
    }
}

void Encoder::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));
    reader.read(reinterpret_cast<void*>(&clumpSize), sizeof(Int2));

    numClumps = Int2(hiddenSize.x / clumpSize.x, hiddenSize.y / clumpSize.y);

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&lr), sizeof(float));
    reader.read(reinterpret_cast<void*>(&falloff), sizeof(float));

    hiddenCIs.resize(numHiddenColumns);
    hiddenRates.resize(numHiddenCells);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hiddenRates[0]), hiddenRates.size() * sizeof(float));

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

        vl.reconstruction = FloatBuffer(numVisibleColumns, 0.0f);

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
