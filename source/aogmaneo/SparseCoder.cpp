// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseCoder.h"

using namespace aon;

void SparseCoder::forward(
    const Int2 &columnPos,
    const Array<const IntBuffer*> &inputCIs,
    int priority
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    if (hiddenPriorities[hiddenColumnIndex] != priority)
        return;

    int maxIndex = -1;
    int maxActivation = 0;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        int sum = 0;

        // For each visible layer
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

                    int wi = inCI + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                    sum += vl.weights[wi] * vl.reconstruction[visibleColumnIndex];
                }
        }

        hiddenActivations[hiddenCellIndex] = sum;

        if (sum > maxActivation || maxIndex == -1) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    hiddenCIs[hiddenColumnIndex] = maxIndex;
}

void SparseCoder::reconstruct(
    const Int2 &columnPos,
    const Array<const IntBuffer*> &inputCIs,
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
    int inCI = (*inputCIs[vli])[visibleColumnIndex];

    int sum = 0;
    int count = 0;

    for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
        for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
            Int2 hiddenPos = Int2(ix, iy);

            int hiddenColumnIndex = address2(hiddenPos, Int2(hiddenSize.x, hiddenSize.y));

            if (hiddenPriorities[hiddenColumnIndex] != priority)
                continue;

            int hiddenCellIndex = address3(Int3(hiddenPos.x, hiddenPos.y, hiddenCIs[hiddenColumnIndex]), hiddenSize);

            Int2 visibleCenter = project(hiddenPos, hToV);

            if (inBounds(columnPos, Int2(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius), Int2(visibleCenter.x + vld.radius + 1, visibleCenter.y + vld.radius + 1))) {
                Int2 offset(columnPos.x - visibleCenter.x + vld.radius, columnPos.y - visibleCenter.y + vld.radius);

                sum += vl.weights[inCI + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex))];
                count++;
            }
        }

    vl.reconstruction[visibleColumnIndex] = max<int>(0, static_cast<int>(vl.reconstruction[visibleColumnIndex]) - sum / max(1, count));
}

void SparseCoder::learn(
    const Int2 &columnPos,
    const Array<const IntBuffer*> &inputCIs
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int secondMaxIndex = -1;
    int secondMaxActivation = 0;

    // Find second highest
    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        if (hc == hiddenCIs[hiddenColumnIndex])
            continue;

        if (hiddenActivations[hiddenCellIndex] > secondMaxActivation || secondMaxIndex == -1) {
            secondMaxActivation = hiddenActivations[hiddenCellIndex];

            secondMaxIndex = hc;
        }
    }

    int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hiddenCIs[hiddenColumnIndex]), hiddenSize);
    int hiddenSecondCellIndex = address3(Int3(columnPos.x, columnPos.y, secondMaxIndex), hiddenSize);

    // For each visible layer
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

                int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));
                int wiStartSecond = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenSecondCellIndex));

                int inCI = (*inputCIs[vli])[visibleColumnIndex];

                for (int vc = 0; vc < vld.size.z; vc++) {
                    int wi = vc + wiStart;
                    int wiSecond = vc + wiStartSecond;

                    vl.weights[wi] = roundftoi(min(255.0f, max(0.0f, vl.weights[wi] + hiddenRates[hiddenCellIndex] * ((vc == inCI ? 255.0f : 0.0f) - static_cast<float>(vl.weights[wi])))));
                    vl.weights[wiSecond] = roundftoi(min(255.0f, max(0.0f, vl.weights[wiSecond] + hiddenRates[hiddenSecondCellIndex] * ((vc == inCI ? 255.0f : 0.0f) - static_cast<float>(vl.weights[wiSecond])))));
                }
            }
    }

    hiddenRates[hiddenCellIndex] -= alpha * hiddenRates[hiddenCellIndex];
    hiddenRates[hiddenSecondCellIndex] -= alpha * hiddenRates[hiddenSecondCellIndex];
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
        int numVisible = numVisibleColumns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(numHiddenCells * area * vld.size.z);

        // Initialize to random values
        for (int i = 0; i < vl.weights.size(); i++)
            vl.weights[i] = rand() % 256;

        vl.reconstruction = ByteBuffer(numVisibleColumns, 0);
    }

    hiddenActivations = IntBuffer(numHiddenCells, 0);

    hiddenCIs = IntBuffer(numHiddenColumns, 0);

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

    // Init recons to 0
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];

        vl.reconstruction.fill(255);
    }
    
    for (int p = 0; p < numPriorities; p++) {
        #pragma omp parallel for
        for (int i = 0; i < numHiddenColumns; i++)
            forward(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCIs, p);

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int numVisibleColumns = vld.size.x * vld.size.y;

            #pragma omp parallel for
            for (int i = 0; i < numVisibleColumns; i++)
                reconstruct(Int2(i / vld.size.y, i % vld.size.y), inputCIs, vli, p);
        }
    }

    if (learnEnabled) {
        #pragma omp parallel for
        for (int i = 0; i < numHiddenColumns; i++)
            learn(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCIs);
    }
}

void SparseCoder::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));
    writer.write(reinterpret_cast<const void*>(&numPriorities), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&alpha), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hiddenPriorities[0]), hiddenPriorities.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hiddenRates[0]), hiddenRates.size() * sizeof(float));
    
    int numVisibleLayers = visibleLayers.size();

    writer.write(reinterpret_cast<const void*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

        int weightsSize = vl.weights.size();

        writer.write(reinterpret_cast<const void*>(&weightsSize), sizeof(int));

        writer.write(reinterpret_cast<const void*>(&vl.weights[0]), vl.weights.size() * sizeof(unsigned char));
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

    hiddenCIs.resize(numHiddenColumns);
    hiddenPriorities.resize(numHiddenColumns);
    hiddenRates.resize(numHiddenCells);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hiddenPriorities[0]), hiddenPriorities.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hiddenRates[0]), hiddenRates.size() * sizeof(float));

    hiddenActivations = IntBuffer(numHiddenCells, 0);

    int numVisibleLayers = visibleLayers.size();

    reader.read(reinterpret_cast<void*>(&numVisibleLayers), sizeof(int));

    visibleLayers.resize(numVisibleLayers);
    visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = visibleLayerDescs[vli];

        reader.read(reinterpret_cast<void*>(&vld), sizeof(VisibleLayerDesc));

        int numVisibleColumns = vld.size.x * vld.size.y;

        int weightsSize;

        reader.read(reinterpret_cast<void*>(&weightsSize), sizeof(int));

        vl.weights.resize(weightsSize);

        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(unsigned char));

        vl.reconstruction = ByteBuffer(numVisibleColumns, 0);
    }
}
