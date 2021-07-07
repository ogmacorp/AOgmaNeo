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
    const Array<const IntBuffer*> &inputCIs
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        float sum = 0.0f;
        float totalImportance = 0.0f;

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

            float subSum = 0.0f;
            int subCount = (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1);

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    int inCI = (*inputCIs[vli])[visibleColumnIndex];

                    subSum += vl.weights[inCI + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex))];
                }

            subSum /= max(1, subCount);

            sum += subSum * vl.importance;
            totalImportance += vl.importance;
        }

        sum /= max(0.0001f, totalImportance);

        hiddenStimuli[hiddenCellIndex] = sum;
        hiddenActivations[hiddenCellIndex] = 0.0f;

        if (sum > maxActivation || maxIndex == -1) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }
}

void Encoder::inhibit(
    const Int2 &columnPos
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int diam = lRadius * 2 + 1;

    // Lower corner
    Int2 fieldLowerBound(columnPos.x - lRadius, columnPos.y - lRadius);

    // Bounds of receptive field, clamped to input size
    Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
    Int2 iterUpperBound(min(hiddenSize.x - 1, columnPos.x + lRadius), min(hiddenSize.y - 1, columnPos.y + lRadius));

    int count = (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1) - 1; // -1 for self-connection

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        float sum = 0.0f;

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int otherHiddenColumnIndex = address2(Int2(ix, iy), Int2(hiddenSize.x, hiddenSize.y));

                if (otherHiddenColumnIndex == hiddenColumnIndex) // No self-inhibition
                    continue;

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int inCI = hiddenCIsTemp[otherHiddenColumnIndex];

                sum += laterals[inCI + hiddenSize.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex))];
            }

        sum /= max(1, count);

        hiddenActivations[hiddenCellIndex] += actRate * (hiddenStimuli[hiddenCellIndex] - sum - hiddenActivations[hiddenCellIndex]);

        if (hiddenActivations[hiddenCellIndex] > maxActivation || maxIndex == -1) {
            maxActivation = hiddenActivations[hiddenCellIndex];
            maxIndex = hc;
        }
    }

    hiddenCIs[hiddenColumnIndex] = maxIndex;
}

void Encoder::learn(
    const Int2 &columnPos,
    const Array<const IntBuffer*> &inputCIs
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellIndexMax = address3(Int3(columnPos.x, columnPos.y, hiddenCIs[hiddenColumnIndex]), hiddenSize);

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

                int inCI = (*inputCIs[vli])[visibleColumnIndex];

                int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndexMax));

                for (int vc = 0; vc < vld.size.z; vc++) {
                    int wi = vc + wiStart;

                    vl.weights[wi] += hiddenRates[hiddenCellIndexMax] * ((vc == inCI) - vl.weights[wi]);
                }
            }
    }

    int diam = lRadius * 2 + 1;

    // Lower corner
    Int2 fieldLowerBound(columnPos.x - lRadius, columnPos.y - lRadius);

    // Bounds of receptive field, clamped to input size
    Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
    Int2 iterUpperBound(min(hiddenSize.x - 1, columnPos.x + lRadius), min(hiddenSize.y - 1, columnPos.y + lRadius));

    for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
        for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
            int otherHiddenColumnIndex = address2(Int2(ix, iy), Int2(hiddenSize.x, hiddenSize.y));

            Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

            int inCI = hiddenCIs[otherHiddenColumnIndex];

            int wiStart = hiddenSize.z * (offset.y + diam * (offset.x + diam * hiddenCellIndexMax));

            for (int vc = 0; vc < hiddenSize.z; vc++) {
                int wi = vc + wiStart;

                laterals[wi] += hiddenRates[hiddenCellIndexMax] * ((vc == inCI) - laterals[wi]);
            }
        }

    hiddenRates[hiddenCellIndexMax] -= lr * hiddenRates[hiddenCellIndexMax];
}

void Encoder::initRandom(
    const Int3 &hiddenSize,
    int lRadius,
    const Array<VisibleLayerDesc> &visibleLayerDescs
) {
    this->visibleLayerDescs = visibleLayerDescs;

    this->hiddenSize = hiddenSize;

    this->lRadius = lRadius;

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

        for (int i = 0; i < vl.weights.size(); i++)
            vl.weights[i] = randf(0.0f, 1.0f);
    }

    hiddenStimuli = FloatBuffer(numHiddenCells, 0.0f);
    hiddenActivations = FloatBuffer(numHiddenCells, 0.0f);

    // Hidden CIs
    hiddenCIs = IntBuffer(numHiddenColumns, 0);
    hiddenCIsTemp = IntBuffer(numHiddenColumns, 0);

    hiddenRates = FloatBuffer(numHiddenCells, 0.5f);

    int diam = lRadius * 2 + 1;
    int area = diam * diam;

    laterals.resize(numHiddenCells * area * hiddenSize.z, 0.0f);
}

// Activate the sparse coder (perform sparse coding)
void Encoder::step(
    const Array<const IntBuffer*> &inputCIs, // Input states
    bool learnEnabled // Whether to learn
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCIs);

    for (int it = 0; it < explainIters; it++) {
        hiddenCIsTemp = hiddenCIs;

        #pragma omp parallel for
        for (int i = 0; i < numHiddenColumns; i++)
            inhibit(Int2(i / hiddenSize.y, i % hiddenSize.y));
    }

    if (learnEnabled) {
        #pragma omp parallel for
        for (int i = 0; i < numHiddenColumns; i++)
            learn(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCIs);
    }
}

int Encoder::size() const {
    int size = sizeof(Int3) + 2 * sizeof(int) + 2 * sizeof(float) + hiddenCIs.size() * sizeof(float) + hiddenActivations.size() * sizeof(float) + sizeof(int);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        size += sizeof(VisibleLayerDesc) + sizeof(int) + vl.weights.size() * sizeof(float);
    }

    size += laterals.size() * sizeof(float);

    return size;
}

int Encoder::stateSize() const {
    return hiddenCIs.size() * sizeof(float);
}

void Encoder::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));
    writer.write(reinterpret_cast<const void*>(&lRadius), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&explainIters), sizeof(int));
    writer.write(reinterpret_cast<const void*>(&actRate), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hiddenRates[0]), hiddenRates.size() * sizeof(float));

    int numVisibleLayers = visibleLayers.size();

    writer.write(reinterpret_cast<const void*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

        int weightsSize = vl.weights.size();

        writer.write(reinterpret_cast<const void*>(&weightsSize), sizeof(int));

        writer.write(reinterpret_cast<const void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));
    }

    int lateralsSize = laterals.size();

    writer.write(reinterpret_cast<const void*>(&lateralsSize), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&laterals[0]), laterals.size() * sizeof(float));
}

void Encoder::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));
    reader.read(reinterpret_cast<void*>(&lRadius), sizeof(int));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&explainIters), sizeof(int));
    reader.read(reinterpret_cast<void*>(&actRate), sizeof(float));
    reader.read(reinterpret_cast<void*>(&lr), sizeof(float));

    hiddenCIs.resize(numHiddenColumns);
    hiddenCIsTemp.resize(numHiddenColumns);
    hiddenRates.resize(numHiddenCells);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hiddenRates[0]), hiddenRates.size() * sizeof(float));

    hiddenStimuli = FloatBuffer(numHiddenCells, 0.0f);
    hiddenActivations = FloatBuffer(numHiddenCells, 0.0f);

    int numVisibleLayers = visibleLayers.size();

    reader.read(reinterpret_cast<void*>(&numVisibleLayers), sizeof(int));

    visibleLayers.resize(numVisibleLayers);
    visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = visibleLayerDescs[vli];

        reader.read(reinterpret_cast<void*>(&vld), sizeof(VisibleLayerDesc));

        int weightsSize;

        reader.read(reinterpret_cast<void*>(&weightsSize), sizeof(int));

        vl.weights.resize(weightsSize);

        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));
    }

    int lateralsSize;

    reader.read(reinterpret_cast<void*>(&lateralsSize), sizeof(int));

    laterals.resize(lateralsSize);

    reader.read(reinterpret_cast<void*>(&laterals[0]), laterals.size() * sizeof(float));
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

