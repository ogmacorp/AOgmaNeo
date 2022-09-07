// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "ImageEncoder.h"

using namespace aon;

void ImageEncoder::activate(
    const Int2 &columnPos,
    const Array<const ByteBuffer*> &inputs,
    unsigned int* state
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    int maxIndex = -1;
    float maxActivation = 0.0f;

    int backupMaxIndex = -1;
    float backupMaxActivation = 0.0f;

    float maxMatch = 0.0f;

    for (int hc = 0; hc < hiddenCommits[hiddenColumnIndex]; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        float sum = 0.0f;
        float weightSum = 0.0f;
        float totalImportance = 0.0f;

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

            int subCount = (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1);

            float subSum = 0.0f;
            float subWeightSum = 0.0f;

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    int visibleCellsStart = visibleColumnIndex * vld.size.z;

                    int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                    for (int vc = 0; vc < vld.size.z; vc++) {
                        int visibleCellIndex = vc + visibleCellsStart;

                        int input = (*inputs[vli])[visibleCellIndex];

                        int wi = vc + wiStart;

                        subSum += min(static_cast<int>(vl.weights0[wi]), input) + min(static_cast<int>(vl.weights1[wi]), 255 - input);

                        subWeightSum += static_cast<float>(vl.weights0[wi]) + static_cast<float>(vl.weights1[wi]);
                    }
                }

            subSum /= subCount * 255;
            subWeightSum /= subCount * 255;

            sum += subSum * vl.importance;
            weightSum += subWeightSum * vl.importance;
            totalImportance += vl.importance;
        }

        sum /= max(0.0001f, totalImportance);
        weightSum /= max(0.0001f, totalImportance);

        float activation = sum / (gap + weightSum);
        float match = sum;

        if (match >= vigilance) {
            if (activation > maxActivation || maxIndex == -1) {
                maxActivation = activation;
                maxIndex = hc;
                maxMatch = match;
            }
        }

        if (activation > backupMaxActivation || backupMaxIndex == -1) {
            backupMaxActivation = activation;
            backupMaxIndex = hc;
        }
    }

    bool found = maxIndex != -1;

    if (!found) {
        if (hiddenCommits[hiddenColumnIndex] >= hiddenSize.z) {
            maxIndex = backupMaxIndex;
            maxMatch = 0.0f;
        }
        else {
            maxIndex = hiddenCommits[hiddenColumnIndex];
            maxMatch = 1.0f + randf(state) * 0.0001f;
        }
    }

    hiddenMatches[hiddenColumnIndex] = maxMatch;
    hiddenFounds[hiddenColumnIndex] = found;
    hiddenCIs[hiddenColumnIndex] = maxIndex;
}

void ImageEncoder::learn(
    const Int2 &columnPos,
    const Array<const ByteBuffer*> &inputs
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellsStart = hiddenColumnIndex * hiddenSize.z;

    float match = hiddenMatches[hiddenColumnIndex];

    if (match < vigilance)
        return;

    // Check in radius
    for (int dx = -lRadius; dx <= lRadius; dx++)
        for (int dy = -lRadius; dy <= lRadius; dy++) {
            if (dx == 0 && dy == 0)
                continue;

            Int2 otherColumnPos(columnPos.x + dx, columnPos.y + dy);

            if (inBounds0(otherColumnPos, Int2(hiddenSize.x, hiddenSize.y))) {
                int otherHiddenColumnIndex = address2(otherColumnPos, Int2(hiddenSize.x, hiddenSize.y));

                if (hiddenMatches[otherHiddenColumnIndex] >= match)
                    return;
            }
        }

    int hiddenCellIndexMax = hiddenCIs[hiddenColumnIndex] + hiddenCellsStart;

    bool commit = (match > 0.0f && !hiddenFounds[hiddenColumnIndex]);

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

                int visibleCellsStart = visibleColumnIndex * vld.size.z;

                int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndexMax));

                if (commit) {
                    for (int vc = 0; vc < vld.size.z; vc++) {
                        int visibleCellIndex = vc + visibleCellsStart;

                        int input = (*inputs[vli])[visibleCellIndex];

                        int wi = vc + wiStart;

                        vl.weights0[wi] = input;
                        vl.weights1[wi] = 255 - input;
                    }
                }
                else {
                    for (int vc = 0; vc < vld.size.z; vc++) {
                        int visibleCellIndex = vc + visibleCellsStart;

                        int input = (*inputs[vli])[visibleCellIndex];

                        int wi = vc + wiStart;

                        vl.weights0[wi] = max(0, roundf(vl.weights0[wi] + lr * min(0, input - vl.weights0[wi])));
                        vl.weights1[wi] = max(0, roundf(vl.weights1[wi] + lr * min(0, 255 - input - vl.weights1[wi])));
                    }
                }
            }
    }

    if (commit && hiddenCommits[hiddenColumnIndex] < hiddenSize.z)
        hiddenCommits[hiddenColumnIndex]++;
}

void ImageEncoder::initRandom(
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

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights0.resize(numHiddenCells * area * vld.size.z);
        vl.weights1.resize(vl.weights0.size());
    }

    hiddenMatches = FloatBuffer(numHiddenColumns, 0.0f);
    hiddenFounds = ByteBuffer(numHiddenColumns, false);

    hiddenCIs = IntBuffer(numHiddenColumns, 0);

    hiddenCommits = IntBuffer(numHiddenColumns, 0);
}

void ImageEncoder::step(
    const Array<const ByteBuffer*> &inputs,
    bool learnEnabled
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    
    unsigned int baseState = rand();

    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++) {
        unsigned int state = baseState + i * 12345;

        activate(Int2(i / hiddenSize.y, i % hiddenSize.y), inputs, &state);
    }

    if (learnEnabled) {
        #pragma omp parallel for
        for (int i = 0; i < numHiddenColumns; i++)
            learn(Int2(i / hiddenSize.y, i % hiddenSize.y), inputs);
    }
}

int ImageEncoder::size() const {
    int size = sizeof(Int3) + 3 * sizeof(float) + sizeof(int) + 2 * hiddenCIs.size() * sizeof(int) + sizeof(int);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];

        size += sizeof(VisibleLayerDesc) + 2 * vl.weights0.size() * sizeof(Byte) +  sizeof(float);
    }

    return size;
}

int ImageEncoder::stateSize() const {
    return hiddenCIs.size() * sizeof(int);
}

void ImageEncoder::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&gap), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&vigilance), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&lRadius), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hiddenCommits[0]), hiddenCommits.size() * sizeof(int));

    int numVisibleLayers = visibleLayers.size();

    writer.write(reinterpret_cast<const void*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

        writer.write(reinterpret_cast<const void*>(&vl.weights0[0]), vl.weights0.size() * sizeof(Byte));
        writer.write(reinterpret_cast<const void*>(&vl.weights1[0]), vl.weights1.size() * sizeof(Byte));

        writer.write(reinterpret_cast<const void*>(&vl.importance), sizeof(float));
    }
}

void ImageEncoder::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&gap), sizeof(float));
    reader.read(reinterpret_cast<void*>(&vigilance), sizeof(float));
    reader.read(reinterpret_cast<void*>(&lr), sizeof(float));
    reader.read(reinterpret_cast<void*>(&lRadius), sizeof(int));

    hiddenMatches = FloatBuffer(numHiddenColumns, 0.0f);
    hiddenFounds = ByteBuffer(numHiddenColumns, false);

    hiddenCIs.resize(numHiddenColumns);
    hiddenCommits.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hiddenCommits[0]), hiddenCommits.size() * sizeof(int));

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

        vl.weights0.resize(numHiddenCells * area * vld.size.z);
        vl.weights1.resize(vl.weights0.size());

        reader.read(reinterpret_cast<void*>(&vl.weights0[0]), vl.weights0.size() * sizeof(Byte));
        reader.read(reinterpret_cast<void*>(&vl.weights1[0]), vl.weights1.size() * sizeof(Byte));

        reader.read(reinterpret_cast<void*>(&vl.importance), sizeof(float));
    }
}

void ImageEncoder::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
}

void ImageEncoder::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
}
