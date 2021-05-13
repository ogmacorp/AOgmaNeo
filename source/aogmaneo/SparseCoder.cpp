// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseCoder.h"

using namespace aon;

void SparseCoder::forwardClump(
    const Int2 &clumpPos,
    const Array<const IntBuffer*> &inputCIs,
    bool learnEnabled
) {
    int clumpIndex = address2(clumpPos, Int2(clumpTilingSize.x, clumpTilingSize.y));

    int columnsPerClump = clumpSize.x * clumpSize.y;

    int count = 0;

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        int diam = vld.radius * 2 + 1;

        // Projection
        Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(clumpTilingSize.x),
            static_cast<float>(vld.size.y) / static_cast<float>(clumpTilingSize.y));

        Int2 visibleCenter = project(clumpPos, hToV);

        // Lower corner
        Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

        // Bounds of receptive field, clamped to input size
        Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
        Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

        count += (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterUpperBound.y + 1);
    }

    int clumpMaxIndex = 0;
    float clumpMaxActivation = 0.0f;
    bool clumpMaxPassed = false;
    bool clumpMaxCommit = false;

    for (int c = 0; c < columnsPerClump; c++) {
        Int2 columnPos(clumpPos.x * clumpSize.x + c / clumpSize.y, clumpPos.y * clumpSize.y + c % clumpSize.y);

        int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

        int maxIndex = 0;
        int originalMaxIndex = 0;

        bool passed = false;
        bool commit = false;

        #pragma omp parallel for
        for (int hc = 0; hc < hiddenCommits[hiddenColumnIndex]; hc++) {
            int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

            int sum = 0;
            int total = 0;

            for (int vli = 0; vli < visibleLayers.size(); vli++) {
                VisibleLayer &vl = visibleLayers[vli];
                const VisibleLayerDesc &vld = visibleLayerDescs[vli];

                int diam = vld.radius * 2 + 1;
    
                // Projection
                Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(clumpTilingSize.x),
                    static_cast<float>(vld.size.y) / static_cast<float>(clumpTilingSize.y));

                Int2 visibleCenter = project(clumpPos, hToV);

                // Lower corner
                Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

                // Bounds of receptive field, clamped to input size
                Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
                Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

                for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                    for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                        int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                        Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                        int wi = offset.y + diam * (offset.x + diam * hiddenCellIndex);
                        
                        if ((*inputCIs[vli])[visibleColumnIndex] == vl.commitCIs[wi])
                            sum += vl.weights[wi];

                        total += vl.weights[wi];
                    }
            }

            hiddenActivations[hiddenCellIndex] = static_cast<float>(sum) / (static_cast<float>(total) + alpha * 255.0f);
            hiddenMatches[hiddenCellIndex] = static_cast<float>(sum) / static_cast<float>(count * 255);
        }

        float maxActivation = -1.0f;

        for (int hc = 0; hc < hiddenCommits[hiddenColumnIndex]; hc++) {
            int hiddenIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

            if (hiddenActivations[hiddenIndex] > maxActivation) {
                maxActivation = hiddenActivations[hiddenIndex];
                maxIndex = hc;
            }
        }

        originalMaxIndex = maxIndex;

        // Vigilance checking cycle
        for (int hc = 0; hc < hiddenCommits[hiddenColumnIndex]; hc++) {
            int hiddenCellIndexMax = address3(Int3(columnPos.x, columnPos.y, maxIndex), hiddenSize);

            if (hiddenMatches[hiddenCellIndexMax] < vigilance) { 
                // Reset
                hiddenActivations[hiddenCellIndexMax] = -1.0f;

                maxActivation = -1.0f;

                for (int ohc = 0; ohc < hiddenCommits[hiddenColumnIndex]; ohc++) {
                    int otherHiddenIndex = address3(Int3(columnPos.x, columnPos.y, ohc), hiddenSize);

                    if (hiddenActivations[otherHiddenIndex] > maxActivation) {
                        maxActivation = hiddenActivations[otherHiddenIndex];
                        maxIndex = ohc;
                    }
                }
            }
            else {
                passed = true;
                break;
            }
        }

        if (!passed) {
            if (learnEnabled && hiddenCommits[hiddenColumnIndex] < hiddenSize.z) {
                maxIndex = hiddenCommits[hiddenColumnIndex];
                hiddenCommits[hiddenColumnIndex]++;
                commit = true;
            }
            else
                maxIndex = originalMaxIndex;
        }

        hiddenCIs[hiddenColumnIndex] = maxIndex;

        int hiddenCellIndexMax = address3(Int3(columnPos.x, columnPos.y, maxIndex), hiddenSize);

        float rating = (commit ? 1.0f : 0.0f) + hiddenActivations[hiddenCellIndexMax];

        if (rating > clumpMaxActivation) {
            clumpMaxActivation = rating;
            clumpMaxIndex = c;
            clumpMaxPassed = passed;
            clumpMaxCommit = commit;
        }
    }

    bool doSlowLearn = learnEnabled && clumpMaxPassed;

    Int2 columnMaxPos(clumpPos.x * clumpSize.x + clumpMaxIndex / clumpSize.y, clumpPos.y * clumpSize.y + clumpMaxIndex % clumpSize.y);
    int hiddenColumnMaxIndex = address2(columnMaxPos, Int2(hiddenSize.x, hiddenSize.y));
    int hiddenCellIndexMax = address3(Int3(columnMaxPos.x, columnMaxPos.y, hiddenCIs[hiddenColumnMaxIndex]), hiddenSize);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        int diam = vld.radius * 2 + 1;

        // Projection
        Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(clumpTilingSize.x),
            static_cast<float>(vld.size.y) / static_cast<float>(clumpTilingSize.y));

        Int2 visibleCenter = project(clumpPos, hToV);

        // Lower corner
        Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

        // Bounds of receptive field, clamped to input size
        Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
        Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int wi = offset.y + diam * (offset.x + diam * hiddenCellIndexMax);

                int inCI = (*inputCIs[vli])[visibleColumnIndex];

                if (clumpMaxCommit) {
                    vl.commitCIs[wi] = inCI;
                    vl.weights[wi] = 255;
                }
                else if (doSlowLearn) {
                    if (vl.commitCIs[wi] != inCI) {
                        Byte weight = vl.weights[wi];

                        int delta = roundftoi(beta * -weight);
                        
                        vl.weights[wi] = max<int>(-delta, weight) + delta;
                    }
                }
            }
    }
}

void SparseCoder::initRandom(
    const Int3 &hiddenSize,
    const Int2 &clumpSize,
    const Array<VisibleLayerDesc> &visibleLayerDescs
) {
    this->visibleLayerDescs = visibleLayerDescs;

    this->hiddenSize = hiddenSize;
    this->clumpSize = clumpSize;

    clumpTilingSize = Int2(hiddenSize.x / clumpSize.x, hiddenSize.y / clumpSize.y);

    int numClumps = clumpTilingSize.x * clumpTilingSize.y;

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

        vl.weights.resize(numHiddenCells * area, 255);

        vl.commitCIs.resize(numHiddenCells * area, 0);
    }

    hiddenCommits = IntBuffer(numHiddenColumns, 0);

    hiddenActivations = FloatBuffer(numHiddenCells, 0.0f);
    hiddenMatches = FloatBuffer(numHiddenCells, 0.0f);

    // Hidden CIs
    hiddenCIs = IntBuffer(numHiddenColumns, 0);
}

void SparseCoder::step(
    const Array<const IntBuffer*> &inputCIs, // Input states
    bool learnEnabled // Whether to learn
) {
    int numClumps = clumpTilingSize.x * clumpTilingSize.y;

    #pragma omp parallel for
    for (int i = 0; i < numClumps; i++)
        forwardClump(Int2(i / clumpTilingSize.y, i % clumpTilingSize.y), inputCIs, learnEnabled);
}

int SparseCoder::size() const {
    int size = sizeof(Int3) + 2 * sizeof(Int2) + 3 * sizeof(float) + 2 * hiddenCIs.size() * sizeof(int) + sizeof(int);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        size += sizeof(VisibleLayerDesc) + vl.weights.size() * sizeof(Byte) + vl.commitCIs.size() * sizeof(int);
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
    writer.write(reinterpret_cast<const void*>(&clumpSize), sizeof(Int2));
    writer.write(reinterpret_cast<const void*>(&clumpTilingSize), sizeof(Int2));

    writer.write(reinterpret_cast<const void*>(&alpha), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&beta), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&vigilance), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hiddenCommits[0]), hiddenCommits.size() * sizeof(int));
    
    int numVisibleLayers = visibleLayers.size();

    writer.write(reinterpret_cast<const void*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

        writer.write(reinterpret_cast<const void*>(&vl.weights[0]), vl.weights.size() * sizeof(Byte));
        writer.write(reinterpret_cast<const void*>(&vl.commitCIs[0]), vl.commitCIs.size() * sizeof(int));
    }
}

void SparseCoder::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));
    reader.read(reinterpret_cast<void*>(&clumpSize), sizeof(Int2));
    reader.read(reinterpret_cast<void*>(&clumpTilingSize), sizeof(Int2));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&alpha), sizeof(float));
    reader.read(reinterpret_cast<void*>(&beta), sizeof(float));
    reader.read(reinterpret_cast<void*>(&vigilance), sizeof(float));

    hiddenCIs.resize(numHiddenColumns);
    hiddenCommits.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hiddenCommits[0]), hiddenCommits.size() * sizeof(int));

    hiddenActivations = FloatBuffer(numHiddenCells, 0.0f);
    hiddenMatches = FloatBuffer(numHiddenCells, 0.0f);

    int numVisibleLayers = visibleLayers.size();

    reader.read(reinterpret_cast<void*>(&numVisibleLayers), sizeof(int));

    visibleLayers.resize(numVisibleLayers);
    visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = visibleLayerDescs[vli];

        reader.read(reinterpret_cast<void*>(&vld), sizeof(VisibleLayerDesc));

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(numHiddenCells * area);
        vl.commitCIs.resize(vl.weights.size());

        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(Byte));
        reader.read(reinterpret_cast<void*>(&vl.commitCIs[0]), vl.commitCIs.size() * sizeof(int));
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
