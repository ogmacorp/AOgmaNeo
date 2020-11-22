// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseCoder.h"
#include <iostream>

using namespace aon;

void SparseCoder::forwardClump(
    const Int2 &clumpPos,
    const Array<const IntBuffer*> &inputCIs,
    bool learnEnabled
) {
    int clumpIndex = address2(clumpPos, Int2(clumpTilingSize.x, clumpTilingSize.y));

    int columnsPerClump = clumpSize.x * clumpSize.y;

    // Iterate
    for (int it = 0; it < columnsPerClump; it++) {
        Int2 columnPos(clumpPos.x * clumpSize.x + it / clumpSize.y, clumpPos.y * clumpSize.y + it % clumpSize.y);

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

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    int cii = offset.y + diam * (offset.x + diam * clumpIndex);
                    
                    if (it == 0)
                        vl.clumpInputs[cii] = 255;

                    count += vl.clumpInputs[cii];
                }
        }

        int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

        if (count == 0) { // No input
            hiddenCIs[hiddenColumnIndex] = 0; // Set null
            continue;
        }

        int maxIndex = 1;
        int originalMaxIndex = 1;

        bool passed = false;
        bool commit = false;

        #pragma omp parallel for
        for (int hc = 1; hc < hiddenCommits[hiddenColumnIndex]; hc++) { // Start at one since we can skip the null
            int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc - 1), Int3(hiddenSize.x, hiddenSize.y, hiddenSize.z - 1)); // -1 since we don't store the null

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
                        int cii = offset.y + diam * (offset.x + diam * clumpIndex);
                        
                        if ((*inputCIs[vli])[visibleColumnIndex] == vl.commitCIs[wi])
                            sum += min<int>(vl.clumpInputs[cii], vl.weights[wi]);

                        total += vl.weights[wi];
                    }
            }

            hiddenActivations[hiddenCellIndex] = static_cast<float>(sum) / (static_cast<float>(total) + alpha * 255.0f);
            hiddenMatches[hiddenCellIndex] = static_cast<float>(sum) / static_cast<float>(count);
        }

        float maxActivation = -1.0f;

        for (int hc = 1; hc < hiddenCommits[hiddenColumnIndex]; hc++) { // Start at one since we can skip the null input
            int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc - 1), Int3(hiddenSize.x, hiddenSize.y, hiddenSize.z - 1)); // -1 since we don't store the null

            if (hiddenActivations[hiddenCellIndex] > maxActivation) {
                maxActivation = hiddenActivations[hiddenCellIndex];
                maxIndex = hc;
            }
        }

        originalMaxIndex = maxIndex;

        // Vigilance checking cycle
        for (int hc = 1; hc < hiddenCommits[hiddenColumnIndex]; hc++) { // Start at one since we can skip the null input
            int hiddenCellIndexMax = address3(Int3(columnPos.x, columnPos.y, maxIndex - 1), Int3(hiddenSize.x, hiddenSize.y, hiddenSize.z - 1)); // -1 since we don't store the null

            float effectiveVigilance = vigilance * (1.0f - static_cast<float>(maxIndex - 1) / static_cast<float>(hiddenSize.z - 1));

            if (hiddenMatches[hiddenCellIndexMax] < effectiveVigilance) { 
                // Reset
                hiddenActivations[hiddenCellIndexMax] = -1.0f;

                maxActivation = -1.0f;

                for (int ohc = 1; ohc < hiddenCommits[hiddenColumnIndex]; ohc++) { // Start at one since we can skip the null input
                    int otherHiddenIndex = address3(Int3(columnPos.x, columnPos.y, ohc - 1), Int3(hiddenSize.x, hiddenSize.y, hiddenSize.z - 1)); // -1 since we don't store the null

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

        // If passed, reduce clump inputs (and learn if that is enabled)
        int hiddenCellIndexMax = address3(Int3(columnPos.x, columnPos.y, maxIndex - 1), Int3(hiddenSize.x, hiddenSize.y, hiddenSize.z - 1)); // -1 since we don't store the null

        bool doSlowLearn = learnEnabled && passed;

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
                    int cii = offset.y + diam * (offset.x + diam * clumpIndex);

                    unsigned char inC = (*inputCIs[vli])[visibleColumnIndex];

                    if (commit) {
                        vl.commitCIs[wi] = inC;
                        vl.weights[wi] = vl.clumpInputs[cii];
                    }
                    else if (doSlowLearn) {
                        if (vl.commitCIs[wi] == inC) {
                            unsigned char weight = vl.weights[wi];

                            int delta = roundftoi(beta * (min<int>(vl.clumpInputs[cii], weight) - weight));
                            
                            vl.weights[wi] = max<int>(-delta, weight) + delta;
                        }
                        else {
                            unsigned char weight = vl.weights[wi];

                            int delta = roundftoi(beta * -weight);
                            
                            vl.weights[wi] = max<int>(-delta, weight) + delta;
                        }
                    }

                    // Reconstruct for next clump member column
                    if (vl.commitCIs[wi] == inC)
                        vl.clumpInputs[cii] = min<int>(static_cast<int>(vl.clumpInputs[cii]), 255 - static_cast<int>(vl.weights[wi]));
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
    int numHidden =  numHiddenColumns * hiddenSize.z;
    int numNonNullHidden =  numHiddenColumns * (hiddenSize.z - 1);

    // Create layers
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisible = numVisibleColumns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(numNonNullHidden * area, 0);

        vl.commitCIs.resize(numNonNullHidden * area, 0);

        vl.clumpInputs.resize(numClumps * area, 0);
    }

    hiddenCommits = IntBuffer(numHiddenColumns, 1); // 1 because 0 is null (no input) which is always committed

    hiddenActivations = FloatBuffer(numNonNullHidden, 0.0f);
    hiddenMatches = FloatBuffer(numNonNullHidden, 0.0f);

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

        int weightsSize = vl.weights.size();

        writer.write(reinterpret_cast<const void*>(&weightsSize), sizeof(int));

        int clumpInputSize = vl.clumpInputs.size();

        writer.write(reinterpret_cast<const void*>(&clumpInputSize), sizeof(int));

        writer.write(reinterpret_cast<const void*>(&vl.weights[0]), vl.weights.size() * sizeof(unsigned char));
        writer.write(reinterpret_cast<const void*>(&vl.commitCIs[0]), vl.commitCIs.size() * sizeof(int));
        writer.write(reinterpret_cast<const void*>(&vl.clumpInputs[0]), vl.clumpInputs.size() * sizeof(unsigned char));
    }
}

void SparseCoder::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));
    reader.read(reinterpret_cast<void*>(&clumpSize), sizeof(Int2));
    reader.read(reinterpret_cast<void*>(&clumpTilingSize), sizeof(Int2));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden =  numHiddenColumns * hiddenSize.z;
    int numNonNullHidden =  numHiddenColumns * (hiddenSize.z - 1);

    reader.read(reinterpret_cast<void*>(&alpha), sizeof(float));
    reader.read(reinterpret_cast<void*>(&beta), sizeof(float));
    reader.read(reinterpret_cast<void*>(&vigilance), sizeof(float));

    hiddenCIs.resize(numHiddenColumns);
    hiddenCommits.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hiddenCommits[0]), hiddenCommits.size() * sizeof(int));

    hiddenActivations = FloatBuffer(numNonNullHidden, 0.0f);
    hiddenMatches = FloatBuffer(numNonNullHidden, 0.0f);

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

        int clumpInputSize;

        reader.read(reinterpret_cast<void*>(&clumpInputSize), sizeof(int));

        vl.weights.resize(weightsSize);
        vl.commitCIs.resize(weightsSize);
        vl.clumpInputs.resize(clumpInputSize);

        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(unsigned char));
        reader.read(reinterpret_cast<void*>(&vl.commitCIs[0]), vl.commitCIs.size() * sizeof(int));
        reader.read(reinterpret_cast<void*>(&vl.clumpInputs[0]), vl.clumpInputs.size() * sizeof(unsigned char));
    }
}
