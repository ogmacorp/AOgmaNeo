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
    const Array<const ByteBuffer*> &inputCs,
    bool learnEnabled
) {
    int clumpIndex = address2(clumpPos, Int2(clumpTilingSize.x, clumpTilingSize.y));

    int columnsPerClump = clumpSize.x * clumpSize.y;

    // Iterate
    for (int it = 0; it < columnsPerClump; it++) {
        Int2 pos(clumpPos.x * clumpSize.x + it % clumpSize.x, clumpPos.y * clumpSize.y + it / clumpSize.x);

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

        bool hasInput = count > 0;

        int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

        int maxIndex = 0;
        int originalMaxIndex = 0;
        int resets = 0;

        bool passed = false;
        bool commit = false;

        if (hasInput) {
            #pragma omp parallel for
            for (int hc = 0; hc < hiddenCommits[hiddenColumnIndex]; hc++) {
                int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

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

                            int wi = offset.y + diam * (offset.x + diam * hiddenIndex);
                            int cii = offset.y + diam * (offset.x + diam * clumpIndex);
                            
                            if ((*inputCs[vli])[visibleColumnIndex] == vl.commitCs[wi])
                                sum += min<int>(vl.clumpInputs[cii], vl.weights[wi]);

                            total += vl.weights[wi];
                        }
                }

                hiddenActivations[hiddenIndex] = static_cast<float>(sum) / (static_cast<float>(total) + alpha * 255.0f);
                hiddenMatches[hiddenIndex] = static_cast<float>(sum) / static_cast<float>(count);
            }

            float maxActivation = -1.0f;

            for (int hc = 0; hc < hiddenCommits[hiddenColumnIndex]; hc++) {
                int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

                if (hiddenActivations[hiddenIndex] > maxActivation) {
                    maxActivation = hiddenActivations[hiddenIndex];
                    maxIndex = hc;
                }
            }

            originalMaxIndex = maxIndex;

            // Vigilance checking cycle
            for (int hc = 0; hc < hiddenCommits[hiddenColumnIndex]; hc++) {
                int hiddenIndexMax = address3(Int3(pos.x, pos.y, maxIndex), hiddenSize);
                
                if (hiddenMatches[hiddenIndexMax] < hiddenVigilances[hiddenColumnIndex]) {
                    resets++;
                    
                    // Reset
                    hiddenActivations[hiddenIndexMax] = -1.0f;

                    maxActivation = -1.0f;

                    for (int ohc = 0; ohc < hiddenSize.z; ohc++) {
                        int hiddenIndex = address3(Int3(pos.x, pos.y, ohc), hiddenSize);

                        if (hiddenActivations[hiddenIndex] > maxActivation) {
                            maxActivation = hiddenActivations[hiddenIndex];
                            maxIndex = ohc;
                        }
                    }
                }
                else {
                    passed = true;
                    break;
                }
            }
        }

        if (!passed) {
            if (learnEnabled && hasInput && hiddenCommits[hiddenColumnIndex] < hiddenSize.z) {
                maxIndex = hiddenCommits[hiddenColumnIndex];
                hiddenCommits[hiddenColumnIndex]++;
                commit = true;
            }
            else
                maxIndex = originalMaxIndex;
        }

        hiddenCs[hiddenColumnIndex] = maxIndex;

        // If passed, reduce clump inputs (and learn if that is enabled)
        int hiddenIndexMax = address3(Int3(pos.x, pos.y, maxIndex), hiddenSize);

        bool doSlowLearn = learnEnabled && hasInput && passed;

        if (learnEnabled && !commit)
            hiddenVigilances[hiddenColumnIndex] = min(1.0f, max(0.0f, hiddenVigilances[hiddenColumnIndex] + gamma * (targetResets - resets)));
        
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

                    int wi = offset.y + diam * (offset.x + diam * hiddenIndexMax);
                    int cii = offset.y + diam * (offset.x + diam * clumpIndex);

                    unsigned char inC = (*inputCs[vli])[visibleColumnIndex];

                    if (commit) {
                        vl.commitCs[wi] = inC;
                        vl.weights[wi] = 255;
                    }
                    else if (doSlowLearn) {
                        if (vl.commitCs[wi] == inC) {
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
                    if (vl.commitCs[wi] == inC)
                        vl.clumpInputs[cii] = max<int>(0, vl.clumpInputs[cii] - vl.weights[wi]);
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

    // Create layers
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisible = numVisibleColumns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(numHidden * area, 0);

        vl.commitCs.resize(numHidden * area, 0);

        vl.clumpInputs.resize(numClumps * area, 0);
    }

    hiddenCommits = ByteBuffer(numHiddenColumns, 0);

    hiddenActivations = FloatBuffer(numHidden, 0.0f);
    hiddenMatches = FloatBuffer(numHidden, 0.0f);

    // Hidden Cs
    hiddenCs = ByteBuffer(numHiddenColumns, 0);

    hiddenVigilances = FloatBuffer(numHiddenColumns, 0.0f);
}

void SparseCoder::step(
    const Array<const ByteBuffer*> &inputCs, // Input states
    bool learnEnabled // Whether to learn
) {
    int numClumps = clumpTilingSize.x * clumpTilingSize.y;

    #pragma omp parallel for
    for (int i = 0; i < numClumps; i++)
        forwardClump(Int2(i / clumpTilingSize.y, i % clumpTilingSize.y), inputCs, learnEnabled);
}

void SparseCoder::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));
    writer.write(reinterpret_cast<const void*>(&clumpSize), sizeof(Int2));
    writer.write(reinterpret_cast<const void*>(&clumpTilingSize), sizeof(Int2));

    writer.write(reinterpret_cast<const void*>(&alpha), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&beta), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&targetResets), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&gamma), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenCs[0]), hiddenCs.size() * sizeof(unsigned char));
    writer.write(reinterpret_cast<const void*>(&hiddenCommits[0]), hiddenCommits.size() * sizeof(unsigned char));
    writer.write(reinterpret_cast<const void*>(&hiddenVigilances[0]), hiddenVigilances.size() * sizeof(float));
    
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
        writer.write(reinterpret_cast<const void*>(&vl.commitCs[0]), vl.commitCs.size() * sizeof(unsigned char));
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

    reader.read(reinterpret_cast<void*>(&alpha), sizeof(float));
    reader.read(reinterpret_cast<void*>(&beta), sizeof(float));
    reader.read(reinterpret_cast<void*>(&targetResets), sizeof(float));
    reader.read(reinterpret_cast<void*>(&gamma), sizeof(float));

    hiddenCs.resize(numHiddenColumns);
    hiddenCommits.resize(numHiddenColumns);
    hiddenVigilances.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCs[0]), hiddenCs.size() * sizeof(unsigned char));
    reader.read(reinterpret_cast<void*>(&hiddenCommits[0]), hiddenCommits.size() * sizeof(unsigned char));
    reader.read(reinterpret_cast<void*>(&hiddenVigilances[0]), hiddenVigilances.size() * sizeof(float));

    hiddenActivations = FloatBuffer(numHidden, 0.0f);
    hiddenMatches = FloatBuffer(numHidden, 0.0f);

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
        vl.commitCs.resize(weightsSize);
        vl.clumpInputs.resize(clumpInputSize);

        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(unsigned char));
        reader.read(reinterpret_cast<void*>(&vl.commitCs[0]), vl.commitCs.size() * sizeof(unsigned char));
        reader.read(reinterpret_cast<void*>(&vl.clumpInputs[0]), vl.clumpInputs.size() * sizeof(unsigned char));
    }
}