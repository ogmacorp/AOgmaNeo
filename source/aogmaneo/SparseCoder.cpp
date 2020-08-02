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
    const Int2 &pos,
    const Array<const ByteBuffer*> &inputCs,
    bool learnEnabled
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    unsigned long state = hiddenMaskSeeds[hiddenColumnIndex];

    int count = 0;

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        int diam = vld.radius * 2 + 1;

        // Projection
        Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
            static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));

        Int2 visibleCenter = project(pos, hToV);

        // Lower corner
        Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

        // Bounds of receptive field, clamped to input size
        Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
        Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int cii = offset.y + diam * (offset.x + diam * hiddenColumnIndex);
                
                // Check mask
                if (randf(&state) < 0.5f)
                    count++;
            }
    }

    int maxIndex = 1;
    int originalMaxIndex = 1;

    bool passed = false;
    bool commit = false;

    float maxActivation = -1.0f;

    for (int hc = 1; hc < hiddenCommits[hiddenColumnIndex]; hc++) { // Start at one since we can skip the null
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc - 1), Int3(hiddenSize.x, hiddenSize.y, hiddenSize.z - 1)); // -1 since we don't store the null

        int sum = 0;
        int total = 0;

        // Reset state
        state = hiddenMaskSeeds[hiddenColumnIndex];

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int diam = vld.radius * 2 + 1;

            // Projection
            Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
                static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));

            Int2 visibleCenter = project(pos, hToV);

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
                    
                    if (randf(&state) < 0.5f) {
                        if ((*inputCs[vli])[visibleColumnIndex] == vl.commitCs[wi])
                            sum += vl.weights[wi];

                        total += vl.weights[wi];
                    }
                }
        }

        hiddenActivations[hiddenIndex] = static_cast<float>(sum) / (static_cast<float>(total) + alpha * 255.0f);
        hiddenMatches[hiddenIndex] = static_cast<float>(sum) / static_cast<float>(count) / 255.0f;

        if (hiddenActivations[hiddenIndex] > maxActivation) {
            maxActivation = hiddenActivations[hiddenIndex];
            maxIndex = hc;
        }
    }

    originalMaxIndex = maxIndex;

    // Vigilance checking cycle
    for (int hc = 1; hc < hiddenCommits[hiddenColumnIndex]; hc++) { // Start at one since we can skip the null input
        int hiddenIndexMax = address3(Int3(pos.x, pos.y, maxIndex - 1), Int3(hiddenSize.x, hiddenSize.y, hiddenSize.z - 1)); // -1 since we don't store the null
        
        if (hiddenMatches[hiddenIndexMax] < hiddenVigilances[hiddenColumnIndex]) { 
            // Reset
            hiddenActivations[hiddenIndexMax] = -1.0f;

            maxActivation = -1.0f;

            for (int ohc = 1; ohc < hiddenCommits[hiddenColumnIndex]; ohc++) { // Start at one since we can skip the null input
                int hiddenIndex = address3(Int3(pos.x, pos.y, ohc - 1), Int3(hiddenSize.x, hiddenSize.y, hiddenSize.z - 1)); // -1 since we don't store the null

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

    if (!passed) {
        if (learnEnabled && hiddenCommits[hiddenColumnIndex] < hiddenSize.z) {
            maxIndex = hiddenCommits[hiddenColumnIndex];
            hiddenCommits[hiddenColumnIndex]++;
            commit = true;
        }
        else
            maxIndex = originalMaxIndex;
    }

    hiddenCs[hiddenColumnIndex] = maxIndex;

    if (learnEnabled) {
        int hiddenIndexMax = address3(Int3(pos.x, pos.y, maxIndex - 1), Int3(hiddenSize.x, hiddenSize.y, hiddenSize.z - 1)); // -1 since we don't store the null

        if (passed)
            hiddenVigilances[hiddenColumnIndex] = min(1.0f, (1.0f + sigma) * hiddenVigilances[hiddenColumnIndex]);
        else
            hiddenVigilances[hiddenColumnIndex] = (1.0f - sigma) * hiddenVigilances[hiddenColumnIndex];

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int diam = vld.radius * 2 + 1;

            // Projection
            Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
                static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));

            Int2 visibleCenter = project(pos, hToV);

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

                    unsigned char inC = (*inputCs[vli])[visibleColumnIndex];

                    if (commit) {
                        vl.commitCs[wi] = inC;
                        vl.weights[wi] = 255;
                    }
                    else if (passed) { // Do slow learn
                        if (vl.commitCs[wi] != inC) {
                            unsigned char weight = vl.weights[wi];

                            int delta = roundftoi(beta * -weight);
                            
                            vl.weights[wi] = max<int>(-delta, weight) + delta;
                        }
                    }
                }
        }
    }
}

void SparseCoder::initRandom(
    const Int3 &hiddenSize,
    float initVigilance,
    const Array<VisibleLayerDesc> &visibleLayerDescs
) {
    this->visibleLayerDescs = visibleLayerDescs;

    this->hiddenSize = hiddenSize;
    
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

        vl.commitCs.resize(numNonNullHidden * area, 0);
    }

    hiddenCommits = ByteBuffer(numHiddenColumns, 1); // 1 because 0 is null (no input) which is always committed

    hiddenMaskSeeds = ByteBuffer(numHiddenColumns);

    for (int i = 0; i < numHiddenColumns; i++)
        hiddenMaskSeeds[i] = rand() % 256;

    hiddenActivations = FloatBuffer(numNonNullHidden, 0.0f);
    hiddenMatches = FloatBuffer(numNonNullHidden, 0.0f);

    hiddenVigilances = FloatBuffer(numHiddenColumns, initVigilance);

    // Hidden Cs
    hiddenCs = ByteBuffer(numHiddenColumns, 0);
}

void SparseCoder::step(
    const Array<const ByteBuffer*> &inputCs, // Input states
    bool learnEnabled // Whether to learn
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCs, learnEnabled);
}

void SparseCoder::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&alpha), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&beta), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&sigma), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenCs[0]), hiddenCs.size() * sizeof(unsigned char));
    writer.write(reinterpret_cast<const void*>(&hiddenMaskSeeds[0]), hiddenMaskSeeds.size() * sizeof(unsigned char));
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

        writer.write(reinterpret_cast<const void*>(&vl.weights[0]), vl.weights.size() * sizeof(unsigned char));
        writer.write(reinterpret_cast<const void*>(&vl.commitCs[0]), vl.commitCs.size() * sizeof(unsigned char));
    }
}

void SparseCoder::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden =  numHiddenColumns * hiddenSize.z;
    int numNonNullHidden =  numHiddenColumns * (hiddenSize.z - 1);

    reader.read(reinterpret_cast<void*>(&alpha), sizeof(float));
    reader.read(reinterpret_cast<void*>(&beta), sizeof(float));
    reader.read(reinterpret_cast<void*>(&sigma), sizeof(float));

    hiddenCs.resize(numHiddenColumns);
    hiddenMaskSeeds.resize(numHiddenColumns);
    hiddenCommits.resize(numHiddenColumns);
    hiddenVigilances.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCs[0]), hiddenCs.size() * sizeof(unsigned char));
    reader.read(reinterpret_cast<void*>(&hiddenMaskSeeds[0]), hiddenMaskSeeds.size() * sizeof(unsigned char));
    reader.read(reinterpret_cast<void*>(&hiddenCommits[0]), hiddenCommits.size() * sizeof(unsigned char));
    reader.read(reinterpret_cast<void*>(&hiddenVigilances[0]), hiddenVigilances.size() * sizeof(float));

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

        vl.weights.resize(weightsSize);
        vl.commitCs.resize(weightsSize);

        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(unsigned char));
        reader.read(reinterpret_cast<void*>(&vl.commitCs[0]), vl.commitCs.size() * sizeof(unsigned char));
    }
}