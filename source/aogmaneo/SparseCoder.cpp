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

    int maxIndex = 0;
    float maxActivation = -1.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        if (hiddenCommits[hiddenIndex] == 0)
            continue;

        int sum = 0;
        int total = 0;
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
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    int start = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex));

                    unsigned char inC = (*inputCs[vli])[visibleColumnIndex];

                    for (int z = 0; z < vld.size.z; z++) {
                        if (vl.mask[z + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenColumnIndex))]) {
                            unsigned char weight0 = vl.weights[0 + 2 * (z + start)];
                            unsigned char weight1 = vl.weights[1 + 2 * (z + start)];
                        
                            total += weight0 + weight1;

                            if (z == inC)
                                sum += weight0;
                            else
                                sum += weight1;

                            count++;
                        }
                    }  
                }
        }

        hiddenActivations[hiddenIndex] = static_cast<float>(sum) / (static_cast<float>(total) + alpha * 255.0f);
        hiddenMatches[hiddenIndex] = static_cast<float>(sum) / static_cast<float>(max(1, count)) / 255.0f;

        if (hiddenActivations[hiddenIndex] > maxActivation) {
            maxActivation = hiddenActivations[hiddenIndex];
            maxIndex = hc;
        }
    }

    int originalMaxIndex = maxIndex;
    bool passed = false;

    // Vigilance checking cycle
    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndexMax = address3(Int3(pos.x, pos.y, maxIndex), hiddenSize);
        
        if (hiddenCommits[hiddenIndexMax] == 0)
            continue;

        if (hiddenMatches[hiddenIndexMax] < minVigilance) {
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

    if (!passed) {
        maxIndex = -1;

        // Search for uncommitted node
        for (int hc = 0; hc < hiddenSize.z; hc++) {
            int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

            if (hiddenCommits[hiddenIndex] == 0) {
                maxIndex = hc;
                break;
            }
        }

        if (maxIndex == -1)
            maxIndex = originalMaxIndex;
        else
            passed = true;
    }

    hiddenCs[hiddenColumnIndex] = maxIndex;

    if (learnEnabled && passed) {
        int hiddenIndexMax = address3(Int3(pos.x, pos.y, maxIndex), hiddenSize);

        float rate = hiddenCommits[hiddenIndexMax] == 0 ? 1.0f : beta;

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

                    unsigned char inC = (*inputCs[vli])[visibleColumnIndex];

                    int start = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndexMax));

                    for (int z = 0; z < vld.size.z; z++) {
                        int wi0 = 0 + 2 * (z + start);
                        int wi1 = 1 + 2 * (z + start);

                        unsigned char weight0 = vl.weights[wi0];
                        unsigned char weight1 = vl.weights[wi1];

                        if (z == inC) {
                            int delta1 = roundftoi(rate * -weight1);

                            vl.weights[wi1] = max<int>(-delta1, weight1) + delta1;
                        }
                        else {
                            int delta0 = roundftoi(rate * -weight0);

                            vl.weights[wi0] = max<int>(-delta0, weight0) + delta0;
                        }
                    }
                }
        }

        hiddenCommits[hiddenIndexMax] = 1;
    }
}

void SparseCoder::initRandom(
    const Int3 &hiddenSize,
    const Array<VisibleLayerDesc> &visibleLayerDescs
) {
    this->visibleLayerDescs = visibleLayerDescs;

    this->hiddenSize = hiddenSize;

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

        vl.weights.resize(numHidden * area * vld.size.z * 2);
        vl.mask.resize(numHiddenColumns * area * vld.size.z);

        // Initialize to random values
        for (int i = 0; i < vl.weights.size(); i++)
            vl.weights[i] = 255;

        for (int i = 0; i < vl.mask.size(); i++)
            vl.mask[i] = randf() < 0.5f ? 1 : 0;
    }

    hiddenCommits = ByteBuffer(numHidden, 0);

    hiddenActivations = FloatBuffer(numHidden, 0.0f);
    hiddenMatches = FloatBuffer(numHidden, 0.0f);

    // Hidden Cs
    hiddenCs = ByteBuffer(numHiddenColumns, 0);
}

// Activate the sparse coder (perform sparse coding)
void SparseCoder::step(
    const Array<const ByteBuffer*> &inputCs, // Input states
    bool learnEnabled // Whether to learn
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCs, learnEnabled);
}