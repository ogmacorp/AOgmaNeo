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
    float maxActivation = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

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
                    
                    for (int z = 0; z < vld.size.z; z++) {
                        unsigned char weight = vl.weights[z + start];
                    
                        total += weight;
                    }

                    unsigned char inC = (*inputCs[vli])[visibleColumnIndex];

                    unsigned char weight = vl.weights[inC + start];
                    
                    sum += weight;
                    count++;
                }
        }

        hiddenActivations[hiddenIndex] = static_cast<float>(sum) / (static_cast<float>(total) + alpha * 255);
        hiddenMatches[hiddenIndex] = static_cast<float>(sum) / static_cast<float>(max(1, count));

        if (hiddenActivations[hiddenIndex] > maxActivation) {
            maxActivation = hiddenActivations[hiddenIndex];
            maxIndex = hc;
        }
    }

    int originalMaxIndex = maxIndex;
    bool passed = true;

    // Vigilance checking cycle
    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndexMax = address3(Int3(pos.x, pos.y, maxIndex), hiddenSize);

        if (hiddenMatches[hiddenIndexMax] < minVigilance) {
            if (hc == hiddenSize.z - 1) {
                maxIndex = originalMaxIndex;
                passed = false;
                break;
            }
            else {
                // Reset
                hiddenActivations[hiddenIndexMax] = -1.0f;

                maxActivation = 0.0f;

                for (int hc = 0; hc < hiddenSize.z; hc++) {
                    int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

                    if (hiddenActivations[hiddenIndex] > maxActivation) {
                        maxActivation = hiddenActivations[hiddenIndex];
                        maxIndex = hc;
                    }
                }
            }
        }
        else
            break;
    }

    hiddenCs[hiddenColumnIndex] = maxIndex;

    if (learnEnabled && passed) {
        int hiddenIndexMax = address3(Int3(pos.x, pos.y, maxIndex), hiddenSize);

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
                        if (z == inC)
                            continue;

                        int wi = z + start;

                        unsigned char weight = vl.weights[wi];

                        int delta = roundftoi(beta * -vl.weights[wi]);

                        vl.weights[wi] = max<int>(-delta, weight) + delta;
                    }
                }
        }
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

        vl.weights.resize(numHidden * area * vld.size.z);

        // Initialize to random values
        for (int i = 0; i < vl.weights.size(); i++)
            vl.weights[i] = randf() < 0.5f ? 255 : 0;
    }

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