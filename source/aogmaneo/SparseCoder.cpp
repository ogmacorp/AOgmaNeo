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
    const Array<const ByteBuffer*> &inputCs
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    int maxIndex = 0;
    int maxActivation = 0;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        int sum = 0;
        int count = 0;

        // For each visible layer
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

                    unsigned char weight = vl.weights[inC + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex))];
                    
                    sum += weight;
                    count++;
                }
        }

        hiddenActivations[hiddenIndex] = hiddenStimuli[hiddenIndex] = static_cast<float>(sum) / static_cast<float>(max(1, count));

        if (sum > maxActivation) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    hiddenCs[hiddenColumnIndex] = maxIndex;
}

void SparseCoder::inhibit(
    const Int2 &pos
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    int maxIndex = 0;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        int sum = 0;
        int count = 0;

        int diam = lRadius * 2 + 1;

        // Lower corner
        Int2 fieldLowerBound(pos.x - lRadius, pos.y - lRadius);

        // Bounds of receptive field, clamped to input size
        Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
        Int2 iterUpperBound(min(hiddenSize.x - 1, pos.x + lRadius), min(hiddenSize.y - 1, pos.y + lRadius));

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                int otherHiddenColumnIndex = address2(Int2(ix, iy), Int2(hiddenSize.x,  hiddenSize.y));

                if (otherHiddenColumnIndex == hiddenColumnIndex) // No self-inhibition
                    continue;

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                unsigned char inC = hiddenCsTemp[otherHiddenColumnIndex];

                unsigned char weight = laterals[inC + hiddenSize.z * (offset.y + diam * (offset.x + diam * hiddenIndex))];
                
                sum += weight;
                count++;
            }

        float inhibition = static_cast<float>(sum) / static_cast<float>(max(1, count));

        hiddenActivations[hiddenIndex] += hiddenStimuli[hiddenIndex] - inhibition;

        if (hiddenActivations[hiddenIndex] > maxActivation) {
            maxActivation = hiddenActivations[hiddenIndex];
            maxIndex = hc;
        }
    }

    hiddenCs[hiddenColumnIndex] = maxIndex;
}

void SparseCoder::learn(
    const Int2 &pos,
    const Array<const ByteBuffer*> &inputCs
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenIndex = address3(Int3(pos.x, pos.y, hiddenCs[hiddenColumnIndex]), hiddenSize);

    // For each visible layer
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

                for (int vc = 0; vc < vld.size.z; vc++) {
                    int wi = vc + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex));

                    float weight = vl.weights[wi];

                    vl.weights[wi] = roundftoi(min(255.0f, max(0.0f, weight + hiddenResources[hiddenIndex] * ((vc == inC ? 255.0f : 0.0f) - weight))));
                }
            }
    }

    int diam = lRadius * 2 + 1;

    // Lower corner
    Int2 fieldLowerBound(pos.x - lRadius, pos.y - lRadius);

    // Bounds of receptive field, clamped to input size
    Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
    Int2 iterUpperBound(min(hiddenSize.x - 1, pos.x + lRadius), min(hiddenSize.y - 1, pos.y + lRadius));

    for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
        for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
            int otherHiddenColumnIndex = address2(Int2(ix, iy), Int2(hiddenSize.x,  hiddenSize.y));

            Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

            unsigned char inC = hiddenCs[otherHiddenColumnIndex];

            for (int ohc = 0; ohc < hiddenSize.z; ohc++) {
                int wi = ohc + hiddenSize.z * (offset.y + diam * (offset.x + diam * hiddenIndex));

                float weight = laterals[wi];

                laterals[wi] = roundftoi(min(255.0f, max(0.0f, weight + hiddenResources[hiddenIndex] * ((ohc == inC ? 255.0f : 0.0f) - weight))));
            }
        }

    hiddenResources[hiddenIndex] -= alpha * hiddenResources[hiddenIndex];
}

void SparseCoder::initRandom(
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
            vl.weights[i] = rand() % 255;

        vl.reconstruction = FloatBuffer(numVisible, 0);
    }

    hiddenStimuli = FloatBuffer(numHidden, 0.0f);
    hiddenActivations = FloatBuffer(numHidden, 0.0f);

    hiddenResources = FloatBuffer(numHidden, 0.5f);

    // Hidden Cs
    hiddenCs = ByteBuffer(numHiddenColumns, 0);
    hiddenCsTemp = ByteBuffer(numHiddenColumns, 0);

    int diam = lRadius * 2 + 1;
    int area = diam * diam;

    laterals.resize(numHidden * area * hiddenSize.z);

    // Initialize to random values
    char range = 16;

    for (int i = 0; i < laterals.size(); i++)
        laterals[i] = rand() % range;
}

// Activate the sparse coder (perform sparse coding)
void SparseCoder::step(
    const Array<const ByteBuffer*> &inputCs, // Input states
    bool learnEnabled // Whether to learn
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCs);

    for (int it = 0; it < explainIters; it++) {
        hiddenCsTemp = hiddenCs;

        #pragma omp parallel for
        for (int i = 0; i < numHiddenColumns; i++)
            inhibit(Int2(i / hiddenSize.y, i % hiddenSize.y));
    }

    if (learnEnabled) {
        #pragma omp parallel for
        for (int i = 0; i < numHiddenColumns; i++)
            learn(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCs);
    }
}