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

        hiddenActivations[hiddenIndex] = hiddenStimuli[hiddenIndex] = static_cast<float>(sum) / static_cast<float>(max(1, count)) / 255.0f;

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

        float inhibition = static_cast<float>(sum) / static_cast<float>(max(1, count)) / 255.0f;

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

                    vl.weights[wi] = roundftoi(min(255.0f, max(0.0f, weight + alpha * ((vc == inC ? 255.0f : 0.0f) - weight))));
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

                laterals[wi] = roundftoi(min(255.0f, max(0.0f, weight + alpha * ((ohc == inC ? 255.0f : 0.0f) - weight))));
            }
        }
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

        for (int i = 0; i < vl.weights.size(); i++)
            vl.weights[i] = rand() % 256;

        vl.reconstruction = FloatBuffer(numVisible, 0);
    }

    hiddenStimuli = FloatBuffer(numHidden, 0.0f);
    hiddenActivations = FloatBuffer(numHidden, 0.0f);

    // Hidden Cs
    hiddenCs = ByteBuffer(numHiddenColumns, 0);
    hiddenCsTemp = ByteBuffer(numHiddenColumns, 0);

    int diam = lRadius * 2 + 1;
    int area = diam * diam;

    laterals.resize(numHidden * area * hiddenSize.z);

    for (int i = 0; i < laterals.size(); i++)
        laterals[i] = rand() % 16;
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

void SparseCoder::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));
    writer.write(reinterpret_cast<const void*>(&lRadius), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&alpha), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&explainIters), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&hiddenCs[0]), hiddenCs.size() * sizeof(unsigned char));

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

    int lateralsSize = laterals.size();

    writer.write(reinterpret_cast<const void*>(&lateralsSize), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&laterals[0]),laterals.size() * sizeof(unsigned char));
}

void SparseCoder::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));
    reader.read(reinterpret_cast<void*>(&lRadius), sizeof(int));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden =  numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&alpha), sizeof(float));
    reader.read(reinterpret_cast<void*>(&explainIters), sizeof(int));

    hiddenCs.resize(numHiddenColumns);
    hiddenCsTemp.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCs[0]), hiddenCs.size() * sizeof(unsigned char));

    hiddenStimuli = FloatBuffer(numHidden, 0.0f);
    hiddenActivations = FloatBuffer(numHidden, 0.0f);

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

        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(unsigned char));
    }

    int lateralsSize;

    reader.read(reinterpret_cast<void*>(&lateralsSize), sizeof(int));

    laterals.resize(lateralsSize);

    reader.read(reinterpret_cast<void*>(&laterals[0]), laterals.size() * sizeof(unsigned char));
}