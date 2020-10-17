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
    const Array<const IntBuffer*> &inputCs,
    const IntBuffer* hiddenPredictions,
    bool learnEnabled
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float sum = 0.0f;

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

                    int inC = (*inputCs[vli])[visibleColumnIndex];

                    sum += vl.weights[inC + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex))];
                }
        }

        if (sum > maxActivation || maxIndex == -1) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    float total = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        hiddenActivations[hiddenIndex] = expf(hiddenActivations[hiddenIndex] - maxActivation);

        total += hiddenActivations[hiddenIndex];
    }

    float scale = 1.0f / max(0.0001f, total);

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        hiddenActivations[hiddenIndex] *= scale;
    }

    hiddenCs[hiddenColumnIndex] = maxIndex;

    if (learnEnabled) {
        int predC = (*hiddenPredictions)[hiddenColumnIndex];

        for (int hc = 0; hc < hiddenSize.z; hc++) {
            int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

            float delta = alpha * ((hc == predC ? 1.0f : 0.0f) - hiddenActivations[hiddenIndex]);

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

                        int inC = (*inputCs[vli])[visibleColumnIndex];

                        vl.weights[inC + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex))] += delta;
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
            vl.weights[i] = randf(0.0f, 1.0f);
    }

    hiddenActivations = FloatBuffer(numHidden, 0.0f);

    // Hidden Cs
    hiddenCs = IntBuffer(numHiddenColumns, 0);
}

void SparseCoder::step(
    const Array<const IntBuffer*> &inputCs,
    const IntBuffer* hiddenPredictions,
    bool learnEnabled
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCs, hiddenPredictions, learnEnabled);
}

void SparseCoder::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&alpha), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenCs[0]), hiddenCs.size() * sizeof(int));
    
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
}

void SparseCoder::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden =  numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&alpha), sizeof(float));

    hiddenCs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCs[0]), hiddenCs.size() * sizeof(int));
    
    hiddenActivations = FloatBuffer(numHidden, 0.0f);

    int numVisibleLayers = visibleLayers.size();

    reader.read(reinterpret_cast<void*>(&numVisibleLayers), sizeof(int));

    visibleLayers.resize(numVisibleLayers);
    visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = visibleLayerDescs[vli];

        reader.read(reinterpret_cast<void*>(&vld), sizeof(VisibleLayerDesc));

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisible = numVisibleColumns * vld.size.z;

        int weightsSize;

        reader.read(reinterpret_cast<void*>(&weightsSize), sizeof(int));

        vl.weights.resize(weightsSize);

        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));
    }
}
