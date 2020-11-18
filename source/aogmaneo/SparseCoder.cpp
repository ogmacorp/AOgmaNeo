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
    const Int2 &columnPos,
    const Array<const IntBuffer*> &inputCIs,
    bool learnEnabled
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int maxIndex = 0;
    float maxActivation = -1.0f;
    int originalMaxIndex = 0;

    bool passed = false;
    bool commit = false;

    for (int hc = 0; hc < hiddenCommits[hiddenColumnIndex]; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

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

            Int2 visibleCenter = project(columnPos, hToV);

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

                    unsigned char normalizedInput = roundftoi(255.0f * static_cast<float>((*inputCIs[vli])[visibleColumnIndex]) / static_cast<float>(vld.size.z - 1));

                    sum += min<int>(normalizedInput, vl.weights0[wi]);
                    sum += min<int>(255 - normalizedInput, vl.weights1[wi]);

                    total += vl.weights0[wi] + vl.weights1[wi];
                    count += 255;
                }
        }

        hiddenActivations[hiddenCellIndex] = static_cast<float>(sum) / (static_cast<float>(total) + alpha * 255.0f);
        hiddenMatches[hiddenCellIndex] = static_cast<float>(sum) / static_cast<float>(max(1, count));

        if (hiddenActivations[hiddenCellIndex] > maxActivation) {
            maxActivation = hiddenActivations[hiddenCellIndex];
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

    bool doSlowLearn = learnEnabled && passed;

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
                int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int wi = offset.y + diam * (offset.x + diam * hiddenCellIndexMax);

                unsigned char normalizedInput = roundftoi(255.0f * static_cast<float>((*inputCIs[vli])[visibleColumnIndex]) / static_cast<float>(vld.size.z - 1));

                if (commit) {
                    vl.weights0[wi] = min<int>(vl.weights0[wi], normalizedInput);
                    vl.weights1[wi] = min<int>(vl.weights1[wi], 255 - normalizedInput);
                }
                else if (doSlowLearn) {
                    int delta0 = roundftoi(beta * (min<int>(normalizedInput, vl.weights0[wi]) - vl.weights0[wi]));
                    int delta1 = roundftoi(beta * (min<int>(255 - normalizedInput, vl.weights1[wi]) - vl.weights1[wi]));
                    
                    vl.weights0[wi] = max<int>(-delta0, vl.weights0[wi]) + delta0;
                    vl.weights1[wi] = max<int>(-delta1, vl.weights1[wi]) + delta1;
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
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    // Create layers
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisible = numVisibleColumns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights0.resize(numHiddenCells * area);
        vl.weights1.resize(vl.weights0.size());
        
        for (int i = 0; i < vl.weights0.size(); i++) {
            vl.weights0[i] = rand() % 256;
            vl.weights1[i] = rand() % 256;
        }
    }

    hiddenCommits = IntBuffer(numHiddenColumns, 0);

    hiddenActivations = FloatBuffer(numHiddenCells, 0.0f);
    hiddenMatches = FloatBuffer(numHiddenCells, 0.0f);

    // Hidden CIs
    hiddenCIs = IntBuffer(numHiddenColumns, 0);
}

void SparseCoder::step(
    const Array<const IntBuffer*> &inputCIs,
    bool learnEnabled
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCIs, learnEnabled);
}

void SparseCoder::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

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

        int weightsSize = vl.weights0.size();

        writer.write(reinterpret_cast<const void*>(&weightsSize), sizeof(int));

        writer.write(reinterpret_cast<const void*>(&vl.weights0[0]), vl.weights0.size() * sizeof(unsigned char));
        writer.write(reinterpret_cast<const void*>(&vl.weights1[0]), vl.weights1.size() * sizeof(unsigned char));
    }
}

void SparseCoder::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

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

        int weightsSize;

        reader.read(reinterpret_cast<void*>(&weightsSize), sizeof(int));

        vl.weights0.resize(weightsSize);
        vl.weights1.resize(weightsSize);

        reader.read(reinterpret_cast<void*>(&vl.weights0[0]), vl.weights0.size() * sizeof(unsigned char));
        reader.read(reinterpret_cast<void*>(&vl.weights1[0]), vl.weights1.size() * sizeof(unsigned char));
    }
}
