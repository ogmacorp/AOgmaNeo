// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "ImageEncoder.h"

using namespace aon;

void ImageEncoder::forward(
    const Int2 &columnPos,
    const Array<const ByteBuffer*> &inputActs,
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

                    for (int vc = 0; vc < vld.size.z; vc++) {
                        int visibleCellIndex = address3(Int3(ix, iy, vc), vld.size);

                        int wi = vc + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                        int input = (*inputActs[vli])[visibleCellIndex];

                        sum += min<int>(input, vl.weights0[wi]);
                        sum += min<int>(255 - input, vl.weights1[wi]);

                        total += vl.weights0[wi] + vl.weights1[wi];
                        count += 255;
                    }
                }
        }

        hiddenActivations[hiddenCellIndex] = static_cast<float>(sum) / (static_cast<float>(total) + alpha * 255.0f);
        hiddenMatches[hiddenCellIndex] = static_cast<float>(sum) / static_cast<float>(max(255, count));

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

                for (int vc = 0; vc < vld.size.z; vc++) {
                    int visibleCellIndex = address3(Int3(ix, iy, vc), vld.size);

                    int wi = vc + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndexMax));

                    int input = (*inputActs[vli])[visibleCellIndex];

                    if (commit) {
                        vl.weights0[wi] = min<int>(vl.weights0[wi], input);
                        vl.weights1[wi] = min<int>(vl.weights1[wi], 255 - input);
                    }
                    else if (doSlowLearn) {
                        int delta0 = roundftoi(beta * (min<int>(input, vl.weights0[wi]) - vl.weights0[wi]));
                        int delta1 = roundftoi(beta * (min<int>(255 - input, vl.weights1[wi]) - vl.weights1[wi]));
                        
                        vl.weights0[wi] = max<int>(-delta0, vl.weights0[wi]) + delta0;
                        vl.weights1[wi] = max<int>(-delta1, vl.weights1[wi]) + delta1;
                    }
                }
            }
    }
}

void ImageEncoder::reconstruct(
    const Int2 &pos,
    const IntBuffer* reconCIs,
    int vli
) {
    VisibleLayer &vl = visibleLayers[vli];
    VisibleLayerDesc &vld = visibleLayerDescs[vli];

    int diam = vld.radius * 2 + 1;

    int visibleColumnIndex = address2(pos, Int2(vld.size.x, vld.size.y));

    // Projection
    Float2 vToH = Float2(static_cast<float>(hiddenSize.x) / static_cast<float>(vld.size.x),
        static_cast<float>(hiddenSize.y) / static_cast<float>(vld.size.y));

    Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
        static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));
                
    Int2 hiddenCenter = project(pos, vToH);

    Int2 reverseRadii(ceilf(vToH.x * vld.radius) + 1, ceilf(vToH.y * vld.radius) + 1);
    
    // Lower corner
    Int2 fieldLowerBound(hiddenCenter.x - reverseRadii.x, hiddenCenter.y - reverseRadii.y);

    // Bounds of receptive field, clamped to input size
    Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
    Int2 iterUpperBound(min(hiddenSize.x - 1, hiddenCenter.x + reverseRadii.x), min(hiddenSize.y - 1, hiddenCenter.y + reverseRadii.y));
    
    // Find current max
    for (int vc = 0; vc < vld.size.z; vc++) {
        int visibleIndex = address3(Int3(pos.x, pos.y, vc), vld.size);

        int sum = 0;
        int count = 0;

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                Int2 hiddenPos = Int2(ix, iy);

                int hiddenColumnIndex = address2(hiddenPos, Int2(hiddenSize.x, hiddenSize.y));
                int hiddenCellIndex = address3(Int3(hiddenPos.x, hiddenPos.y, (*reconCIs)[hiddenColumnIndex]), hiddenSize);

                Int2 visibleCenter = project(hiddenPos, hToV);

                if (inBounds(pos, Int2(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius), Int2(visibleCenter.x + vld.radius + 1, visibleCenter.y + vld.radius + 1))) {
                    Int2 offset(pos.x - visibleCenter.x + vld.radius, pos.y - visibleCenter.y + vld.radius);

                    int wi = vc + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                    sum += vl.weights0[wi];
                    sum += 255 - static_cast<int>(vl.weights1[wi]);
                    count += 2;
                }
            }

        vl.reconstruction[visibleIndex] = sum / max(1, count);
    }
}

void ImageEncoder::initRandom(
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
        int numVisibleCells = numVisibleColumns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights0.resize(numHiddenCells * area * vld.size.z);
        vl.weights1.resize(vl.weights0.size());
        
        for (int i = 0; i < vl.weights0.size(); i++) {
            vl.weights0[i] = 255;
            vl.weights1[i] = 255;
        }

        vl.reconstruction = ByteBuffer(numVisibleCells, 0);
    }

    hiddenCommits = IntBuffer(numHiddenColumns, 0);

    hiddenActivations = FloatBuffer(numHiddenCells, 0.0f);
    hiddenMatches = FloatBuffer(numHiddenCells, 0.0f);

    // Hidden CIs
    hiddenCIs = IntBuffer(numHiddenColumns, 0);
}

void ImageEncoder::step(
    const Array<const ByteBuffer*> &inputActs,
    bool learnEnabled
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), inputActs, learnEnabled);
}

void ImageEncoder::reconstruct(
    const IntBuffer* reconCIs
) {
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;

        #pragma omp parallel for
        for (int i = 0; i < numVisibleColumns; i++)
            reconstruct(Int2(i / vld.size.y, i % vld.size.y), reconCIs, vli);
    }
}

void ImageEncoder::write(
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

        writer.write(reinterpret_cast<const void*>(&vl.reconstruction[0]), vl.reconstruction.size() * sizeof(unsigned char));
    }
}

void ImageEncoder::read(
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

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisibleCells = numVisibleColumns * vld.size.z;

        int weightsSize;

        reader.read(reinterpret_cast<void*>(&weightsSize), sizeof(int));

        vl.weights0.resize(weightsSize);
        vl.weights1.resize(weightsSize);

        reader.read(reinterpret_cast<void*>(&vl.weights0[0]), vl.weights0.size() * sizeof(unsigned char));
        reader.read(reinterpret_cast<void*>(&vl.weights1[0]), vl.weights1.size() * sizeof(unsigned char));

        vl.reconstruction.resize(numVisibleCells);

        reader.read(reinterpret_cast<void*>(&vl.reconstruction[0]), vl.reconstruction.size() * sizeof(unsigned char));
    }
}
