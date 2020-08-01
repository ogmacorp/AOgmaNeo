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
    const Int2 &pos,
    const Array<const ByteBuffer*> &inputs,
    bool learnEnabled
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

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
                for (int vc = 0; vc < vld.size.z; vc++) {
                    int visibleIndex = address3(Int3(ix, iy, vc), vld.size);

                    count += (*inputs[vli])[visibleIndex];
                }
            }
    }

    int maxIndex = 0;
    int originalMaxIndex = 0;
    int resets = 0;

    bool passed = false;
    bool commit = false;

    for (int hc = 0; hc < hiddenCommits[hiddenColumnIndex]; hc++) { // Start at one since we can skip the null input
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        int sum = 0;
        int total = 0;

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

                    int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex));
                    
                    for (int vc = 0; vc < vld.size.z; vc++) {
                        int visibleIndex = address3(Int3(ix, iy, vc), vld.size);

                        int wi = wiStart + vc;

                        sum += min<int>((*inputs[vli])[visibleIndex], vl.weights[wi]);
                        total += vl.weights[wi];
                    }
                }
        }

        hiddenActivations[hiddenIndex] = static_cast<float>(sum) / (static_cast<float>(total) + alpha * 255.0f);
        hiddenMatches[hiddenIndex] = static_cast<float>(sum) / static_cast<float>(count);
    }

    float maxActivation = -1.0f;

    for (int hc = 0; hc < hiddenCommits[hiddenColumnIndex]; hc++) { // Start at one since we can skip the null input
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        if (hiddenActivations[hiddenIndex] > maxActivation) {
            maxActivation = hiddenActivations[hiddenIndex];
            maxIndex = hc;
        }
    }

    originalMaxIndex = maxIndex;

    // Vigilance checking cycle
    for (int hc = 0; hc < hiddenCommits[hiddenColumnIndex]; hc++) { // Start at one since we can skip the null input
        int hiddenIndexMax = address3(Int3(pos.x, pos.y, maxIndex), hiddenSize);
        
        if (hiddenMatches[hiddenIndexMax] < hiddenVigilances[hiddenColumnIndex]) {
            resets++;
            
            // Reset
            hiddenActivations[hiddenIndexMax] = -1.0f;

            maxActivation = -1.0f;

            for (int ohc = 0; ohc < hiddenCommits[hiddenColumnIndex]; ohc++) { // Start at one since we can skip the null input
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
        if (learnEnabled && hiddenCommits[hiddenColumnIndex] < hiddenSize.z) {
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

    bool doSlowLearn = learnEnabled && passed;

    if (learnEnabled) {
        if (passed)
            hiddenVigilances[hiddenColumnIndex] = min(1.0f, (1.0f + sigma) * hiddenVigilances[hiddenColumnIndex]);
        else
            hiddenVigilances[hiddenColumnIndex] = (1.0f - sigma) * hiddenVigilances[hiddenColumnIndex];
    }

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

                int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndexMax));

                if (commit) {
                    for (int vc = 0; vc < vld.size.z; vc++) {
                        int visibleIndex = address3(Int3(ix, iy, vc), vld.size);

                        int wi = wiStart + vc;

                        vl.weights[wi] = (*inputs[vli])[visibleIndex];
                    }
                }
                else if (doSlowLearn) {
                    for (int vc = 0; vc < vld.size.z; vc++) {
                        int visibleIndex = address3(Int3(ix, iy, vc), vld.size);

                        int wi = wiStart + vc;

                        unsigned char weight = vl.weights[wi];

                        int delta = roundftoi(beta * (min<int>((*inputs[vli])[visibleIndex], weight) - weight));
                        
                        vl.weights[wi] = max<int>(-delta, weight) + delta;
                    }
                }
            }
    }
}

void ImageEncoder::reconstruct(
    const Int2 &pos,
    const ByteBuffer* reconCs,
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
                int hiddenIndex = address3(Int3(hiddenPos.x, hiddenPos.y, (*reconCs)[hiddenColumnIndex]), hiddenSize);

                Int2 visibleCenter = project(hiddenPos, hToV);

                if (inBounds(pos, Int2(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius), Int2(visibleCenter.x + vld.radius + 1, visibleCenter.y + vld.radius + 1))) {
                    Int2 offset(pos.x - visibleCenter.x + vld.radius, pos.y - visibleCenter.y + vld.radius);

                    unsigned char weight = vl.weights[vc + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex))];

                    sum += weight;
                    count++;
                }
            }

        vl.reconstruction[visibleIndex] = sum / max(1, count);
    }
}

void ImageEncoder::initRandom(
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

    // Create layers
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisible = numVisibleColumns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(numHidden * area * vld.size.z, 0);

        vl.reconstruction = ByteBuffer(numVisible, 0);
    }

    hiddenCommits = ByteBuffer(numHiddenColumns, 0);

    hiddenActivations = FloatBuffer(numHidden, 0.0f);
    hiddenMatches = FloatBuffer(numHidden, 0.0f);

    hiddenVigilances = FloatBuffer(numHiddenColumns, initVigilance);

    // Hidden Cs
    hiddenCs = ByteBuffer(numHiddenColumns, 0);
}

void ImageEncoder::step(
    const Array<const ByteBuffer*> &inputs,
    bool learnEnabled
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), inputs, learnEnabled);
}

void ImageEncoder::reconstruct(
    const ByteBuffer* reconCs
) {
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;

        #pragma omp parallel for
        for (int i = 0; i < numVisibleColumns; i++)
            reconstruct(Int2(i / vld.size.y, i % vld.size.y), reconCs, vli);
    }
}

void ImageEncoder::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&alpha), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&beta), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&sigma), sizeof(float));

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

        writer.write(reinterpret_cast<const void*>(&vl.weights[0]), vl.weights.size() * sizeof(unsigned char));
    }
}

void ImageEncoder::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden =  numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&alpha), sizeof(float));
    reader.read(reinterpret_cast<void*>(&beta), sizeof(float));
    reader.read(reinterpret_cast<void*>(&sigma), sizeof(float));

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

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisible = numVisibleColumns * vld.size.z;

        int weightsSize;

        reader.read(reinterpret_cast<void*>(&weightsSize), sizeof(int));

        vl.weights.resize(weightsSize);

        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(unsigned char));

        vl.reconstruction = ByteBuffer(numVisible, 0);
    }
}