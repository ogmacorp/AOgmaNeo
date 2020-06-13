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

    int maxIndex = 0;
    int maxActivation = -999999;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        int sum = 0;

        // For each visible layer
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const ImageEncoderVisibleLayerDesc &vld = visibleLayerDescs[vli];

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
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    int start = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex));

                    for (int vc = 0; vc < vld.size.z; vc++) {
                        unsigned char input = (*inputs[vli])[address3(Int3(ix, iy, vc), vld.size)];

                        unsigned char weight = vl.weights[start + vc];

                        int delta = static_cast<int>(input) - static_cast<int>(weight);

                        sum += -delta * delta;
                    }
                }
        }

        hiddenActivations[hiddenIndex].a = sum;
        hiddenActivations[hiddenIndex].i = hiddenIndex;

        if (sum > maxActivation) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    hiddenCs[hiddenColumnIndex] = maxIndex;

    if (learnEnabled) {
        int startIndex = address3(Int3(pos.x, pos.y, 0), hiddenSize);

        quicksort<IntInt>(hiddenActivations, startIndex, startIndex + hiddenSize.z);

        for (int hc = 0; hc < hiddenSize.z; hc++) {
            int hiddenIndex = hiddenActivations[address3(Int3(pos.x, pos.y, hc), hiddenSize)].i;

            float dist = static_cast<float>(hiddenSize.z - 1 - hc) / static_cast<float>(hiddenSize.z);

            float strength = hiddenResources[hiddenIndex] * expf(-gamma * dist * dist / max(0.0001f, hiddenResources[hiddenIndex]));

            // For each visible layer
            for (int vli = 0; vli < visibleLayers.size(); vli++) {
                VisibleLayer &vl = visibleLayers[vli];
                const ImageEncoderVisibleLayerDesc &vld = visibleLayerDescs[vli];

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
                        int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                        Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                        int start = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex));

                        for (int vc = 0; vc < vld.size.z; vc++) {
                            unsigned char input = (*inputs[vli])[address3(Int3(ix, iy, vc), vld.size)];

                            unsigned char weight = vl.weights[start + vc];

                            vl.weights[start + vc] = roundftoi(min(255.0f, max(0.0f, vl.weights[start + vc] + strength * (static_cast<float>(input) - static_cast<float>(weight)))));
                        }
                    }
            }

            hiddenResources[hiddenIndex] -= alpha * strength;
        }
    }
}

void ImageEncoder::reconstruct(
    const Int2 &pos,
    const ByteBuffer* reconCs,
    int vli
) {
    VisibleLayer &vl = visibleLayers[vli];
    ImageEncoderVisibleLayerDesc &vld = visibleLayerDescs[vli];

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
    const Array<ImageEncoderVisibleLayerDesc> &visibleLayerDescs
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
        const ImageEncoderVisibleLayerDesc &vld = this->visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;
        int numVisible = numVisibleColumns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(numHidden * area * vld.size.z);

        // Initialize to random values
        for (int i = 0; i < vl.weights.size(); i++)
            vl.weights[i] = rand() % 255;

        vl.reconstruction = ByteBuffer(numVisible, 0);
    }

    hiddenActivations.resize(numHidden);

    // Hidden Cs
    hiddenCs = ByteBuffer(numHiddenColumns, 0);

    hiddenResources = FloatBuffer(numHidden, 1.0f);
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
        const ImageEncoderVisibleLayerDesc &vld = visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;

        #pragma omp parallel for
        for (int i = 0; i < numVisibleColumns; i++)
            reconstruct(Int2(i / vld.size.y, i % vld.size.y), reconCs, vli);
    }
}