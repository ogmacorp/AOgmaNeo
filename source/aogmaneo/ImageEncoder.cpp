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
    const Array<const FloatBuffer*> &inputs,
    bool learnEnabled
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    int maxIndex = 0;
    int maxActivation = 0;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float sum = 0.0f;

        // For each visible layer
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];

            sum += vl.weights.multiply(*inputs[vli], hiddenIndex);
        }

        if (learnEnabled) {
            hiddenActivations[hiddenIndex].f = sum;
            hiddenActivations[hiddenIndex].i = hc;
        }

        if (sum > maxActivation) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    hiddenCs[hiddenColumnIndex] = maxIndex;

    if (learnEnabled) {
        int startIndex = address3(Int3(pos.x, pos.y, 0), hiddenSize);

        quicksort(hiddenActivations, startIndex, startIndex + hiddenSize.z);

        for (int hc = 0; hc < hiddenSize.z; hc++) {
            int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

            int sortedHiddenIndex = address3(Int3(pos.x, pos.y, hiddenActivations[hiddenIndex].i), hiddenSize);

            float strength = alpha * hiddenResources[sortedHiddenIndex] * expf(-gamma * hc / max(0.0001f, hiddenResources[sortedHiddenIndex]));

            // For each visible layer
            for (int vli = 0; vli < visibleLayers.size(); vli++) {
                VisibleLayer &vl = visibleLayers[vli];

                vl.weights.hebb(*inputs[vli], sortedHiddenIndex, strength);
            }

            hiddenResources[sortedHiddenIndex] *= gamma;
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

    // Find current max
    for (int vc = 0; vc < vld.size.z; vc++) {
        int visibleIndex = address3(Int3(pos.x, pos.y, vc), vld.size);

        vl.reconstruction[visibleIndex] = vl.weights.multiplyOHVsT(*reconCs, visibleIndex, hiddenSize.z) / vl.weights.countT(visibleIndex);
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

        vl.weights.initSMLocalRF(vld.size, hiddenSize, vld.radius);

        for (int i = 0; i < vl.weights.nonZeroValues.size(); i++)
            vl.weights.nonZeroValues[i] = randf();

        vl.reconstruction = FloatBuffer(numVisible, 0.0f);
    }

    hiddenActivations.resize(numHidden);

    hiddenCs = ByteBuffer(numHiddenColumns, 0);

    hiddenResources = FloatBuffer(numHidden, 1.0f);
}

void ImageEncoder::step(
    const Array<const FloatBuffer*> &inputs,
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