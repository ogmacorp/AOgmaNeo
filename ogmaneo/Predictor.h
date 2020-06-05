// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Helpers.h"
#include <iostream>

namespace ogmaneo {
// Visible layer descriptor
struct PredictorVisibleLayerDesc {
    Int3 size; // Size of input

    int radius; // Radius onto input

    // Defaults
    PredictorVisibleLayerDesc()
    :
    size(4, 4, 16),
    radius(2)
    {}
};

// A prediction layer (predicts x_(t+1))
template <typename T>
class Predictor {
public:
    // Visible layer
    struct VisibleLayer {
        Array<T> weights;

        ByteBuffer inputCsPrev; // Previous timestep (prev) input states
    };

private:
    Int3 hiddenSize; // Size of the output/hidden/prediction

    ByteBuffer hiddenCs; // Hidden state

    // Visible layers and descs
    Array<VisibleLayer> visibleLayers;
    Array<PredictorVisibleLayerDesc> visibleLayerDescs;

    // --- Kernels ---

    void forward(
        const Int2 &pos,
        const Array<const ByteBuffer*> &inputCs
    ) {
        int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

        int maxIndex = 0;
        int maxActivation = 0;

        for (int hc = 0; hc < hiddenSize.z; hc++) {
            int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

            int sum = 0;

            // For each visible layer
            for (int vli = 0; vli < visibleLayers.size(); vli++) {
                VisibleLayer &vl = visibleLayers[vli];
                const PredictorVisibleLayerDesc &vld = visibleLayerDescs[vli];

                int diam = vld.radius * 2 + 1;
                int area = diam * diam;

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

                        T weight = vl.weights[offset.y + offset.x * diam + area * hiddenIndex];

                        unsigned char inC = (*inputCs[vli])[visibleColumnIndex];

                        sum += (weight & (1 << inC)) == 0 ? 0 : 1;
                    }
            }

            if (sum > maxActivation) {
                maxActivation = sum;
                maxIndex = hc;
            }
        }

        hiddenCs[hiddenColumnIndex] = maxIndex;
    }

    void learn(
        const Int2 &pos,
        const ByteBuffer* hiddenTargetCs,
        unsigned long* state
    ) {
        int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

        int targetC = (*hiddenTargetCs)[hiddenColumnIndex];

        int maxIndex = 0;
        float maxActivation = 0;

        for (int hc = 0; hc < hiddenSize.z; hc++) {
            int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

            int sum = 0;

            for (int vli = 0; vli < visibleLayers.size(); vli++) {
                VisibleLayer &vl = visibleLayers[vli];
                const PredictorVisibleLayerDesc &vld = visibleLayerDescs[vli];

                int diam = vld.radius * 2 + 1;
                int area = diam * diam;

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

                        T weight = vl.weights[offset.y + offset.x * diam + area * hiddenIndex];

                        unsigned char inC = vl.inputCsPrev[visibleColumnIndex];

                        sum += (weight & (1 << inC)) == 0 ? 0 : 1;
                    }
            }

            if (sum > maxActivation) {
                maxActivation = sum;
                maxIndex = hc;
            }
        }

        if (maxIndex != targetC) {
            for (int hc = 0; hc < hiddenSize.z; hc++) {
                int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

                int sum = 0;
                int count = 0;

                for (int vli = 0; vli < visibleLayers.size(); vli++) {
                    VisibleLayer &vl = visibleLayers[vli];
                    const PredictorVisibleLayerDesc &vld = visibleLayerDescs[vli];

                    int diam = vld.radius * 2 + 1;
                    int area = diam * diam;

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

                            T weight = vl.weights[offset.y + offset.x * diam + area * hiddenIndex];

                            unsigned char inC = vl.inputCsPrev[visibleColumnIndex];

                            sum += (weight & (1 << inC)) == 0 ? 0 : 1;
                            count++;
                        }
                }

                if (count > 0) {
                    float prob = static_cast<float>(sum) / static_cast<float>(count);
                    
                    if (hc == targetC) {
                        float probIncrease = alpha * (1.0f - prob);

                        for (int vli = 0; vli < visibleLayers.size(); vli++) {
                            VisibleLayer &vl = visibleLayers[vli];
                            const PredictorVisibleLayerDesc &vld = visibleLayerDescs[vli];

                            int diam = vld.radius * 2 + 1;
                            int area = diam * diam;

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
                                    if (randf(state) < probIncrease) {
                                        int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                                        Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                                        int wi = offset.y + offset.x * diam + area * hiddenIndex;

                                        T weight = vl.weights[wi];

                                        unsigned char inC = vl.inputCsPrev[visibleColumnIndex];

                                        vl.weights[wi] = weight | (1 << inC);
                                    }
                                }
                        }
                    }
                    else {
                        float probDecrease = alpha * prob;

                        for (int vli = 0; vli < visibleLayers.size(); vli++) {
                            VisibleLayer &vl = visibleLayers[vli];
                            const PredictorVisibleLayerDesc &vld = visibleLayerDescs[vli];

                            int diam = vld.radius * 2 + 1;
                            int area = diam * diam;

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
                                    if (randf(state) < probDecrease) {
                                        int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                                        Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                                        int wi = offset.y + offset.x * diam + area * hiddenIndex;

                                        T weight = vl.weights[wi];

                                        unsigned char inC = vl.inputCsPrev[visibleColumnIndex];

                                        vl.weights[wi] = weight & ~(1 << inC);
                                    }
                                }
                        }
                    }
                }
            }
        }
    }

public:
    float alpha; // Learning rate

    // Defaults
    Predictor()
    :
    alpha(1.0f)
    {}

    // Create with random initialization
    void initRandom(
        const Int3 &hiddenSize, // Hidden/output/prediction size
        const Array<PredictorVisibleLayerDesc> &visibleLayerDescs // First visible layer must be from current hidden state, second must be feed back state, rest can be whatever
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
            const PredictorVisibleLayerDesc &vld = this->visibleLayerDescs[vli];

            int numVisibleColumns = vld.size.x * vld.size.y;

            int diam = vld.radius * 2 + 1;
            int area = diam * diam;

            vl.weights.resize(numHidden * area);

            // Initialize to random values
            for (int i = 0; i < vl.weights.size(); i++)
                vl.weights[i] = randBits<T>();

            vl.inputCsPrev = ByteBuffer(numVisibleColumns, 0);
        }

        // Hidden Cs
        hiddenCs = ByteBuffer(numHiddenColumns, 0);
    }

    // Activate the predictor (predict values)
    void activate(
        const Array<const ByteBuffer*> &inputCs // Hidden/output/prediction size
    ) {
        // Forward kernel
        for (int x = 0; x < hiddenSize.x; x++)
            for (int y = 0; y < hiddenSize.y; y++)
                forward(Int2(x, y), inputCs);

        // Copy to prevs
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];

            vl.inputCsPrev = *inputCs[vli];
        }
    }

    // Learning predictions (update weights)
    void learn(
        const ByteBuffer* hiddenTargetCs
    ) {
        // Learn kernel
        for (int x = 0; x < hiddenSize.x; x++)
            for (int y = 0; y < hiddenSize.y; y++) {
                unsigned long state = rand();

                learn(Int2(x, y), hiddenTargetCs, &state);
            }
    }

    // Get number of visible layers
    int getNumVisibleLayers() const {
        return visibleLayers.size();
    }

    // Get a visible layer
    const VisibleLayer &getVisibleLayer(
        int i // Index of visible layer
    ) const {
        return visibleLayers[i];
    }

    // Get a visible layer descriptor
    const PredictorVisibleLayerDesc &getVisibleLayerDesc(
        int i // Index of visible layer
    ) const {
        return visibleLayerDescs[i];
    }

    // Get the hidden activations (predictions)
    const ByteBuffer &getHiddenCs() const {
        return hiddenCs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }
};
} // Namespace ogmaneo
