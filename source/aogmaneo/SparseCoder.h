// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Helpers.h"

namespace aon {
// Visible layer descriptor
struct SparseCoderVisibleLayerDesc {
    Int3 size; // Size of input

    int radius; // Radius onto input

    // Defaults
    SparseCoderVisibleLayerDesc()
    :
    size(4, 4, 16),
    radius(2)
    {}
};

// Sparse coder
template <typename T>
class SparseCoder {
public:
    // Visible layer
    struct VisibleLayer {
        Array<T> weights; // Binary weight matrix

        FloatBuffer visibleProbs; // Temporary storage for probabilties
    };

private:
    Int3 hiddenSize; // Size of hidden/output layer

    ByteBuffer hiddenCs; // Hidden states
    ByteBuffer hiddenCsPrev; // Previous hidden states

    // Visible layers and associated descriptors
    Array<VisibleLayer> visibleLayers;
    Array<SparseCoderVisibleLayerDesc> visibleLayerDescs;
    
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
                const SparseCoderVisibleLayerDesc &vld = visibleLayerDescs[vli];

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
        const ByteBuffer* inputCs,
        int vli,
        unsigned long* state
    ) {
        VisibleLayer &vl = visibleLayers[vli];
        SparseCoderVisibleLayerDesc &vld = visibleLayerDescs[vli];

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

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
        
        unsigned char targetC = (*inputCs)[visibleColumnIndex];

        int maxIndex = 0;
        int maxActivation = 0;

        // Find current max
        for (int vc = 0; vc < vld.size.z; vc++) {
            int visibleIndex = address3(Int3(pos.x, pos.y, vc), vld.size);

            int sum = 0;
            int count = 0;

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    Int2 hiddenPos = Int2(ix, iy);

                    int hiddenColumnIndex = address2(hiddenPos, Int2(hiddenSize.x, hiddenSize.y));
                    int hiddenIndex = address3(Int3(hiddenPos.x, hiddenPos.y, hiddenCs[hiddenColumnIndex]), hiddenSize);

                    Int2 visibleCenter = project(hiddenPos, hToV);

                    if (inBounds(pos, Int2(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius), Int2(visibleCenter.x + vld.radius + 1, visibleCenter.y + vld.radius + 1))) {
                        Int2 offset(pos.x - visibleCenter.x + vld.radius, pos.y - visibleCenter.y + vld.radius);

                        T weight = vl.weights[offset.y + offset.x * diam + area * hiddenIndex];

                        sum += (weight & (1 << vc)) == 0 ? 0 : 1;
                        count++;
                    }
                }

            vl.visibleProbs[visibleIndex] = static_cast<float>(sum) / static_cast<float>(count);

            if (sum > maxActivation) {
                maxActivation = sum;
                maxIndex = vc;
            }
        }

        if (maxIndex != targetC) {
            for (int vc = 0; vc < vld.size.z; vc++) {
                int visibleIndex = address3(Int3(pos.x, pos.y, vc), vld.size);

                float prob = vl.visibleProbs[visibleIndex];

                T (*update)(T, unsigned char) = vc == targetC ? &increase<T> : &decrease<T>;

                float updateProb = (vc == targetC ? alpha * (1.0f - prob) : alpha * prob);
                
                for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                    for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                        Int2 hiddenPos = Int2(ix, iy);

                        int hiddenColumnIndex = address2(hiddenPos, Int2(hiddenSize.x, hiddenSize.y));
                        int hiddenIndex = address3(Int3(hiddenPos.x, hiddenPos.y, hiddenCs[hiddenColumnIndex]), hiddenSize);

                        Int2 visibleCenter = project(hiddenPos, hToV);

                        if (inBounds(pos, Int2(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius), Int2(visibleCenter.x + vld.radius + 1, visibleCenter.y + vld.radius + 1))) {
                            if (randf(state) < updateProb) {
                                Int2 offset(pos.x - visibleCenter.x + vld.radius, pos.y - visibleCenter.y + vld.radius);

                                int wi = offset.y + offset.x * diam + area * hiddenIndex;
                            
                                T weight = vl.weights[wi];
                            
                                vl.weights[wi] = (*update)(weight, vc);
                            }
                        }
                    }
            }
        }
    }

public:
    float alpha; // Learning rate

    // Defaults
    SparseCoder()
    :
    alpha(0.3f)
    {}

    // Create a sparse coding layer with random initialization
    void initRandom(
        const Int3 &hiddenSize, // Hidden/output size
        const Array<SparseCoderVisibleLayerDesc> &visibleLayerDescs // Descriptors for visible layers
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
            const SparseCoderVisibleLayerDesc &vld = this->visibleLayerDescs[vli];

            int numVisibleColumns = vld.size.x * vld.size.y;
            int numVisible = numVisibleColumns * vld.size.z;

            int diam = vld.radius * 2 + 1;
            int area = diam * diam;

            vl.weights.resize(numHidden * area);

            // Initialize to random values
            for (int i = 0; i < vl.weights.size(); i++)
                vl.weights[i] = randBits<T>();

            vl.visibleProbs = FloatBuffer(numVisible, 0.0f);
        }

        // Hidden Cs
        hiddenCs = ByteBuffer(numHiddenColumns, 0);
        hiddenCsPrev = ByteBuffer(numHiddenColumns, 0);
    }

    // Activate the sparse coder (perform sparse coding)
    void step(
        const Array<const ByteBuffer*> &inputCs, // Input states
        bool learnEnabled // Whether to learn
    ) {
        int numHiddenColumns = hiddenSize.x * hiddenSize.y;
        
        #pragma omp parallel for
        for (int i = 0; i < numHiddenColumns; i++)
            forward(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCs);

        if (learnEnabled) {
            for (int vli = 0; vli < visibleLayers.size(); vli++) {
                const SparseCoderVisibleLayerDesc &vld = visibleLayerDescs[vli];

                int numVisibleColumns = vld.size.x * vld.size.y;
            
                unsigned long baseState = rand();

                #pragma omp parallel for
                for (int i = 0; i < numVisibleColumns; i++) {
                    unsigned long state = baseState + i;

                    learn(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCs[vli], vli, &state);
                }
            }
        }

        hiddenCsPrev = hiddenCs;
    }

    // Get the number of visible layers
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
    const SparseCoderVisibleLayerDesc &getVisibleLayerDesc(
        int i // Index of visible layer
    ) const {
        return visibleLayerDescs[i];
    }

    // Get the hidden states
    const ByteBuffer &getHiddenCs() const {
        return hiddenCs;
    }

    // Get the previous tick hidden states
    const ByteBuffer &getHiddenCsPrev() const {
        return hiddenCsPrev;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }
};
} // namespace aon
