// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Helpers.h"
#include "SparseMatrix.h"

namespace aon {
// Visible layer descriptor
struct ActorVisibleLayerDesc {
    Int3 size; // Visible/input size

    int radius; // Radius onto input

    // Defaults
    ActorVisibleLayerDesc()
    :
    size(4, 4, 16),
    radius(2)
    {}
};

// A reinforcement learning layer
template <typename T>
class Actor {
public:
    // Visible layer
    struct VisibleLayer {
        SparseMatrix valueWeights; // Value function weights
        Array<T> actionWeights; // Action function weights
    };

    // History sample for delayed updates
    struct HistorySample {
        Array<ByteBuffer> inputCs;
        ByteBuffer hiddenTargetCsPrev;

        FloatBuffer hiddenValuesPrev;
        
        float reward;
    };

private:
    Int3 hiddenSize; // Hidden/output/action size

    // Current history size - fixed after initialization. Determines length of wait before updating
    int historySize;

    ByteBuffer hiddenCs; // Hidden states

    FloatBuffer hiddenValues; // Hidden value function output buffer

    FloatBuffer hiddenProbs; // Temporary storage for probabilties

    CircleBuffer<HistorySample> historySamples; // History buffer, fixed length

    // Visible layers and descriptors
    Array<VisibleLayer> visibleLayers;
    Array<ActorVisibleLayerDesc> visibleLayerDescs;

    // --- Kernels ---

    void forward(
        const Int2 &pos,
        const Array<const ByteBuffer*> &inputCs
    ) {
        int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

        // --- Value ---

        float value = 0.0f;
        int count = 0;

        // For each visible layer
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const ActorVisibleLayerDesc &vld = visibleLayerDescs[vli];

            value += vl.valueWeights.multiplyOHVs(*inputCs[vli], hiddenColumnIndex, vld.size.z);
            count += vl.valueWeights.count(hiddenColumnIndex) / vld.size.z;
        }

        hiddenValues[hiddenColumnIndex] = value / count;

        // --- Action ---

        int maxIndex = 0;
        int maxActivation = 0;

        for (int hc = 0; hc < hiddenSize.z; hc++) {
            int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

            int sum = 0;

            // For each visible layer
            for (int vli = 0; vli < visibleLayers.size(); vli++) {
                VisibleLayer &vl = visibleLayers[vli];
                const ActorVisibleLayerDesc &vld = visibleLayerDescs[vli];

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

                        T weight = vl.actionWeights[offset.y + diam * (offset.x + diam * hiddenIndex)];

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
        const Array<const ByteBuffer*> &inputCsPrev,
        const ByteBuffer* hiddenTargetCsPrev,
        const FloatBuffer* hiddenValuesPrev,
        float q,
        float g,
        bool mimic,
        unsigned long* state
    ) {
        int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

        // --- Value Prev ---

        float newValue = q + g * hiddenValues[hiddenColumnIndex];

        float value = 0.0f;
        int count = 0;

        // For each visible layer
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const ActorVisibleLayerDesc &vld = visibleLayerDescs[vli];

            value += vl.valueWeights.multiplyOHVs(*inputCsPrev[vli], hiddenColumnIndex, vld.size.z);
            count += vl.valueWeights.count(hiddenColumnIndex) / vld.size.z;
        }

        value /= count;

        float tdErrorValue = newValue - value;
        
        float deltaValue = alpha * tdErrorValue;

        // For each visible layer
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const ActorVisibleLayerDesc &vld = visibleLayerDescs[vli];

            vl.valueWeights.deltaOHVs(*inputCsPrev[vli], deltaValue, hiddenColumnIndex, vld.size.z);
        }

        // --- Action ---

        float tdErrorAction = newValue - (*hiddenValuesPrev)[hiddenColumnIndex];

        if (tdErrorAction > 0.0f || mimic) {
            unsigned char targetC = (*hiddenTargetCsPrev)[hiddenColumnIndex];

            int maxIndex = 0;
            float maxActivation = 0;

            for (int hc = 0; hc < hiddenSize.z; hc++) {
                int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

                int sum = 0;
                count = 0;

                for (int vli = 0; vli < visibleLayers.size(); vli++) {
                    VisibleLayer &vl = visibleLayers[vli];
                    const ActorVisibleLayerDesc &vld = visibleLayerDescs[vli];

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

                            T weight = vl.actionWeights[offset.y + diam * (offset.x + diam * hiddenIndex)];

                            unsigned char inC = (*inputCsPrev[vli])[visibleColumnIndex];

                            sum += (weight & (1 << inC)) == 0 ? 0 : 1;
                            count++;
                        }
                }

                hiddenProbs[hiddenIndex] = static_cast<float>(sum) / static_cast<float>(count);

                if (sum > maxActivation) {
                    maxActivation = sum;
                    maxIndex = hc;
                }
            }

            if (maxIndex != targetC) {
                for (int hc = 0; hc < hiddenSize.z; hc++) {
                    int hiddenIndexTarget = address3(Int3(pos.x, pos.y, targetC), hiddenSize);
                    int hiddenIndexMax = address3(Int3(pos.x, pos.y, maxIndex), hiddenSize);

                    float probIncrease = beta * (1.0f - hiddenProbs[hiddenIndexTarget]);
                    float probDecrease = beta * hiddenProbs[hiddenIndexMax];

                    for (int vli = 0; vli < visibleLayers.size(); vli++) {
                        VisibleLayer &vl = visibleLayers[vli];
                        const ActorVisibleLayerDesc &vld = visibleLayerDescs[vli];

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

                                if (randf(state) < probIncrease) {
                                    int wi = offset.y + diam * (offset.x + diam * hiddenIndexTarget);

                                    T weight = vl.actionWeights[wi];

                                    unsigned char inC = (*inputCsPrev[vli])[visibleColumnIndex];

                                    vl.actionWeights[wi] = increase<T>(weight, inC);
                                }

                                if (randf(state) < probDecrease) {
                                    int wi = offset.y + diam * (offset.x + diam * hiddenIndexMax);

                                    T weight = vl.actionWeights[wi];

                                    unsigned char inC = (*inputCsPrev[vli])[visibleColumnIndex];

                                    vl.actionWeights[wi] = decrease<T>(weight, inC);
                                }
                            }
                    }
                }
            }
        }
    }

public:
    float alpha; // Value learning rate
    float beta; // Action learning rate
    float gamma; // Discount factor
    int minSteps;
    int historyIters;

    // Defaults
    Actor()
    :
    alpha(0.02f),
    beta(0.5f),
    gamma(0.99f),
    minSteps(8),
    historyIters(8)
    {}

    // Initialized randomly
    void initRandom(
        const Int3 &hiddenSize,
        int historyCapacity,
        const Array<ActorVisibleLayerDesc> &visibleLayerDescs
    ) {
        this->visibleLayerDescs = visibleLayerDescs;

        this->hiddenSize = hiddenSize;

        visibleLayers.resize(visibleLayerDescs.size());

        // Pre-compute dimensions
        int numHiddenColumns = hiddenSize.x * hiddenSize.y;
        int numHidden = numHiddenColumns * hiddenSize.z;

        // Create layers
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            ActorVisibleLayerDesc &vld = this->visibleLayerDescs[vli];

            // Create weight matrix for this visible layer and initialize randomly
            vl.valueWeights.initSMLocalRF(vld.size, Int3(hiddenSize.x, hiddenSize.y, 1), vld.radius);

            for (int i = 0; i < vl.valueWeights.nonZeroValues.size(); i++)
                vl.valueWeights.nonZeroValues[i] = randf(-0.01f, 0.01f);

            int diam = vld.radius * 2 + 1;
            int area = diam * diam;

            vl.actionWeights.resize(numHidden * area);

            // Initialize to random values
            for (int i = 0; i < vl.actionWeights.size(); i++)
                vl.actionWeights[i] = randBits<T>();
        }

        hiddenCs = ByteBuffer(numHiddenColumns, 0);

        hiddenValues = FloatBuffer(numHiddenColumns, 0.0f);

        hiddenProbs = FloatBuffer(numHidden, 0.0f);

        // Create (pre-allocated) history samples
        historySize = 0;
        historySamples.resize(historyCapacity);

        for (int i = 0; i < historySamples.size(); i++) {
            historySamples[i].inputCs.resize(visibleLayers.size());

            for (int vli = 0; vli < visibleLayers.size(); vli++) {
                ActorVisibleLayerDesc &vld = this->visibleLayerDescs[vli];

                int numVisibleColumns = vld.size.x * vld.size.y;

                historySamples[i].inputCs[vli] = ByteBuffer(numVisibleColumns);
            }

            historySamples[i].hiddenTargetCsPrev = ByteBuffer(numHiddenColumns);

            historySamples[i].hiddenValuesPrev = FloatBuffer(numHiddenColumns);
        }
    }

    // Step (get actions and update)
    void step(
        const Array<const ByteBuffer*> &inputCs,
        const ByteBuffer* hiddenTargetCsPrev,
        float reward,
        bool learnEnabled,
        bool mimic
    ) {
        int numHiddenColumns = hiddenSize.x * hiddenSize.y;

        // Forward kernel
        #pragma omp parallel for
        for (int i = 0; i < numHiddenColumns; i++)
            forward(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCs);

        historySamples.pushFront();

        // If not at cap, increment
        if (historySize < historySamples.size())
            historySize++;
        
        // Add new sample
        {
            HistorySample &s = historySamples[0];

            for (int vli = 0; vli < visibleLayers.size(); vli++)
                s.inputCs[vli] = *inputCs[vli];

            // Copy hidden Cs
            s.hiddenTargetCsPrev = *hiddenTargetCsPrev;

            // Copy hidden values
            s.hiddenValuesPrev = hiddenValues;

            s.reward = reward;
        }

        // Learn (if have sufficient samples)
        if (learnEnabled && historySize > minSteps + 1) {
            for (int it = 0; it < historyIters; it++) {
                int historyIndex = rand() % (historySize - 1 - minSteps) + minSteps;

                const HistorySample &sPrev = historySamples[historyIndex + 1];
                const HistorySample &s = historySamples[historyIndex];

                // Compute (partial) values, rest is completed in the kernel
                float q = 0.0f;
                float g = 1.0f;

                for (int t = historyIndex; t >= 0; t--) {
                    q += historySamples[t].reward * g;

                    g *= gamma;
                }

                unsigned long baseState = rand();

                #pragma omp parallel for
                for (int i = 0; i < numHiddenColumns; i++) {
                    unsigned long state = baseState + i;

                    learn(Int2(i / hiddenSize.y, i % hiddenSize.y), constGet(sPrev.inputCs), &s.hiddenTargetCsPrev, &sPrev.hiddenValuesPrev, q, g, mimic, &state);
                }
            }
        }
    }

    // Get number of visible layers
    int getNumVisibleLayers() const {
        return visibleLayers.size();
    }

    // Get a visible layer
    const VisibleLayer &getVisibleLayer(
        int i // Index of layer
    ) const {
        return visibleLayers[i];
    }

    // Get a visible layer descriptor
    const ActorVisibleLayerDesc &getVisibleLayerDesc(
        int i // Index of layer
    ) const {
        return visibleLayerDescs[i];
    }

    // Get hidden state/output/actions
    const ByteBuffer &getHiddenCs() const {
        return hiddenCs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }
};
} // namespace aon
