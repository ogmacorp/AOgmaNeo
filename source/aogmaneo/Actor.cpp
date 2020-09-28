// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Actor.h"

using namespace aon;

void Actor::activate(
    const Int2 &pos,
    const Array<const IntBuffer*> &inputCs,
    unsigned long* state
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        float sum = 0.0f;
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
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    int inC = (*inputCs[vli])[visibleColumnIndex];

                    float weight = vl.weights[inC + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex))];
                    
                    sum += weight;
                    count++;
                }
        }

        sum /= max(1, count);

        hiddenActivations[hiddenIndex] = sum;

        maxActivation = max(maxActivation, sum);
    }

    float total = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        hiddenProbabilities[hiddenIndex] = expf((hiddenActivations[hiddenIndex] - maxActivation) / temperature);
        
        total += hiddenProbabilities[hiddenIndex];
    }

    float cusp = randf(state) * total;

    int selectIndex = 0;
    float sumSoFar = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

        sumSoFar += hiddenProbabilities[hiddenIndex];

        if (sumSoFar >= cusp) {
            selectIndex = hc;

            break;
        }
    }
    
    hiddenCs[hiddenColumnIndex] = selectIndex;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

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

                    int inC = (*inputCs[vli])[visibleColumnIndex];

                    for (int vc = 0; vc < vld.size.z; vc++) {
                        int wi = vc + wiStart;

                        if (vc == inC && hc == selectIndex)
                            vl.traces[wi] = 1.0f;
                        else
                            vl.traces[wi] *= traceDecay;
                    }
                }
        }
    }
}

void Actor::learn(
    const Int2 &pos,
    const FloatBuffer* hiddenErrors
) {
    int hiddenColumnIndex = address2(pos, Int2(hiddenSize.x, hiddenSize.y));

    float reward = sigmoid((*hiddenErrors)[hiddenColumnIndex]) * 2.0f - 1.0f;

    float maxActivation = hiddenActivations[address3(Int3(pos.x, pos.y, 0), hiddenSize)];

    for (int hc = 1; hc < hiddenSize.z; hc++)
        maxActivation = max(maxActivation, hiddenActivations[address3(Int3(pos.x, pos.y, 0), hiddenSize)]);

    float delta = alpha * (reward + gamma * maxActivation - hiddenValuesPrev[hiddenColumnIndex]);

    hiddenValuesPrev[hiddenColumnIndex] = hiddenActivations[address3(Int3(pos.x, pos.y, hiddenCs[hiddenColumnIndex]), hiddenSize)];

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenIndex = address3(Int3(pos.x, pos.y, hc), hiddenSize);

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
                        int wi = vc + wiStart;

                        vl.weights[wi] += delta * vl.tracesBackup[wi];
                        vl.tracesBackup[wi] = vl.traces[wi];
                    }
                }
        }
    }
}

void Actor::initRandom(
    const Int3 &hiddenSize,
    const Array<VisibleLayerDesc> &visibleLayerDescs
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
        VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        // If last one (recurrent), init conservatively
        vl.weights.resize(numHidden * area * vld.size.z);
        vl.traces.resize(vl.weights.size());

        if (vli != visibleLayers.size() - 1) {
            for (int i = 0; i < vl.weights.size(); i++) {
                vl.weights[i] = randf(-1.0f, 1.0f);
                vl.traces[i] = 0.0f;
            }
        }
        else {
            for (int i = 0; i < vl.weights.size(); i++) {
                vl.weights[i] = randf(-0.01f, 0.01f);
                vl.traces[i] = 0.0f;
            }
        }

        vl.tracesBackup = vl.traces;
    }

    hiddenCs = IntBuffer(numHiddenColumns, 0);

    hiddenActivations = FloatBuffer(numHidden, 0.0f);
    hiddenProbabilities = FloatBuffer(numHidden, 0.0f);

    hiddenValuesPrev = FloatBuffer(numHiddenColumns, 0.0f);
}

void Actor::activate(
    const Array<const IntBuffer*> &inputCs
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    hiddenCsPrev = hiddenCs;

    unsigned int baseState = rand();

    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++) {
        unsigned long state = baseState + i * 12345;

        activate(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCs, &state);
    }
}

void Actor::learn(
    const FloatBuffer* hiddenErrors
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        learn(Int2(i / hiddenSize.y, i % hiddenSize.y), hiddenErrors);
}

void Actor::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&temperature), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&alpha), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&gamma), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&traceDecay), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenCs[0]), hiddenCs.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hiddenValuesPrev[0]), hiddenValuesPrev.size() * sizeof(float));

    int numVisibleLayers = visibleLayers.size();

    writer.write(reinterpret_cast<const void*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

        int weightsSize = vl.weights.size();

        writer.write(reinterpret_cast<const void*>(&weightsSize), sizeof(int));

        writer.write(reinterpret_cast<const void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));
        writer.write(reinterpret_cast<const void*>(&vl.traces[0]), vl.traces.size() * sizeof(float));
    }
}

void Actor::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHidden = numHiddenColumns * hiddenSize.z;
    
    reader.read(reinterpret_cast<void*>(&temperature), sizeof(float));
    reader.read(reinterpret_cast<void*>(&alpha), sizeof(float));
    reader.read(reinterpret_cast<void*>(&gamma), sizeof(float));
    reader.read(reinterpret_cast<void*>(&traceDecay), sizeof(float));

    hiddenCs.resize(numHiddenColumns);
    hiddenCsPrev.resize(numHiddenColumns);
    hiddenValuesPrev.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCs[0]), hiddenCs.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hiddenValuesPrev[0]), hiddenValuesPrev.size() * sizeof(float));

    hiddenActivations = FloatBuffer(numHidden, 0.0f);
    hiddenProbabilities = FloatBuffer(numHidden, 0.0f);
    
    int numVisibleLayers = visibleLayers.size();

    reader.read(reinterpret_cast<void*>(&numVisibleLayers), sizeof(int));

    visibleLayers.resize(numVisibleLayers);
    visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = visibleLayerDescs[vli];

        reader.read(reinterpret_cast<void*>(&vld), sizeof(VisibleLayerDesc));

        int numVisibleColumns = vld.size.x * vld.size.y;

        int weightsSize;

        reader.read(reinterpret_cast<void*>(&weightsSize), sizeof(int));

        vl.weights.resize(weightsSize);
        vl.traces.resize(weightsSize);

        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));
        reader.read(reinterpret_cast<void*>(&vl.traces[0]), vl.traces.size() * sizeof(float));
    }
}
