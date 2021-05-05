// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Actor.h"

using namespace aon;

void Actor::forward(
    const Int2 &columnPos,
    const Array<const IntBuffer*> &inputCIs,
    const IntBuffer* hiddenTargetCIsPrev,
    float reward,
    bool learnEnabled,
    unsigned int* state
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    // Pre-count
    int count = 0;

    // For each visible layer
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        int diam = vld.radius * 2 + 1;

        // Projection
        Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
            static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));

        Int2 visibleCenter = project(columnPos, hToV);

        visibleCenter = minOverhang(visibleCenter, Int2(vld.size.x, vld.size.y), vld.radius);

        // Lower corner
        Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

        // Bounds of receptive field, clamped to input size
        Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
        Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

        count += (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1);
    }

    int targetCI = (*hiddenTargetCIsPrev)[hiddenColumnIndex];

    float valuePrev = hiddenValues[address3(Int3(columnPos.x, columnPos.y, targetCI), hiddenSize)];

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        float sum = 0.0f;

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int diam = vld.radius * 2 + 1;

            // Projection
            Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
                static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));

            Int2 visibleCenter = project(columnPos, hToV);

            visibleCenter = minOverhang(visibleCenter, Int2(vld.size.x, vld.size.y), vld.radius);

            // Lower corner
            Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

            // Bounds of receptive field, clamped to input size
            Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
            Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                    int inCI = (*inputCIs[vli])[visibleColumnIndex];

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    sum += vl.weights[inCI + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex))];
                }
        }

        sum /= max(1, count);

        hiddenValues[hiddenCellIndex] = sum;

        if (sum > maxActivation || maxIndex == -1) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    int selectIndex = maxIndex;

    if (randf(state) < epsilon)
        selectIndex = rand(state) % hiddenSize.z;

    float nextValue = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);
    
        nextValue += hiddenValues[hiddenCellIndex] * (hc == maxIndex ? 1.0f - epsilon : epsilon / (hiddenSize.z - 1));
    }

    float delta = lr * (reward + discount * nextValue - valuePrev);

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int diam = vld.radius * 2 + 1;

            // Projection
            Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
                static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));

            Int2 visibleCenter = project(columnPos, hToV);

            visibleCenter = minOverhang(visibleCenter, Int2(vld.size.x, vld.size.y), vld.radius);

            // Lower corner
            Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

            // Bounds of receptive field, clamped to input size
            Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
            Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                    int inCIPrev = vl.inputCIsPrev[visibleColumnIndex];

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                    for (int vc = 0; vc < vld.size.z; vc++) {
                        int wi = vc + wiStart;

                        if (vc == inCIPrev && hc == targetCI)
                            vl.traces[wi] = vl.traces[wi] * traceDecay + 1.0f;
                        else
                            vl.traces[wi] *= traceDecay;

                        if (learnEnabled)
                            vl.weights[wi] += delta * vl.traces[wi];
                    }
                }
        }
    }

    hiddenCIs[hiddenColumnIndex] = selectIndex;
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
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    // Create layers
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        // Create weight matrix for this visible layer and initialize randomly
        vl.weights.resize(numHiddenCells * area * vld.size.z);

        for (int i = 0; i < vl.weights.size(); i++)
            vl.weights[i] = randf(-0.001f, 0.001f);

        vl.traces = FloatBuffer(vl.weights.size(), 0.0f);

        vl.inputCIsPrev = IntBuffer(numVisibleColumns, 0);
    }

    hiddenValues = FloatBuffer(numHiddenCells, 0.0f);
    hiddenCIs = IntBuffer(numHiddenColumns, 0);
}

void Actor::step(
    const Array<const IntBuffer*> &inputCIs,
    const IntBuffer* hiddenTargetCIsPrev,
    float reward,
    bool learnEnabled
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    unsigned int baseState = rand();

    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++) {
        unsigned int state = baseState + i * 12345;

        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCIs, hiddenTargetCIsPrev, reward, learnEnabled, &state);
    }

    // Updates prevs
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];

        vl.inputCIsPrev = *inputCIs[vli];
    }
}

int Actor::size() const {
    int size = sizeof(Int3) + 4 * sizeof(float) + hiddenValues.size() * sizeof(float) + hiddenCIs.size() * sizeof(int) + sizeof(int);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        size += sizeof(VisibleLayerDesc) + 2 * vl.weights.size() * sizeof(float);
    }

    return size;
}

int Actor::stateSize() const {
    return hiddenValues.size() * sizeof(float) + hiddenCIs.size() * sizeof(int) + sizeof(int);
}

void Actor::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&discount), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&traceDecay), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&epsilon), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenValues[0]), hiddenValues.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    int numVisibleCellsLayers = visibleLayers.size();

    writer.write(reinterpret_cast<const void*>(&numVisibleCellsLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

        writer.write(reinterpret_cast<const void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));
        writer.write(reinterpret_cast<const void*>(&vl.traces[0]), vl.traces.size() * sizeof(float));

        writer.write(reinterpret_cast<const void*>(&vl.inputCIsPrev[0]), vl.inputCIsPrev.size() * sizeof(int));
    }
}

void Actor::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;
    
    reader.read(reinterpret_cast<void*>(&lr), sizeof(float));
    reader.read(reinterpret_cast<void*>(&discount), sizeof(float));
    reader.read(reinterpret_cast<void*>(&traceDecay), sizeof(float));
    reader.read(reinterpret_cast<void*>(&epsilon), sizeof(float));

    hiddenValues.resize(numHiddenCells);
    hiddenCIs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenValues[0]), hiddenValues.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    int numVisibleCellsLayers = visibleLayers.size();

    reader.read(reinterpret_cast<void*>(&numVisibleCellsLayers), sizeof(int));

    visibleLayers.resize(numVisibleCellsLayers);
    visibleLayerDescs.resize(numVisibleCellsLayers);
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = visibleLayerDescs[vli];

        reader.read(reinterpret_cast<void*>(&vld), sizeof(VisibleLayerDesc));

        int numVisibleColumns = vld.size.x * vld.size.y;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        int weightsSize = numHiddenCells * area * vld.size.z;

        vl.weights.resize(weightsSize);
        vl.traces.resize(weightsSize);

        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));
        reader.read(reinterpret_cast<void*>(&vl.traces[0]), vl.traces.size() * sizeof(float));

        vl.inputCIsPrev.resize(numVisibleColumns);

        reader.read(reinterpret_cast<void*>(&vl.inputCIsPrev[0]), vl.inputCIsPrev.size() * sizeof(int));
    }
}

void Actor::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenValues[0]), hiddenValues.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];

        writer.write(reinterpret_cast<const void*>(&vl.traces[0]), vl.traces.size() * sizeof(float));

        writer.write(reinterpret_cast<const void*>(&vl.inputCIsPrev[0]), vl.inputCIsPrev.size() * sizeof(int));
    }
}

void Actor::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenValues[0]), hiddenValues.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];

        reader.read(reinterpret_cast<void*>(&vl.traces[0]), vl.traces.size() * sizeof(float));

        reader.read(reinterpret_cast<void*>(&vl.inputCIsPrev[0]), vl.inputCIsPrev.size() * sizeof(int));
    }
}
