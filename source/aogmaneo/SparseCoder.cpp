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
    const FloatBuffer* hiddenErrors,
    const Array<const IntBuffer*> &inputCIs,
    bool learnEnabled
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    if (learnEnabled) {
        for (int hc = 0; hc < hiddenSize.z; hc++) {
            int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

            float delta = alpha * tanh((*hiddenErrors)[hiddenColumnIndex]) * ((hc == hiddenCIs[hiddenColumnIndex] ? 1.0f : 0.0f) - hiddenActivations[hiddenCellIndex]);

            hiddenBiases[hiddenCellIndex] += gamma * (1.0f / hiddenSize.z - (hc == hiddenCIs[hiddenColumnIndex] ? 1.0f : 0.0f));

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
                        int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                        Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                        int inCI = vl.inputCIsPrev[visibleColumnIndex];

                        // Missing value handling
                        if (inCI == -1)
                            continue;

                        int wi = inCI + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                        vl.weights[wi] += delta;
                    }
            }
        }
    }

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        float sum = hiddenBiases[hiddenCellIndex];
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
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    int inCI = (*inputCIs[vli])[visibleColumnIndex];

                    // Missing value handling
                    if (inCI == -1)
                        continue;

                    sum += vl.weights[inCI + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex))];
                    count++;
                }
        }

        sum /= max(1, count);

        hiddenActivations[hiddenCellIndex] = sum;

        if (sum > maxActivation || maxIndex == -1) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    // Softmax
    float total = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        hiddenActivations[hiddenCellIndex] = expf(hiddenActivations[hiddenCellIndex] - maxActivation);

        total += hiddenActivations[hiddenCellIndex];
    }

    float scale = 1.0f / max(0.0001f, total);

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        hiddenActivations[hiddenCellIndex] *= scale;
    }

    hiddenCIs[hiddenColumnIndex] = maxIndex;
}

void SparseCoder::learnRecon(
    const Int2 &columnPos,
    int vli
) {
    VisibleLayer &vl = visibleLayers[vli];
    VisibleLayerDesc &vld = visibleLayerDescs[vli];

    int diam = vld.radius * 2 + 1;

    int visibleColumnIndex = address2(columnPos, Int2(vld.size.x, vld.size.y));

    int targetCI = vl.inputCIsPrev[visibleColumnIndex];

    if (targetCI == -1)
        return;

    // Projection
    Float2 vToH = Float2(static_cast<float>(hiddenSize.x) / static_cast<float>(vld.size.x),
        static_cast<float>(hiddenSize.y) / static_cast<float>(vld.size.y));

    Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
        static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));
                
    Int2 hiddenCenter = project(columnPos, vToH);

    Int2 reverseRadii(ceilf(vToH.x * (vld.radius * 2 + 1) * 0.5f), ceilf(vToH.y * (vld.radius * 2 + 1) * 0.5f));

    // Lower corner
    Int2 fieldLowerBound(hiddenCenter.x - reverseRadii.x, hiddenCenter.y - reverseRadii.y);

    // Bounds of receptive field, clamped to input size
    Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
    Int2 iterUpperBound(min(hiddenSize.x - 1, hiddenCenter.x + reverseRadii.x), min(hiddenSize.y - 1, hiddenCenter.y + reverseRadii.y));

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int vc = 0; vc < vld.size.z; vc++) {
        int visibleCellIndex = address3(Int3(columnPos.x, columnPos.y, vc), vld.size);

        float sum = 0.0f;
        int count = 0;

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                Int2 hiddenPos = Int2(ix, iy);

                int hiddenColumnIndex = address2(hiddenPos, Int2(hiddenSize.x, hiddenSize.y));
                int hiddenCellIndex = address3(Int3(hiddenPos.x, hiddenPos.y, hiddenCIs[hiddenColumnIndex]), hiddenSize);

                Int2 visibleCenter = project(hiddenPos, hToV);

                if (inBounds(columnPos, Int2(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius), Int2(visibleCenter.x + vld.radius + 1, visibleCenter.y + vld.radius + 1))) {
                    Int2 offset(columnPos.x - visibleCenter.x + vld.radius, columnPos.y - visibleCenter.y + vld.radius);

                    sum += vl.weights[vc + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex))];
                    count++;
                }
            }

        sum /= max(1, count);

        vl.reconstruction[visibleCellIndex] = sum;

        if (sum > maxActivation || maxIndex == -1) {
            maxActivation = sum;
            maxIndex = vc;
        }
    }

    // Softmax
    float total = 0.0f;

    for (int vc = 0; vc < vld.size.z; vc++) {
        int visibleCellIndex = address3(Int3(columnPos.x, columnPos.y, vc), vld.size);

        vl.reconstruction[visibleCellIndex] = expf(vl.reconstruction[visibleCellIndex] - maxActivation);

        total += vl.reconstruction[visibleCellIndex];
    }

    float scale = 1.0f / max(0.0001f, total);

    for (int vc = 0; vc < vld.size.z; vc++) {
        int visibleCellIndex = address3(Int3(columnPos.x, columnPos.y, vc), vld.size);

        vl.reconstruction[visibleCellIndex] *= scale;
    }

    for (int vc = 0; vc < vld.size.z; vc++) {
        int visibleCellIndex = address3(Int3(columnPos.x, columnPos.y, vc), vld.size);

        float delta = beta * ((vc == targetCI ? 1.0f : 0.0f) - vl.reconstruction[visibleCellIndex]);
  
        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                Int2 hiddenPos = Int2(ix, iy);

                int hiddenColumnIndex = address2(hiddenPos, Int2(hiddenSize.x, hiddenSize.y));

                int hiddenCellIndex = address3(Int3(hiddenPos.x, hiddenPos.y, hiddenCIs[hiddenColumnIndex]), hiddenSize);

                Int2 visibleCenter = project(hiddenPos, hToV);

                if (inBounds(columnPos, Int2(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius), Int2(visibleCenter.x + vld.radius + 1, visibleCenter.y + vld.radius + 1))) {
                    Int2 offset(columnPos.x - visibleCenter.x + vld.radius, columnPos.y - visibleCenter.y + vld.radius);

                    vl.weights[vc + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex))] += delta;
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
        int numVisibleCells = numVisibleColumns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(numHiddenCells * area * vld.size.z);

        // Initialize to random values
        for (int i = 0; i < vl.weights.size(); i++)
            vl.weights[i] = randf(0.0f, 1.0f);

        vl.reconstruction = FloatBuffer(numVisibleCells, 0.0f);

        vl.inputCIsPrev = IntBuffer(numVisibleColumns, 0);
    }

    hiddenActivations = FloatBuffer(numHiddenCells, 0.0f);

    hiddenCIs = IntBuffer(numHiddenColumns, 0);

    hiddenBiases = FloatBuffer(numHiddenCells, 0.0f);
}

void SparseCoder::step(
    const Array<const IntBuffer*> &inputCIs,
    const FloatBuffer* hiddenErrors,
    bool learnEnabled
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    if (learnEnabled) {
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int numVisibleColumns = vld.size.x * vld.size.y;
        
            #pragma omp parallel for
            for (int i = 0; i < numVisibleColumns; i++)
                learnRecon(Int2(i / vld.size.y, i % vld.size.y), vli);
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), hiddenErrors, inputCIs, learnEnabled);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];

        vl.inputCIsPrev = *inputCIs[vli];
    }
}

int SparseCoder::size() const {
    int size = sizeof(Int3) + 2 * sizeof(float) + hiddenActivations.size() * sizeof(float) + hiddenCIs.size() * sizeof(int) + hiddenBiases.size() * sizeof(float) + sizeof(int);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];

        size += sizeof(VisibleLayerDesc) + sizeof(int) + vl.weights.size() * sizeof(float) + vl.inputCIsPrev.size() * sizeof(int);
    }

    return size;
}

int SparseCoder::stateSize() const {
    int size = hiddenActivations.size() * sizeof(float) + hiddenCIs.size() * sizeof(int);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];

        size += vl.inputCIsPrev.size() * sizeof(int);
    }

    return size;
}

void SparseCoder::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&alpha), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&beta), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&gamma), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenActivations[0]), hiddenActivations.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hiddenBiases[0]), hiddenBiases.size() * sizeof(float));
    
    int numVisibleLayers = visibleLayers.size();

    writer.write(reinterpret_cast<const void*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

        int weightsSize = vl.weights.size();

        writer.write(reinterpret_cast<const void*>(&weightsSize), sizeof(int));

        writer.write(reinterpret_cast<const void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));

        writer.write(reinterpret_cast<const void*>(&vl.inputCIsPrev[0]), vl.inputCIsPrev.size() * sizeof(int));
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

    hiddenActivations.resize(numHiddenCells);
    hiddenCIs.resize(numHiddenColumns);
    hiddenBiases.resize(numHiddenCells);

    reader.read(reinterpret_cast<void*>(&hiddenActivations[0]), hiddenActivations.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hiddenBiases[0]), hiddenBiases.size() * sizeof(float));

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

        vl.weights.resize(weightsSize);

        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));

        vl.inputCIsPrev.resize(numVisibleColumns);

        reader.read(reinterpret_cast<void*>(&vl.inputCIsPrev[0]), vl.inputCIsPrev.size() * sizeof(int));

        vl.reconstruction = FloatBuffer(numVisibleCells, 0.0f);
    }
}

void SparseCoder::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];

        writer.write(reinterpret_cast<const void*>(&vl.inputCIsPrev[0]), vl.inputCIsPrev.size() * sizeof(int));
    }
}

void SparseCoder::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];

        reader.read(reinterpret_cast<void*>(&vl.inputCIsPrev[0]), vl.inputCIsPrev.size() * sizeof(int));
    }
}
