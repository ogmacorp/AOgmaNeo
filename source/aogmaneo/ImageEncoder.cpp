// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "ImageEncoder.h"

using namespace aon;

void ImageEncoder::forward(
    const Int2 &columnPos,
    const Array<const ByteBuffer*> &inputs,
    bool learnEnabled
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int maxIndex = -1;
    int maxActivation = -999999;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        int sum = 0;

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

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                    for (int vc = 0; vc < vld.size.z; vc++) {
                        int input = (*inputs[vli])[address3(Int3(ix, iy, vc), vld.size)];

                        int delta = input - static_cast<int>(vl.protos[vc + wiStart]);

                        sum -= delta * delta;
                    }
                }
        }

        if (sum > maxActivation || maxIndex == -1) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    hiddenCIs[hiddenColumnIndex] = maxIndex;

    if (learnEnabled) {
        for (int hc = 0; hc < hiddenSize.z; hc++) {
            int dist = maxIndex - hc;

            int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

            float strength = expf(-dist * dist * gamma / max(0.0001f, hiddenRates[hiddenCellIndex])) * hiddenRates[hiddenCellIndex];

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

                for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                    for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                        int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                        Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                        int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                        for (int vc = 0; vc < vld.size.z; vc++) {
                            unsigned char input = (*inputs[vli])[address3(Int3(ix, iy, vc), vld.size)];

                            unsigned char weight = vl.protos[vc + wiStart];

                            vl.protos[wiStart + vc] = roundftoi(min(255.0f, max(0.0f, weight + strength * (static_cast<float>(input) - static_cast<float>(weight)))));
                        }
                    }
            }

            hiddenRates[hiddenCellIndex] -= alpha * strength;
        }
    }
}

void ImageEncoder::reconstruct(
    const Int2 &columnPos,
    const IntBuffer* reconCIs,
    int vli
) {
    VisibleLayer &vl = visibleLayers[vli];
    VisibleLayerDesc &vld = visibleLayerDescs[vli];

    int diam = vld.radius * 2 + 1;

    int visibleColumnIndex = address2(columnPos, Int2(vld.size.x, vld.size.y));

    // Projection
    Float2 vToH = Float2(static_cast<float>(hiddenSize.x) / static_cast<float>(vld.size.x),
        static_cast<float>(hiddenSize.y) / static_cast<float>(vld.size.y));

    Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
                static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));
                
    Int2 hiddenCenter = project(columnPos, vToH);

    Int2 reverseRadii(ceilf(vToH.x * vld.radius) + 1, ceilf(vToH.y * vld.radius) + 1);
    
    // Lower corner
    Int2 fieldLowerBound(hiddenCenter.x - reverseRadii.x, hiddenCenter.y - reverseRadii.y);

    // Bounds of receptive field, clamped to input size
    Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
    Int2 iterUpperBound(min(hiddenSize.x - 1, hiddenCenter.x + reverseRadii.x), min(hiddenSize.y - 1, hiddenCenter.y + reverseRadii.y));
    
    // Find current max
    for (int vc = 0; vc < vld.size.z; vc++) {
        int visibleIndex = address3(Int3(columnPos.x, columnPos.y, vc), vld.size);

        float sum = 0.0f;
        float total = 0.0f;

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                Int2 hiddenPos = Int2(ix, iy);

                int hiddenColumnIndex = address2(hiddenPos, Int2(hiddenSize.x, hiddenSize.y));
                int hiddenCellIndex = address3(Int3(hiddenPos.x, hiddenPos.y, (*reconCIs)[hiddenColumnIndex]), hiddenSize);

                Int2 visibleCenter = project(hiddenPos, hToV);

                visibleCenter = minOverhang(visibleCenter, Int2(vld.size.x, vld.size.y), vld.radius);

                float distX = static_cast<float>(abs(columnPos.x - visibleCenter.x)) / static_cast<float>(vld.radius + 1);
                float distY = static_cast<float>(abs(columnPos.y - visibleCenter.y)) / static_cast<float>(vld.radius + 1);

                float strength = min(1.0f - distX, 1.0f - distY);

                if (inBounds(columnPos, Int2(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius), Int2(visibleCenter.x + vld.radius + 1, visibleCenter.y + vld.radius + 1))) {
                    Int2 offset(columnPos.x - visibleCenter.x + vld.radius, columnPos.y - visibleCenter.y + vld.radius);

                    sum += strength * vl.protos[vc + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex))];
                    total += strength;
                }
            }

        vl.reconstruction[visibleIndex] = roundftoi(sum / max(0.0001f, total));
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

        vl.protos.resize(numHiddenCells * area * vld.size.z);

        // Initialize to random values
        for (int i = 0; i < vl.protos.size(); i++)
            vl.protos[i] = rand() % 256;

        vl.reconstruction = ByteBuffer(numVisibleCells, 0);
    }

    // Hidden CIs
    hiddenCIs = IntBuffer(numHiddenColumns, 0);

    hiddenRates = FloatBuffer(numHiddenCells, 1.0f);
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

int ImageEncoder::size() const {
    int size = sizeof(Int3) + 2 * sizeof(float) + hiddenCIs.size() * sizeof(int) + hiddenRates.size() * sizeof(float) + sizeof(int);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        size += sizeof(VisibleLayerDesc) + vl.protos.size() * sizeof(unsigned char);
    }

    return size;
}

void ImageEncoder::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&alpha), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&gamma), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hiddenRates[0]), hiddenRates.size() * sizeof(float));
    
    int numVisibleLayers = visibleLayers.size();

    writer.write(reinterpret_cast<const void*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

        writer.write(reinterpret_cast<const void*>(&vl.protos[0]), vl.protos.size() * sizeof(unsigned char));
    }
}

void ImageEncoder::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&alpha), sizeof(float));
    reader.read(reinterpret_cast<void*>(&gamma), sizeof(float));

    hiddenCIs.resize(numHiddenColumns);
    hiddenRates.resize(numHiddenCells);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hiddenRates[0]), hiddenRates.size() * sizeof(float));

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

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.protos.resize(numHiddenCells * area * vld.size.z);

        reader.read(reinterpret_cast<void*>(&vl.protos[0]), vl.protos.size() * sizeof(unsigned char));

        vl.reconstruction = ByteBuffer(numVisibleCells, 0);
    }
}

