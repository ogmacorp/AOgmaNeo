// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Layer.h"

using namespace aon;

void Layer::forward(
    const Int2 &columnPos,
    const Array<const IntBuffer*> &inputCIs,
    unsigned int* state
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        float sum = 0.0f;
        int count = 0;

        // For each visible layer
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            if (inputCIs[vli] == nullptr)
                continue;

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
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    int inCI = (*inputCIs[vli])[visibleColumnIndex];

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

    hiddenCIs[hiddenColumnIndex] = maxIndex;

    float total = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        hiddenActivations[hiddenCellIndex] = expf(hiddenActivations[hiddenCellIndex] - maxActivation);
        
        total += hiddenActivations[hiddenCellIndex];
    }

    float cusp = randf(state) * total;

    int selectIndex = 0;
    float sumSoFar = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        sumSoFar += hiddenActivations[hiddenCellIndex];

        if (sumSoFar >= cusp) {
            selectIndex = hc;

            break;
        }
    }
    
    hiddenRandomCIs[hiddenColumnIndex] = selectIndex;
}

void Layer::backward(
    const Int2 &columnPos,
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

        for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
            for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                Int2 hiddenPos = Int2(ix, iy);

                int hiddenColumnIndex = address2(hiddenPos, Int2(hiddenSize.x, hiddenSize.y));
                int hiddenCellIndex = address3(Int3(hiddenPos.x, hiddenPos.y, hiddenCIs[hiddenColumnIndex]), hiddenSize);

                Int2 visibleCenter = project(hiddenPos, hToV);

                if (inBounds(columnPos, Int2(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius), Int2(visibleCenter.x + vld.radius + 1, visibleCenter.y + vld.radius + 1))) {
                    Int2 offset(columnPos.x - visibleCenter.x + vld.radius, columnPos.y - visibleCenter.y + vld.radius);

                    sum += vl.weights[vc + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex))];
                }
            }

        if (sum > maxActivation || maxIndex == -1) {
            maxActivation = sum;
            maxIndex = vc;
        }
    }

    vl.visibleCIs[visibleColumnIndex] = maxIndex;
}

void Layer::reconBackward(
    const Int2 &columnPos,
    int vli,
    unsigned int* state
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
                int hiddenCellIndex = address3(Int3(hiddenPos.x, hiddenPos.y, hiddenRandomCIs[hiddenColumnIndex]), hiddenSize);

                Int2 visibleCenter = project(hiddenPos, hToV);

                if (inBounds(columnPos, Int2(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius), Int2(visibleCenter.x + vld.radius + 1, visibleCenter.y + vld.radius + 1))) {
                    Int2 offset(columnPos.x - visibleCenter.x + vld.radius, columnPos.y - visibleCenter.y + vld.radius);

                    sum += vl.weights[vc + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex))];
                    count++;
                }
            }

        sum /= max(1, count);

        vl.visibleActivations[visibleCellIndex] = sum;

        if (sum > maxActivation || maxIndex == -1) {
            maxActivation = sum;
            maxIndex = vc;
        }
    }

    float total = 0.0f;

    for (int vc = 0; vc < vld.size.z; vc++) {
        int visibleCellIndex = address3(Int3(columnPos.x, columnPos.y, vc), vld.size);

        vl.visibleActivations[visibleCellIndex] = expf(vl.visibleActivations[visibleCellIndex] - maxActivation);
        
        total += vl.visibleActivations[visibleCellIndex];
    }

    float cusp = randf(state) * total;

    int selectIndex = 0;
    float sumSoFar = 0.0f;

    for (int vc = 0; vc < vld.size.z; vc++) {
        int visibleCellIndex = address3(Int3(columnPos.x, columnPos.y, vc), vld.size);

        sumSoFar += vl.visibleActivations[visibleCellIndex];

        if (sumSoFar >= cusp) {
            selectIndex = vc;

            break;
        }
    }
    
    vl.visibleRandomCIs[visibleColumnIndex] = selectIndex;
}

void Layer::reconForward(
    const Int2 &columnPos,
    unsigned int* state
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        float sum = 0.0f;
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

            // Lower corner
            Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

            // Bounds of receptive field, clamped to input size
            Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
            Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    int inCI = vl.visibleRandomCIs[visibleColumnIndex];

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

    float total = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        hiddenActivations[hiddenCellIndex] = expf(hiddenActivations[hiddenCellIndex] - maxActivation);
        
        total += hiddenActivations[hiddenCellIndex];
    }

    float cusp = randf(state) * total;

    int selectIndex = 0;
    float sumSoFar = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        sumSoFar += hiddenActivations[hiddenCellIndex];

        if (sumSoFar >= cusp) {
            selectIndex = hc;

            break;
        }
    }
    
    hiddenRandomCIs[hiddenColumnIndex] = selectIndex;
}

void Layer::learn(
    const Int2 &columnPos,
    const Array<const IntBuffer*> &inputCIs
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hiddenCIs[hiddenColumnIndex]), hiddenSize);
    int hiddenRandomCellIndex = address3(Int3(columnPos.x, columnPos.y, hiddenRandomCIs[hiddenColumnIndex]), hiddenSize);

    // For each visible layer
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];
        
        assert(inputCIs[vli] != nullptr);

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
                int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                int inCI = (*inputCIs[vli])[visibleColumnIndex];
                int inRandomCI = vl.visibleRandomCIs[visibleColumnIndex];

                vl.weights[inCI + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex))] += alpha;
                vl.weights[inRandomCI + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenRandomCellIndex))] -= alpha;
            }
    }
}

void Layer::initRandom(
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

        for (int i = 0; i < vl.weights.size(); i++)
            vl.weights[i] = randf(-1.0f, 1.0f);

        vl.visibleActivations = FloatBuffer(numVisibleCells, 0.0f);

        vl.visibleCIs = IntBuffer(numVisibleColumns, 0);
        vl.visibleRandomCIs = IntBuffer(numVisibleColumns, 0);
    }

    hiddenActivations = FloatBuffer(numHiddenCells, 0.0f);

    // Hidden CIs
    hiddenCIs = IntBuffer(numHiddenColumns, 0);
    hiddenRandomCIs = IntBuffer(numHiddenColumns, 0);
}

void Layer::activate(
    const Array<const IntBuffer*> &inputCIs,
    bool generate
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    
    unsigned int baseState = rand();

    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++) {
        unsigned int state = baseState + i * 12345;

        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCIs, &state);
    }

    if (generate) {
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            if (inputCIs[vli] != nullptr)
                continue;

            int numVisibleColumns = vld.size.x * vld.size.y;
        
            #pragma omp parallel for
            for (int i = 0; i < numVisibleColumns; i++)
                backward(Int2(i / vld.size.y, i % vld.size.y), vli);
        }
    }
}

void Layer::learn(
    const Array<const IntBuffer*> &inputCIs
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    for (int it = 0; it < gibbsIters; it++) {
        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int numVisibleColumns = vld.size.x * vld.size.y;
        
            unsigned int baseState = rand();

            #pragma omp parallel for
            for (int i = 0; i < numVisibleColumns; i++) {
                unsigned int state = baseState + i * 12345;
                
                reconBackward(Int2(i / vld.size.y, i % vld.size.y), vli, &state);
            }
        }

        unsigned int baseState = rand();

        #pragma omp parallel for
        for (int i = 0; i < numHiddenColumns; i++) {
            unsigned int state = baseState + i * 12345;

            reconForward(Int2(i / hiddenSize.y, i % hiddenSize.y), &state);
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        learn(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCIs);
}

void Layer::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&alpha), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&gibbsIters), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hiddenRandomCIs[0]), hiddenRandomCIs.size() * sizeof(int));

    int numVisibleCellsLayers = visibleLayers.size();

    writer.write(reinterpret_cast<const void*>(&numVisibleCellsLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

        int weightsSize = vl.weights.size();

        writer.write(reinterpret_cast<const void*>(&weightsSize), sizeof(int));

        writer.write(reinterpret_cast<const void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));

        writer.write(reinterpret_cast<const void*>(&vl.visibleCIs[0]), vl.visibleCIs.size() * sizeof(int));
        writer.write(reinterpret_cast<const void*>(&vl.visibleRandomCIs[0]), vl.visibleRandomCIs.size() * sizeof(int));
    }
}

void Layer::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&alpha), sizeof(float));
    reader.read(reinterpret_cast<void*>(&gibbsIters), sizeof(int));

    hiddenCIs.resize(numHiddenColumns);
    hiddenRandomCIs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hiddenRandomCIs[0]), hiddenRandomCIs.size() * sizeof(int));

    hiddenActivations = FloatBuffer(numHiddenCells, 0.0f);

    int numVisibleCellsLayers = visibleLayers.size();

    reader.read(reinterpret_cast<void*>(&numVisibleCellsLayers), sizeof(int));

    visibleLayers.resize(numVisibleCellsLayers);
    visibleLayerDescs.resize(numVisibleCellsLayers);
    
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

        vl.visibleCIs.resize(numVisibleColumns);
        vl.visibleRandomCIs.resize(numVisibleColumns);

        reader.read(reinterpret_cast<void*>(&vl.visibleCIs[0]), vl.visibleCIs.size() * sizeof(int));
        reader.read(reinterpret_cast<void*>(&vl.visibleRandomCIs[0]), vl.visibleRandomCIs.size() * sizeof(int));

        vl.visibleActivations = FloatBuffer(numVisibleCells, 0.0f);
    }
}
