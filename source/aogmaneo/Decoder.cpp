// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Decoder.h"

using namespace aon;

void Decoder::forward(
    const Int2 &columnPos,
    const Array<const IntBuffer*> &inputCIs
) {
    int hiddenColumnIndex = address2(columnPos, Int2(outputSize.x, outputSize.y));

    int numCellsPerColumn = outputSize.z * numDendrites;

    int hiddenCellsStart = hiddenColumnIndex * numCellsPerColumn;

    int maxIndex = -1;
    float maxActivation = -999999.0f;

    for (int hc = 0; hc < numCellsPerColumn; hc++) {
        int hiddenCellIndex = hc + hiddenCellsStart;

        float sum = 0.0f;

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int diam = vld.radius * 2 + 1;

            // Projection
            Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(outputSize.x),
                static_cast<float>(vld.size.y) / static_cast<float>(outputSize.y));

            Int2 visibleCenter = project(columnPos, hToV);

            // Lower corner
            Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

            // Bounds of receptive field, clamped to input size
            Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
            Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                    int inCI = (*inputCIs[vli])[visibleColumnIndex];

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                    sum += logf(max(0.0001f, vl.protos[inCI + wiStart]));

                    for (int vc = 0; vc < vld.size.z; vc++) {
                        if (vc == inCI)
                            continue;

                        int wi = vc + wiStart;

                        sum += logf(max(0.0001f, 1.0f - vl.protos[wi]));
                    }
                }
        }

        hiddenActivations[hiddenCellIndex] = sum;

        if (sum > maxActivation || maxIndex == -1) {
            maxActivation = sum;
            maxIndex = hc;
        }
    }

    outputCIs[hiddenColumnIndex] = maxIndex / numDendrites;
}

void Decoder::learn(
    const Int2 &columnPos,
    const IntBuffer* hiddenTargetCIs
) {
    int hiddenColumnIndex = address2(columnPos, Int2(outputSize.x, outputSize.y));

    int numCellsPerColumn = outputSize.z * numDendrites;

    int hiddenCellsStart = hiddenColumnIndex * numCellsPerColumn;

    int targetCI = (*hiddenTargetCIs)[hiddenColumnIndex];

    // Select strongest dendrite
    int maxDendriteIndex = -1;
    float maxDendriteActivation = -999999.0f;

    for (int di = 0; di < numDendrites; di++) {
        int hiddenCellIndex = (targetCI * numDendrites + di) + hiddenCellsStart;

        if (hiddenActivations[hiddenCellIndex] > maxDendriteActivation || maxDendriteIndex == -1) {
            maxDendriteActivation = hiddenActivations[hiddenCellIndex];
            maxDendriteIndex = di;
        }
    }

    for (int di = 0; di < numDendrites; di++) {
        int hiddenCellIndex = (targetCI * numDendrites + di) + hiddenCellsStart;

        float rate = (di == maxDendriteIndex ? lr : boost);

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int diam = vld.radius * 2 + 1;

            // Projection
            Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(outputSize.x),
                static_cast<float>(vld.size.y) / static_cast<float>(outputSize.y));

            Int2 visibleCenter = project(columnPos, hToV);

            // Lower corner
            Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

            // Bounds of receptive field, clamped to input size
            Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
            Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                    int inCIPrev = vl.inputCIsPrev[visibleColumnIndex];

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                    for (int vc = 0; vc < vld.size.z; vc++) {
                        int wi = vc + wiStart;

                        vl.protos[wi] += rate * ((vc == inCIPrev) - vl.protos[wi]);
                    }
                }
        }
    }
}

void Decoder::initRandom(
    const Int3 &outputSize,
    int numDendrites,
    const Array<VisibleLayerDesc> &visibleLayerDescs
) {
    this->visibleLayerDescs = visibleLayerDescs; 

    this->outputSize = outputSize;
    this->numDendrites = numDendrites;

    visibleLayers.resize(visibleLayerDescs.size());

    // Pre-compute dimensions
    int numHiddenColumns = outputSize.x * outputSize.y;
    int numHiddenCells = numHiddenColumns * outputSize.z * numDendrites;
    
    // Create layers
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(numHiddenCells * area * vld.size.z);
        vl.protos.resize(vl.weights.size());

        for (int i = 0; i < vl.weights.size(); i++) {
            vl.weights[i] = randf(-0.01f, 0.01f);
            vl.protos[i] = randf(0.0f, 1.0f);
        }

        vl.inputCIsPrev = IntBuffer(numVisibleColumns, 0);
    }

    hiddenActivations = FloatBuffer(numHiddenCells, 0.0f);

    outputCIs = IntBuffer(numHiddenColumns, 0);
}

void Decoder::activate(
    const Array<const IntBuffer*> &inputCIs
) {
    int numHiddenColumns = outputSize.x * outputSize.y;

    // Forward kernel
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        forward(Int2(i / outputSize.y, i % outputSize.y), inputCIs);

    // Copy to prevs
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];

        vl.inputCIsPrev = *inputCIs[vli];
    }
}

void Decoder::learn(
    const IntBuffer* hiddenTargetCIs
) {
    int numHiddenColumns = outputSize.x * outputSize.y;
    
    // Learn kernel
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        learn(Int2(i / outputSize.y, i % outputSize.y), hiddenTargetCIs);
}

int Decoder::size() const {
    int size = sizeof(Int3) + sizeof(int) + 2 * sizeof(float) + hiddenActivations.size() * sizeof(float) + outputCIs.size() * sizeof(int) + sizeof(int);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        size += sizeof(VisibleLayerDesc) + 2 * vl.weights.size() * sizeof(float) + vl.inputCIsPrev.size() * sizeof(int);
    }

    return size;
}

int Decoder::stateSize() const {
    int size = hiddenActivations.size() * sizeof(float) + outputCIs.size() * sizeof(int);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];

        size += vl.inputCIsPrev.size() * sizeof(int);
    }

    return size;
}

void Decoder::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&outputSize), sizeof(Int3));
    writer.write(reinterpret_cast<const void*>(&numDendrites), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&boost), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenActivations[0]), hiddenActivations.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&outputCIs[0]), outputCIs.size() * sizeof(int));
    
    int numVisibleLayers = visibleLayers.size();

    writer.write(reinterpret_cast<const void*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

        writer.write(reinterpret_cast<const void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));

        writer.write(reinterpret_cast<const void*>(&vl.inputCIsPrev[0]), vl.inputCIsPrev.size() * sizeof(int));
    }
}

void Decoder::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&outputSize), sizeof(Int3));
    reader.read(reinterpret_cast<void*>(&numDendrites), sizeof(int));

    int numHiddenColumns = outputSize.x * outputSize.y;
    int numHiddenCells = numHiddenColumns * outputSize.z * numDendrites;

    reader.read(reinterpret_cast<void*>(&lr), sizeof(float));
    reader.read(reinterpret_cast<void*>(&boost), sizeof(float));

    hiddenActivations.resize(numHiddenCells);
    outputCIs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenActivations[0]), hiddenActivations.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&outputCIs[0]), outputCIs.size() * sizeof(int));

    int numVisibleLayers = visibleLayers.size();

    reader.read(reinterpret_cast<void*>(&numVisibleLayers), sizeof(int));

    visibleLayers.resize(numVisibleLayers);
    visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = visibleLayerDescs[vli];

        reader.read(reinterpret_cast<void*>(&vld), sizeof(VisibleLayerDesc));

        int numVisibleColumns = vld.size.x * vld.size.y;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(numHiddenCells * area * vld.size.z);
        vl.protos.resize(vl.weights.size());

        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));
        reader.read(reinterpret_cast<void*>(&vl.protos[0]), vl.protos.size() * sizeof(float));

        vl.inputCIsPrev.resize(numVisibleColumns);

        reader.read(reinterpret_cast<void*>(&vl.inputCIsPrev[0]), vl.inputCIsPrev.size() * sizeof(int));
    }
}

void Decoder::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenActivations[0]), hiddenActivations.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&outputCIs[0]), outputCIs.size() * sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];

        writer.write(reinterpret_cast<const void*>(&vl.inputCIsPrev[0]), vl.inputCIsPrev.size() * sizeof(int));
    }
}

void Decoder::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenActivations[0]), hiddenActivations.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&outputCIs[0]), outputCIs.size() * sizeof(int));

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];

        reader.read(reinterpret_cast<void*>(&vl.inputCIsPrev[0]), vl.inputCIsPrev.size() * sizeof(int));
    }
}
