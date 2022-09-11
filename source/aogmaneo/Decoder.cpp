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
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenBranchesStart = hiddenColumnIndex * hiddenSize.z;

    int superMaxIndex = 0;
    float superMaxActivation = 0.0f;
    
    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenBranchIndex = hc + hiddenBranchesStart;

        int hiddenCellsStart = hiddenBranchIndex * numDendrites;

        int maxIndex = -1;
        float maxActivation = 0.0f;

        int backupMaxIndex = -1;
        float backupMaxActivation = 0.0f;

        for (int di = 0; di < hiddenCommits[hiddenBranchIndex]; di++) {
            int hiddenCellIndex = di + hiddenCellsStart;

            float sum = 0.0f;
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

                count += (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1);

                for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                    for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                        int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                        int inCI = (*inputCIs[vli])[visibleColumnIndex];

                        Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                        int wi = inCI + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndex));

                        sum += vl.weights[wi];
                    }
            }

            sum /= count;

            float activation = sum / (gap + hiddenTotals[hiddenCellIndex]);
            float match = sum;

            if (match >= vigilance) {
                if (activation > maxActivation || maxIndex == -1) {
                    maxActivation = activation;
                    maxIndex = di;
                }
            }

            if (activation > backupMaxActivation || backupMaxIndex == -1) {
                backupMaxActivation = activation;
                backupMaxIndex = di;
            }
        }

        hiddenModes[hiddenBranchIndex] = update;

        bool found = maxIndex != -1;

        if (!found) {
            if (hiddenCommits[hiddenBranchIndex] >= numDendrites) {
                maxIndex = backupMaxIndex;
                maxActivation = backupMaxActivation;
            }
            else {
                maxIndex = hiddenCommits[hiddenBranchIndex];
                hiddenModes[hiddenBranchIndex] = commit;
            }
        }

        if (maxActivation > superMaxActivation || superMaxIndex == -1) {
            superMaxActivation = maxActivation;
            superMaxIndex = hc;
        }

        hiddenDIs[hiddenBranchIndex] = maxIndex;
    }

    hiddenCIs[hiddenColumnIndex] = superMaxIndex;
}

void Decoder::learn(
    const Int2 &columnPos,
    const IntBuffer* hiddenTargetCIs
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    int hiddenBranchesStart = hiddenColumnIndex * hiddenSize.z;

    int targetCI = (*hiddenTargetCIs)[hiddenColumnIndex];

    if (hiddenCIs[hiddenColumnIndex] == targetCI)
        return;

    {
        int hiddenBranchIndexTarget = targetCI + hiddenBranchesStart;

        int hiddenCellsStartTarget = hiddenBranchIndexTarget * numDendrites;

        int hiddenCellIndexTarget = hiddenDIs[hiddenBranchIndexTarget] + hiddenCellsStartTarget;

        float total = 0.0f;
        int count = 0;

        if (hiddenModes[hiddenBranchIndexTarget] == commit) {
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

                count += (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1) * vld.size.z;

                for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                    for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                        int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                        int inCIPrev = vl.inputCIsPrev[visibleColumnIndex];

                        Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                        int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndexTarget));

                        for (int vc = 0; vc < vld.size.z; vc++) {
                            int wi = vc + wiStart;

                            vl.weights[wi] = 1.0f - lr * (vc != inCIPrev);

                            total += vl.weights[wi];
                        }
                    }
            }

            if (hiddenCommits[hiddenBranchIndexTarget] < numDendrites)
                hiddenCommits[hiddenBranchIndexTarget]++;
        }
        else {
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

                count += (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1) * vld.size.z;

                for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                    for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                        int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                        int inCIPrev = vl.inputCIsPrev[visibleColumnIndex];

                        Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                        int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndexTarget));

                        for (int vc = 0; vc < vld.size.z; vc++) {
                            int wi = vc + wiStart;

                            vl.weights[wi] += lr * ((vc == inCIPrev) - vl.weights[wi]);

                            total += vl.weights[wi];
                        }
                    }
            }
        }

        total /= count;

        hiddenTotals[hiddenCellIndexTarget] = total;
    }

    {
        int hiddenBranchIndexMax = hiddenCIs[hiddenColumnIndex] + hiddenBranchesStart;

        int hiddenCellsStartMax = hiddenBranchIndexMax * numDendrites;

        int hiddenCellIndexMax = hiddenDIs[hiddenBranchIndexMax] + hiddenCellsStartMax;

        float total = 0.0f;
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

            count += (iterUpperBound.x - iterLowerBound.x + 1) * (iterUpperBound.y - iterLowerBound.y + 1) * vld.size.z;

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x,  vld.size.y));

                    int inCIPrev = vl.inputCIsPrev[visibleColumnIndex];

                    Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    int wiStart = vld.size.z * (offset.y + diam * (offset.x + diam * hiddenCellIndexMax));

                    vl.weights[inCIPrev + wiStart] -= lr * vl.weights[inCIPrev + wiStart];

                    for (int vc = 0; vc < vld.size.z; vc++) {
                        int wi = vc + wiStart;

                        total += vl.weights[wi];
                    }
                }
        }

        total /= count;

        hiddenTotals[hiddenCellIndexMax] = total;
    }
}

void Decoder::initRandom(
    const Int3 &hiddenSize,
    int numDendrites,
    const Array<VisibleLayerDesc> &visibleLayerDescs
) {
    this->visibleLayerDescs = visibleLayerDescs; 

    this->hiddenSize = hiddenSize;
    this->numDendrites = numDendrites;

    visibleLayers.resize(visibleLayerDescs.size());

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenBranches = numHiddenColumns * hiddenSize.z;
    int numHiddenCells = numHiddenBranches * numDendrites;
    
    // Create layers
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

        int numVisibleColumns = vld.size.x * vld.size.y;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(numHiddenCells * area * vld.size.z);

        vl.inputCIsPrev = IntBuffer(numVisibleColumns, 0);
    }

    hiddenModes = Array<Mode>(numHiddenBranches);

    hiddenDIs = IntBuffer(numHiddenBranches, 0);
    hiddenCIs = IntBuffer(numHiddenColumns, 0);

    hiddenCommits = IntBuffer(numHiddenBranches, 0);

    hiddenTotals = FloatBuffer(numHiddenCells);
}

void Decoder::activate(
    const Array<const IntBuffer*> &inputCIs
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    // Forward kernel
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCIs);

    // Copy to prevs
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];

        vl.inputCIsPrev = *inputCIs[vli];
    }
}

void Decoder::learn(
    const IntBuffer* hiddenTargetCIs
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    
    // Learn kernel
    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++)
        learn(Int2(i / hiddenSize.y, i % hiddenSize.y), hiddenTargetCIs);
}

int Decoder::size() const {
    int size = sizeof(Int3) + sizeof(int) + 2 * sizeof(float) + hiddenDIs.size() * sizeof(int) + 2 * hiddenCIs.size() * sizeof(int) + hiddenTotals.size() * sizeof(float) + sizeof(int);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        size += sizeof(VisibleLayerDesc) + vl.weights.size() * sizeof(float) + vl.inputCIsPrev.size() * sizeof(int);
    }

    return size;
}

int Decoder::stateSize() const {
    int size = hiddenDIs.size() * sizeof(int) + hiddenCIs.size() * sizeof(int);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];

        size += vl.inputCIsPrev.size() * sizeof(int);
    }

    return size;
}

void Decoder::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));
    writer.write(reinterpret_cast<const void*>(&numDendrites), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&vigilance), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hiddenDIs[0]), hiddenDIs.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    writer.write(reinterpret_cast<const void*>(&hiddenCommits[0]), hiddenCommits.size() * sizeof(int));

    writer.write(reinterpret_cast<const void*>(&hiddenTotals[0]), hiddenTotals.size() * sizeof(float));
    
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
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));
    reader.read(reinterpret_cast<void*>(&numDendrites), sizeof(int));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenBranches = numHiddenColumns * hiddenSize.z;
    int numHiddenCells = numHiddenBranches * numDendrites;

    reader.read(reinterpret_cast<void*>(&vigilance), sizeof(float));
    reader.read(reinterpret_cast<void*>(&lr), sizeof(float));

    hiddenModes = Array<Mode>(numHiddenBranches);

    hiddenDIs.resize(numHiddenBranches);
    hiddenCIs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenDIs[0]), hiddenDIs.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    hiddenCommits.resize(numHiddenBranches);

    reader.read(reinterpret_cast<void*>(&hiddenCommits[0]), hiddenCommits.size() * sizeof(int));

    hiddenTotals.resize(numHiddenCells);

    reader.read(reinterpret_cast<void*>(&hiddenTotals[0]), hiddenTotals.size() * sizeof(float));

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

        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));

        vl.inputCIsPrev.resize(numVisibleColumns);

        reader.read(reinterpret_cast<void*>(&vl.inputCIsPrev[0]), vl.inputCIsPrev.size() * sizeof(int));
    }
}

void Decoder::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenDIs[0]), hiddenDIs.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];

        writer.write(reinterpret_cast<const void*>(&vl.inputCIsPrev[0]), vl.inputCIsPrev.size() * sizeof(int));
    }
}

void Decoder::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenDIs[0]), hiddenDIs.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];

        reader.read(reinterpret_cast<void*>(&vl.inputCIsPrev[0]), vl.inputCIsPrev.size() * sizeof(int));
    }
}
