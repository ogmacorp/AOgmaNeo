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
    unsigned int* state
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    // --- Value ---

    float value = 0.0f;
    int count = 0;

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        value += vl.valueWeights.multiplyCIs(hiddenColumnIndex, *inputCIs[vli]);
        count += vl.valueWeights.count(hiddenColumnIndex);
    }

    value /= max(1, count);

    hiddenValues[hiddenColumnIndex] = value;

    // --- Action ---

    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        float sum = 0.0f;

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            sum += vl.actionWeights.multiplyCIs(hiddenCellIndex, *inputCIs[vli]);
        }

        sum /= max(1, count);

        hiddenActivations[hiddenCellIndex] = sum;

        maxActivation = max(maxActivation, sum);
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
    
    hiddenCIs[hiddenColumnIndex] = selectIndex;
}

void Actor::learn(
    const Int2 &columnPos,
    const Array<const IntBuffer*> &inputCIsPrev,
    const IntBuffer* hiddenTargetCIsPrev,
    const FloatBuffer* hiddenValuesPrev,
    float q,
    float g,
    bool mimic
) {
    int hiddenColumnIndex = address2(columnPos, Int2(hiddenSize.x, hiddenSize.y));

    // --- Value Prev ---

    float newValue = q + g * hiddenValues[hiddenColumnIndex];

    float value = 0.0f;
    int count = 0;

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        value += vl.valueWeights.multiplyCIs(hiddenColumnIndex, *inputCIsPrev[vli]);
        count += vl.valueWeights.count(hiddenColumnIndex);
    }

    value /= max(1, count);

    float tdErrorValue = newValue - value;
    
    float deltaValue = vlr * tdErrorValue;

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        vl.valueWeights.deltaCIs(hiddenColumnIndex, *inputCIsPrev[vli], deltaValue);
    }

    // --- Action ---

    float tdErrorAction = newValue - (*hiddenValuesPrev)[hiddenColumnIndex];

    int targetCI = (*hiddenTargetCIsPrev)[hiddenColumnIndex];

    float maxActivation = -999999.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        float sum = 0.0f;

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            sum += vl.actionWeights.multiplyCIs(hiddenCellIndex, *inputCIsPrev[vli]);
        }

        sum /= max(1, count);

        hiddenActivations[hiddenCellIndex] = sum;

        maxActivation = max(maxActivation, sum);
    }

    float total = 0.0f;

    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        hiddenActivations[hiddenCellIndex] = expf(hiddenActivations[hiddenCellIndex] - maxActivation);
        
        total += hiddenActivations[hiddenCellIndex];
    }
    
    for (int hc = 0; hc < hiddenSize.z; hc++) {
        int hiddenCellIndex = address3(Int3(columnPos.x, columnPos.y, hc), hiddenSize);

        float deltaAction = (mimic || tdErrorAction > 0.0f ? alr : -alr) * ((hc == targetCI ? 1.0f : 0.0f) - hiddenActivations[hiddenCellIndex] / max(0.0001f, total));

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayer &vl = visibleLayers[vli];
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            vl.actionWeights.deltaCIs(hiddenCellIndex, *inputCIsPrev[vli], deltaAction);
        }
    }
}

void Actor::initRandom(
    const Int3 &hiddenSize,
    int historyCapacity,
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

        // Create weight matrix for this visible layer and initialize randomly
        vl.valueWeights.init(vld.size, Int3(hiddenSize.x, hiddenSize.y, 1), vld.radius);
        vl.actionWeights.init(vld.size, hiddenSize, vld.radius);

        for (int i = 0; i < vl.valueWeights.values.size(); i++)
            vl.valueWeights.values[i] = randf(-0.01f, 0.01f);

        for (int i = 0; i < vl.actionWeights.values.size(); i++)
            vl.actionWeights.values[i] = randf(-0.01f, 0.01f);
    }

    hiddenActivations = FloatBuffer(numHiddenCells, 0.0f);

    hiddenCIs = IntBuffer(numHiddenColumns, 0);

    hiddenValues = FloatBuffer(numHiddenColumns, 0.0f);

    // Create (pre-allocated) history samples
    historySize = 0;
    historySamples.resize(historyCapacity);

    for (int i = 0; i < historySamples.size(); i++) {
        historySamples[i].inputCIs.resize(visibleLayers.size());

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            VisibleLayerDesc &vld = this->visibleLayerDescs[vli];

            int numVisibleColumns = vld.size.x * vld.size.y;

            historySamples[i].inputCIs[vli] = IntBuffer(numVisibleColumns);
        }

        historySamples[i].hiddenTargetCIsPrev = IntBuffer(numHiddenColumns);

        historySamples[i].hiddenValuesPrev = FloatBuffer(numHiddenColumns);
    }
}

void Actor::step(
    const Array<const IntBuffer*> &inputCIs,
    const IntBuffer* hiddenTargetCIsPrev,
    float reward,
    bool learnEnabled,
    bool mimic
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;

    // Forward kernel
    unsigned int baseState = rand();

    #pragma omp parallel for
    for (int i = 0; i < numHiddenColumns; i++) {
        unsigned int state = baseState + i * 12345;

        forward(Int2(i / hiddenSize.y, i % hiddenSize.y), inputCIs, &state);
    }

    historySamples.pushFront();

    // If not at cap, increment
    if (historySize < historySamples.size())
        historySize++;
    
    // Add new sample
    {
        HistorySample &s = historySamples[0];

        for (int vli = 0; vli < visibleLayers.size(); vli++)
            s.inputCIs[vli] = *inputCIs[vli];

        // Copy hidden CIs
        s.hiddenTargetCIsPrev = *hiddenTargetCIsPrev;

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

                g *= discount;
            }

            #pragma omp parallel for
            for (int i = 0; i < numHiddenColumns; i++)
                learn(Int2(i / hiddenSize.y, i % hiddenSize.y), constGet(sPrev.inputCIs), &s.hiddenTargetCIsPrev, &sPrev.hiddenValuesPrev, q, g, mimic);
        }
    }
}

int Actor::size() const {
    int size = sizeof(Int3) + 3 * sizeof(float) + 2 * sizeof(int) + hiddenCIs.size() * sizeof(int) + hiddenValues.size() * sizeof(float) + sizeof(int);

    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        size += sizeof(VisibleLayerDesc) + vl.valueWeights.size() * sizeof(float) + vl.actionWeights.size() * sizeof(float);
    }

    size += 3 * sizeof(int);

    int sampleSize = 0;

    const HistorySample &s = historySamples[0];

    for (int vli = 0; vli < visibleLayers.size(); vli++)
        sampleSize += s.inputCIs[vli].size() * sizeof(int);

    sampleSize += s.hiddenTargetCIsPrev.size() * sizeof(int) + s.hiddenValuesPrev.size() * sizeof(float) + sizeof(float);

    size += historySamples.size() * sampleSize;

    return size;
}

int Actor::stateSize() const {
    int size = hiddenCIs.size() * sizeof(int) + hiddenValues.size() * sizeof(float) + sizeof(int);

    int sampleSize = 0;

    const HistorySample &s = historySamples[0];

    for (int vli = 0; vli < visibleLayers.size(); vli++)
        sampleSize += s.inputCIs[vli].size() * sizeof(int);

    sampleSize += s.hiddenTargetCIsPrev.size() * sizeof(int) + s.hiddenValuesPrev.size() * sizeof(float) + sizeof(float);

    size += historySamples.size() * sampleSize;

    return size;
}

void Actor::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&vlr), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&alr), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&discount), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&minSteps), sizeof(int));
    writer.write(reinterpret_cast<const void*>(&historyIters), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hiddenValues[0]), hiddenValues.size() * sizeof(float));

    int numVisibleLayers = visibleLayers.size();

    writer.write(reinterpret_cast<const void*>(&numVisibleLayers), sizeof(int));
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        const VisibleLayer &vl = visibleLayers[vli];
        const VisibleLayerDesc &vld = visibleLayerDescs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(VisibleLayerDesc));

        vl.valueWeights.write(writer);
        vl.actionWeights.write(writer);
    }

    writer.write(reinterpret_cast<const void*>(&historySize), sizeof(int));

    int numHistorySamples = historySamples.size();

    writer.write(reinterpret_cast<const void*>(&numHistorySamples), sizeof(int));

    int historyStart = historySamples.start;

    writer.write(reinterpret_cast<const void*>(&historyStart), sizeof(int));

    for (int t = 0; t < historySamples.size(); t++) {
        const HistorySample &s = historySamples[t];

        for (int vli = 0; vli < visibleLayers.size(); vli++)
            writer.write(reinterpret_cast<const void*>(&s.inputCIs[vli][0]), s.inputCIs[vli].size() * sizeof(int));

        writer.write(reinterpret_cast<const void*>(&s.hiddenTargetCIsPrev[0]), s.hiddenTargetCIsPrev.size() * sizeof(int));
        writer.write(reinterpret_cast<const void*>(&s.hiddenValuesPrev[0]), s.hiddenValuesPrev.size() * sizeof(float));

        writer.write(reinterpret_cast<const void*>(&s.reward), sizeof(float));
    }
}

void Actor::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;
    
    reader.read(reinterpret_cast<void*>(&vlr), sizeof(float));
    reader.read(reinterpret_cast<void*>(&alr), sizeof(float));
    reader.read(reinterpret_cast<void*>(&discount), sizeof(float));
    reader.read(reinterpret_cast<void*>(&minSteps), sizeof(int));
    reader.read(reinterpret_cast<void*>(&historyIters), sizeof(int));

    hiddenCIs.resize(numHiddenColumns);
    hiddenValues.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hiddenValues[0]), hiddenValues.size() * sizeof(float));

    hiddenActivations = FloatBuffer(numHiddenCells, 0.0f);
    
    int numVisibleLayers = visibleLayers.size();

    reader.read(reinterpret_cast<void*>(&numVisibleLayers), sizeof(int));

    visibleLayers.resize(numVisibleLayers);
    visibleLayerDescs.resize(numVisibleLayers);
    
    for (int vli = 0; vli < visibleLayers.size(); vli++) {
        VisibleLayer &vl = visibleLayers[vli];
        VisibleLayerDesc &vld = visibleLayerDescs[vli];

        reader.read(reinterpret_cast<void*>(&vld), sizeof(VisibleLayerDesc));

        vl.valueWeights.read(reader);
        vl.actionWeights.read(reader);
    }

    reader.read(reinterpret_cast<void*>(&historySize), sizeof(int));

    int numHistorySamples;

    reader.read(reinterpret_cast<void*>(&numHistorySamples), sizeof(int));

    int historyStart;

    reader.read(reinterpret_cast<void*>(&historyStart), sizeof(int));

    historySamples.resize(numHistorySamples);
    historySamples.start = historyStart;

    for (int t = 0; t < historySamples.size(); t++) {
        HistorySample &s = historySamples[t];

        s.inputCIs.resize(numVisibleLayers);

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            int numVisibleColumns = vld.size.x * vld.size.y;

            s.inputCIs[vli].resize(numVisibleColumns);

            reader.read(reinterpret_cast<void*>(&s.inputCIs[vli][0]), s.inputCIs[vli].size() * sizeof(int));
        }

        s.hiddenTargetCIsPrev.resize(numHiddenColumns);
        s.hiddenValuesPrev.resize(numHiddenColumns);

        reader.read(reinterpret_cast<void*>(&s.hiddenTargetCIsPrev[0]), s.hiddenTargetCIsPrev.size() * sizeof(int));
        reader.read(reinterpret_cast<void*>(&s.hiddenValuesPrev[0]), s.hiddenValuesPrev.size() * sizeof(float));

        reader.read(reinterpret_cast<void*>(&s.reward), sizeof(float));
    }
}

void Actor::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hiddenValues[0]), hiddenValues.size() * sizeof(float));

    int historyStart = historySamples.start;

    writer.write(reinterpret_cast<const void*>(&historyStart), sizeof(int));

    for (int t = 0; t < historySamples.size(); t++) {
        const HistorySample &s = historySamples[t];

        for (int vli = 0; vli < visibleLayers.size(); vli++)
            writer.write(reinterpret_cast<const void*>(&s.inputCIs[vli][0]), s.inputCIs[vli].size() * sizeof(int));

        writer.write(reinterpret_cast<const void*>(&s.hiddenTargetCIsPrev[0]), s.hiddenTargetCIsPrev.size() * sizeof(int));
        writer.write(reinterpret_cast<const void*>(&s.hiddenValuesPrev[0]), s.hiddenValuesPrev.size() * sizeof(float));

        writer.write(reinterpret_cast<const void*>(&s.reward), sizeof(float));
    }
}

void Actor::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenCIs[0]), hiddenCIs.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hiddenValues[0]), hiddenValues.size() * sizeof(float));

    int historyStart;

    reader.read(reinterpret_cast<void*>(&historyStart), sizeof(int));

    historySamples.start = historyStart;

    for (int t = 0; t < historySamples.size(); t++) {
        HistorySample &s = historySamples[t];

        for (int vli = 0; vli < visibleLayers.size(); vli++) {
            const VisibleLayerDesc &vld = visibleLayerDescs[vli];

            reader.read(reinterpret_cast<void*>(&s.inputCIs[vli][0]), s.inputCIs[vli].size() * sizeof(int));
        }

        reader.read(reinterpret_cast<void*>(&s.hiddenTargetCIsPrev[0]), s.hiddenTargetCIsPrev.size() * sizeof(int));
        reader.read(reinterpret_cast<void*>(&s.hiddenValuesPrev[0]), s.hiddenValuesPrev.size() * sizeof(float));

        reader.read(reinterpret_cast<void*>(&s.reward), sizeof(float));
    }
}
