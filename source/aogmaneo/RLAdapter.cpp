// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "RLAdapter.h"

using namespace aon;

void RLAdapter::initRandom(
    const Int3 &hiddenSize,
    int numGoals
) {
    this->hiddenSize = hiddenSize;
    this->numGoals = numGoals;

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    protos.resize(numGoals * numHiddenCells);

    for (int i = 0; i < protos.size(); i++)
        protos[i] = randf(0.0f, 1.0f);

    values.resize(numGoals);

    for (int i = 0; i < values.size(); i++)
        values[i] = randf(-0.001f, 0.001f);

    traces = FloatBuffer(values.size(), 0.0f);

    goalCIs = IntBuffer(numHiddenColumns, 0);

    rates = FloatBuffer(numGoals, 0.5f);
}

void RLAdapter::step(
    const IntBuffer* hiddenCIs,
    float reward, 
    bool learnEnabled
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    
    // Find goal at hiddenCIs
    int maxGoalIndex = -1;
    float maxActivation = -999999.0f;

    for (int g = 0; g < numGoals; g++) {
        float sum = 0.0f;

        for (int i = 0; i < numHiddenColumns; i++)
            sum += protos[(*hiddenCIs)[i] + hiddenSize.z * (i + numHiddenColumns * g)];

        if (sum > maxActivation || maxGoalIndex == -1) {
            maxActivation = sum;
            maxGoalIndex = g;
        }
    }

    // Learn
    if (learnEnabled) {
        float qTarget = (1.0f - discount) * reward + discount * values[maxGoalIndex];

        for (int g = 0; g < numGoals; g++) {
            values[g] += (qTarget - values[g]) * traces[g];

            float strength = expf(-falloff * abs(maxGoalIndex - g) / max(0.0001f, rates[g])) * rates[g];

            for (int i = 0; i < numHiddenColumns; i++) {
                for (int hc = 0; hc < hiddenSize.z; hc++) {
                    int wi = hc + hiddenSize.z * (i + numHiddenColumns * g); 

                    protos[wi] += strength * ((hc == (*hiddenCIs)[i]) - protos[wi]);
                }
            }

            rates[g] -= lr * strength;
        }
    }

    for (int g = 0; g < numGoals; g++)
        traces[g] += traceDecay * ((g == maxGoalIndex) - traces[g]);

    // Determine goal from protos and goalIndex
    for (int i = 0; i < numHiddenColumns; i++) {
        int maxIndex = -1;
        float maxP = -999999.0f;

        for (int hc = 0; hc < hiddenSize.z; hc++) {
            int wi = hc + hiddenSize.z * (i + numHiddenColumns * maxGoalIndex); 

            if (protos[wi] > maxP || maxIndex == -1) {
                maxP = protos[wi];
                maxIndex = hc;
            }
        }

        goalCIs[i] = maxIndex;
    }
}

int RLAdapter::size() const {
    int size = sizeof(Int3) + sizeof(int) + 4 * sizeof(float) + goalCIs.size() * sizeof(int);

    size += protos.size() * sizeof(float) + 3 * values.size() * sizeof(float);

    return size;
}

int RLAdapter::stateSize() const {
    return goalCIs.size() * sizeof(int) + traces.size() * sizeof(float);
}

void RLAdapter::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));
    writer.write(reinterpret_cast<const void*>(&numGoals), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&falloff), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&discount), sizeof(float));
    writer.write(reinterpret_cast<const void*>(&traceDecay), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&goalCIs[0]), goalCIs.size() * sizeof(int));

    writer.write(reinterpret_cast<const void*>(&protos[0]), protos.size() * sizeof(float));

    writer.write(reinterpret_cast<const void*>(&values[0]), values.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&traces[0]), traces.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&rates[0]), rates.size() * sizeof(float));
}

void RLAdapter::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));
    reader.read(reinterpret_cast<void*>(&numGoals), sizeof(int));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&lr), sizeof(float));
    reader.read(reinterpret_cast<void*>(&falloff), sizeof(float));
    reader.read(reinterpret_cast<void*>(&discount), sizeof(float));
    reader.read(reinterpret_cast<void*>(&traceDecay), sizeof(float));

    goalCIs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&goalCIs[0]), goalCIs.size() * sizeof(int));

    protos.resize(numGoals * numHiddenCells);

    reader.read(reinterpret_cast<void*>(&protos[0]), protos.size() * sizeof(float));

    values.resize(numGoals);
    traces.resize(numGoals);
    rates.resize(numGoals);

    reader.read(reinterpret_cast<void*>(&values[0]), values.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&traces[0]), traces.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&rates[0]), rates.size() * sizeof(float));
}

void RLAdapter::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&goalCIs[0]), goalCIs.size() * sizeof(int));

    writer.write(reinterpret_cast<const void*>(&traces[0]), traces.size() * sizeof(float));
}

void RLAdapter::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&goalCIs[0]), goalCIs.size() * sizeof(int));

    reader.read(reinterpret_cast<void*>(&traces[0]), traces.size() * sizeof(float));
}
