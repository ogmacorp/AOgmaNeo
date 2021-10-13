// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "RLAdapter.h"
#include <iostream>

using namespace aon;

void RLAdapter::init(
    const Int3 &hiddenSize,
    int maxSamples
) {
    this->hiddenSize = hiddenSize;

    // Pre-compute dimensions
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    numSamples = 0;
    samples.resize(maxSamples);

    goalCIs = IntBuffer(numHiddenColumns, 0);
}

void RLAdapter::step(
    const IntBuffer* hiddenCIs,
    float reward, 
    bool learnEnabled
) {
    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    
    // Check for similar states
    int minDifferenceIndex = -1;
    int minDifference = 999999;

    for (int s = 0; s < numSamples; s++) {
        int difference = 0;

        for (int i = 0; i < numHiddenColumns; i++)
            difference += ((*hiddenCIs)[i] != samples[s].hiddenCIs[i]);

        if (difference < minDifference || minDifferenceIndex == -1) {
            minDifference = difference;
            minDifferenceIndex = s;
        }

        if (learnEnabled) {
            float updateAmount = 1.0f - static_cast<float>(difference) / static_cast<float>(numHiddenColumns);

            updateAmount *= updateAmount * updateAmount * updateAmount;

            // Merge
            for (int i = 0; i < numHiddenColumns; i++) {
                if (randf() < updateAmount)
                    samples[minDifferenceIndex].hiddenCIs[i] = (*hiddenCIs)[i];
            }

            samples[minDifferenceIndex].reward += lr * updateAmount * (reward - samples[minDifferenceIndex].reward);
        }
    }

    if (learnEnabled && minDifference > minOverlap && numSamples < samples.size()) {
        samples.pushFront();

        if (numSamples < samples.size())
            numSamples++;

        samples[0].hiddenCIs = *hiddenCIs;
        samples[0].reward = reward;
    }

    // Goal is highest rewarding state
    int maxIndex = -1;
    float maxReward = -999999.0f;

    for (int s = 0; s < numSamples; s++) {
        if (samples[s].reward > maxReward || maxIndex == -1) {
            maxReward = samples[s].reward;
            maxIndex = s;
        }
    }

    goalCIs = samples[maxIndex].hiddenCIs;
}

int RLAdapter::size() const {
    int size = sizeof(Int3) + sizeof(float) + goalCIs.size() * sizeof(int);

    return size;
}

int RLAdapter::stateSize() const {
    return 0;
}

void RLAdapter::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));

    writer.write(reinterpret_cast<const void*>(&goalCIs[0]), goalCIs.size() * sizeof(int));
}

void RLAdapter::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int numHiddenColumns = hiddenSize.x * hiddenSize.y;
    int numHiddenCells = numHiddenColumns * hiddenSize.z;

    reader.read(reinterpret_cast<void*>(&lr), sizeof(float));

    goalCIs.resize(numHiddenColumns);

    reader.read(reinterpret_cast<void*>(&goalCIs[0]), goalCIs.size() * sizeof(int));
}

void RLAdapter::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&goalCIs[0]), goalCIs.size() * sizeof(int));
}

void RLAdapter::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&goalCIs[0]), goalCIs.size() * sizeof(int));
}
