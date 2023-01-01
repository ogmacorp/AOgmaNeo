// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "LocationInvariant.h"

using namespace aon;

void LocationInvariant::initRandom(
    const Int3 &hiddenSize,
    const Int3 &sensorSize,
    const Int3 &whereSize,
    int radius
) {
    this->sensorSize = sensorSize;
    this->whereSize = whereSize;

    int numSensorCells = sensorSize.x * sensorSize.y * sensorSize.z; 
    int numWhereCells = whereSize.x * whereSize.y * whereSize.z; 

    memory = ByteBuffer(numWhereCells * numSensorCells, 0);

    Array<ImageEncoder::VisibleLayerDesc> visibleLayerDescs(1);

    visibleLayerDescs[0].size = Int3(numWhereCells, numSensorCells, 1);
    visibleLayerDescs[0].radius = radius;

    enc.initRandom(hiddenSize, visibleLayerDescs);
}

void LocationInvariant::step(
    const IntBuffer* sensorCIs,
    const IntBuffer* whereCIs,
    bool learnEnabled
) {
    int numSensorColumns = sensorSize.x * sensorSize.y;
    int numWhereColumns = whereSize.x * whereSize.y; 

    int numSensorCells = numSensorColumns * sensorSize.z; 

    // Decay
    for (int i = 0; i < memory.size(); i++)
        memory[i] = max(0, static_cast<int>(memory[i] * (1.0f - decay)));

    // Update memory
    for (int i = 0; i < numWhereColumns; i++) {
        int whereCI = (*whereCIs)[i];

        int whereCellIndex = whereCI + i * whereSize.z;

        int sensorCellsStart = numSensorCells * whereCellIndex;

        for (int j = 0; j < numSensorColumns; j++) {
            int sensorCI = (*sensorCIs)[j];

            int sensorCellIndex = sensorCI + j * sensorSize.z;

            int mi = sensorCellIndex + sensorCellsStart;

            memory[mi] = 255;
        }
    }

    // Learn on memory
    Array<const ByteBuffer*> inputs(1);

    inputs[0] = &memory;

    enc.step(inputs, learnEnabled);
}

int LocationInvariant::size() const {
    return 2 * sizeof(Int3) + sizeof(float) + enc.size() + memory.size() * sizeof(Byte);
}

int LocationInvariant::stateSize() const {
    return memory.size() * sizeof(Byte);
}

void LocationInvariant::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&sensorSize), sizeof(Int3));
    writer.write(reinterpret_cast<const void*>(&whereSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&decay), sizeof(float));

    enc.write(writer);
    
    writer.write(reinterpret_cast<const void*>(&memory[0]), memory.size() * sizeof(Byte));
}

void LocationInvariant::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&sensorSize), sizeof(Int3));
    reader.read(reinterpret_cast<void*>(&whereSize), sizeof(Int3));

    int numSensorCells = sensorSize.x * sensorSize.y * sensorSize.z; 
    int numWhereCells = whereSize.x * whereSize.y * whereSize.z; 

    reader.read(reinterpret_cast<void*>(&decay), sizeof(float));

    enc.read(reader);

    memory.resize(numWhereCells * numSensorCells);

    reader.read(reinterpret_cast<void*>(&memory[0]), memory.size() * sizeof(Byte));
}

void LocationInvariant::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&memory[0]), memory.size() * sizeof(Byte));
}

void LocationInvariant::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&memory[0]), memory.size() * sizeof(Byte));
}
