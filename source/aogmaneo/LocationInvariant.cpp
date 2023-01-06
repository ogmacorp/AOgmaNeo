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

    int numSensorColumns = sensorSize.x * sensorSize.y;
    int numSensorCells = numSensorColumns * sensorSize.z; 
    int numWhereCells = whereSize.x * whereSize.y * whereSize.z; 

    memoryActs = FloatBuffer(numWhereCells * numSensorCells, 0.0f);
    memoryCIs = IntBuffer(numWhereCells * numSensorColumns, 0); 

    Array<Encoder::VisibleLayerDesc> visibleLayerDescs(1);

    visibleLayerDescs[0].size = Int3(numWhereCells, numSensorColumns, sensorSize.z);
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

    // Update memory
    for (int i = 0; i < numWhereColumns; i++) {
        int whereCI = (*whereCIs)[i];

        int whereCellIndex = whereCI + i * whereSize.z;

        int sensorCellsStart = numSensorCells * whereCellIndex;

        for (int j = 0; j < numSensorColumns; j++) {
            int sensorCI = (*sensorCIs)[j];

            int maxIndex = -1;
            float maxActivation = -999999.0f;

            for (int k = 0; k < sensorSize.z; k++) {
                int sensorCellIndex = k + j * sensorSize.z;

                int mi = sensorCellIndex + sensorCellsStart;

                if (k == sensorCI)
                    memoryActs[mi] += 1.0f;
                else
                    memoryActs[mi] *= 1.0f - decay;

                if (memoryActs[mi] > maxActivation || maxIndex == -1) {
                    maxActivation = memoryActs[mi];
                    maxIndex = k;
                }
            }

            memoryCIs[j + numSensorColumns * whereCellIndex] = maxIndex;
        }
    }

    // Learn on memory
    Array<const IntBuffer*> inputs(1);

    inputs[0] = &memoryCIs;

    enc.step(inputs, learnEnabled);
}

int LocationInvariant::size() const {
    return 2 * sizeof(Int3) + sizeof(float) + enc.size() + memoryActs.size() * sizeof(float) + memoryCIs.size() * sizeof(int);
}

int LocationInvariant::stateSize() const {
    return memoryActs.size() * sizeof(float) + memoryCIs.size() * sizeof(int);
}

void LocationInvariant::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&sensorSize), sizeof(Int3));
    writer.write(reinterpret_cast<const void*>(&whereSize), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&decay), sizeof(float));

    enc.write(writer);
    
    writer.write(reinterpret_cast<const void*>(&memoryActs[0]), memoryActs.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&memoryCIs[0]), memoryCIs.size() * sizeof(int));
}

void LocationInvariant::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&sensorSize), sizeof(Int3));
    reader.read(reinterpret_cast<void*>(&whereSize), sizeof(Int3));

    int numSensorColumns = sensorSize.x * sensorSize.y;
    int numWhereColumns = whereSize.x * whereSize.y; 

    int numSensorCells = numSensorColumns * sensorSize.z; 
    int numWhereCells = numWhereColumns * whereSize.z; 

    reader.read(reinterpret_cast<void*>(&decay), sizeof(float));

    enc.read(reader);

    memoryActs.resize(numWhereCells * numSensorCells);
    memoryCIs.resize(numWhereCells * numSensorColumns);

    reader.read(reinterpret_cast<void*>(&memoryActs[0]), memoryActs.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&memoryCIs[0]), memoryCIs.size() * sizeof(int));
}

void LocationInvariant::writeState(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&memoryActs[0]), memoryActs.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&memoryCIs[0]), memoryCIs.size() * sizeof(int));
}

void LocationInvariant::readState(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&memoryActs[0]), memoryActs.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&memoryCIs[0]), memoryCIs.size() * sizeof(int));
}
