// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Helpers.h"

#include "Encoder.h"

namespace aon {
class LocationInvariant {
private:
    Int3 sensorSize;
    Int3 whereSize;

    Encoder enc;

    // Fast associative memory
    FloatBuffer memoryActs;
    IntBuffer memoryCIs; // Inhibited

public:
    float decay;

    LocationInvariant()
    :
    decay(0.1f)
    {}

    void initRandom(
        const Int3 &hiddenSize,
        const Int3 &sensorSize,
        const Int3 &whereSize,
        int radius // Radius of hidden onto memory
    );

    void step(
        const IntBuffer* sensorCIs,
        const IntBuffer* whereCIs,
        bool learnEnabled
    );

    void clearState() {
        memoryActs.fill(0.0f);
        memoryCIs.fill(0);
    }

    // Serialization
    int size() const; // Returns size in bytes
    int stateSize() const; // Returns size of state in bytes

    void write(
        StreamWriter &writer
    ) const;

    void read(
        StreamReader &reader
    );

    void writeState(
        StreamWriter &writer
    ) const;

    void readState(
        StreamReader &reader
    );

    const Int3 &getHiddenSize() const {
        return enc.getHiddenSize();
    }

    const Int3 &getSensorSize() const {
        return sensorSize;
    }

    const Int3 &getWhereSize() const {
        return whereSize;
    }

    Encoder &getEnc() {
        return enc;
    }

    const Encoder &getEnc() const {
        return enc;
    }

    const IntBuffer &getHiddenCIs() const {
        return enc.getHiddenCIs();
    }

    const FloatBuffer &getMemoryActs() const {
        return memoryActs;
    }

    const IntBuffer &getMemoryCIs() const {
        return memoryCIs;
    }
};
}
