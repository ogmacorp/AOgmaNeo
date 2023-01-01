// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Helpers.h"

#include "ImageEncoder.h"

namespace aon {
class LocationInvariant {
private:
    Int3 sensorSize;
    Int3 whereSize;

    ImageEncoder enc;

    // Fast associative memory
    ByteBuffer memory;

public:
    float decay;

    LocationInvariant()
    :
    decay(0.01f)
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
        memory.fill(0);
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

    ImageEncoder &getEnc() {
        return enc;
    }

    const ImageEncoder &getEnc() const {
        return enc;
    }

    const IntBuffer &getHiddenCIs() const {
        return enc.getHiddenCIs();
    }

    const ByteBuffer &getMemory() const {
        return memory;
    }
};
}
