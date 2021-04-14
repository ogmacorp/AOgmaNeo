// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Helpers.h"

namespace aon {
class BlockSparseMatrix {
private:
    int rows;
    int columns;
    Int3 visibleSize;
    Int3 hiddenSize;

    Array<int> rowRanges;
    Array<int> columnIndices;

public:
    Array<float> values;

    void init(
        const Int3 &visibleSize,
        const Int3 &hiddenSize,
        int radius
    );

    int count(
        int row
    ) const;

    float multiply(
        int row,
        const FloatBuffer &visibleValues
    ) const;

    void reverse(
        int row,
        float value,
        FloatBuffer &accum
    ) const;

    float multiplyCIs(
        int row,
        const IntBuffer &visibleCIs
    ) const;

    void reverseCIs(
        int row,
        float value,
        const IntBuffer &visibleCIs,
        FloatBuffer &accum
    ) const;

    void deltaCIs(
        int row,
        const IntBuffer &visibleCIs,
        float delta
    );

    // Serialization
    int size() const; // Returns size in bytes

    void write(
        StreamWriter &writer
    ) const;

    void read(
        StreamReader &reader
    );
};
}
