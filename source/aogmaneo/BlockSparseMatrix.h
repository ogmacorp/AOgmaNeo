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

    // Transpose
    Array<int> valueIndices;
    Array<int> columnRanges;
    Array<int> rowIndices;

public:
    Array<float> values;

    void init(
        const Int3 &visibleSize,
        const Int3 &hiddenSize,
        int radius
    );

    void initT(
        int radius
    );

    int count(
        int row
    ) const;

    int countT(
        int column
    ) const;

    float multiply(
        int row,
        const FloatBuffer &visibleValues
    ) const;

    float multiplyT(
        int column,
        const FloatBuffer &hiddenValues
    ) const;

    float multiplyCIs(
        int row,
        const IntBuffer &visibleCIs
    ) const;

    float multiplyCIsT(
        int column,
        const IntBuffer &hiddenCIs
    ) const;

    void deltaCIs(
        int row,
        const IntBuffer &visibleCIs,
        float delta
    );

    void deltaCIsT(
        int column,
        const IntBuffer &hiddenCIs,
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
