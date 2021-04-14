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

    Array<float> values;

    Array<int> rowRanges;
    Array<int> columnIndices;

    // Transpose
    Array<int> valueIndices;
    Array<int> columnRanges;
    Array<int> rowIndices;

public:
    void init(
        const Int3 &visibleSize,
        const Int3 &hiddenSize,
        int radius
    );

    void initT();

    float multiplyCIs(
        int row,
        const IntBuffer &visibleCIs
    );

    float multiplyCIsT(
        int column,
        const IntBuffer &hiddenCIs
    );
};
}
