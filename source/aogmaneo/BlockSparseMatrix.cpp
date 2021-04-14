// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "BlockSparseMatrix.h"

using namespace aon;

void BlockSparseMatrix::init(
    const Int3 &visibleSize,
    const Int3 &hiddenSize,
    int radius
) {
    this->visibleSize = visibleSize;
    this->hiddenSize = hiddenSize;

    int numHidden = hiddenSize.x * hiddenSize.y * hiddenSize.z;

    // Projection
    Float2 hToV = Float2(static_cast<float>(visibleSize.x) / static_cast<float>(hiddenSize.x),
        static_cast<float>(visibleSize.y) / static_cast<float>(hiddenSize.y));

    int diam = radius * 2 + 1;

    int numBlocksPerHidden = diam * diam;
    int numValuesPerHidden = numBlocksPerHidden * visibleSize.z;

    int blocksSize = numHidden * numBlocksPerHidden;
    int valuesSize = numHidden * numValuesPerHidden;

    values = FloatBuffer(valuesSize, 0.0f);

    rowRanges.resize(numHidden + 1);

    columnIndices.resize(blocksSize);
    int columnIndex = 0;

    // Initialize weight matrix
    for (int hx = 0; hx < hiddenSize.x; hx++)
        for (int hy = 0; hy < hiddenSize.y; hy++) {
            Int2 visibleCenter = project(Int2(hx, hy), hToV);

            // Lower corner
            Int2 fieldLowerBound(visibleCenter.x - radius, visibleCenter.y - radius);

            // Bounds of receptive field, clamped to input size
            Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
            Int2 iterUpperBound(min(visibleSize.x - 1, visibleCenter.x + radius), min(visibleSize.y - 1, visibleCenter.y + radius));

            for (int hz = 0; hz < hiddenSize.z; hz++) {
                Int3 hiddenPosition(hx, hy, hz);

                int nonZeroInRow = 0;

                for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                    for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                        columnIndices[columnIndex++] = address2(Int2(ix, iy), Int2(visibleSize.x, visibleSize.y));
                        
                        nonZeroInRow++;
                    }

                rowRanges[address3(hiddenPosition, hiddenSize)] = nonZeroInRow;
            }
        }

    // Optionally resize to fit
    //columnIndices.resize(columnIndex);

    // Convert rowRanges from counts to cumulative counts
    int offset = 0;

    for (int i = 0; i < numHidden; i++) {
        int temp = rowRanges[i];

        rowRanges[i] = offset;

        offset += temp;
    }

    rowRanges[numHidden] = offset;

    rows = numHidden;
    columns = visibleSize.x * visibleSize.y * visibleSize.z;
}

void BlockSparseMatrix::initT(
    int radius
) {
    //int numHidden = rows;
    int numVisible = columns;

    // Projection
    Float2 vToH = Float2(static_cast<float>(hiddenSize.x) / static_cast<float>(visibleSize.x),
        static_cast<float>(hiddenSize.y) / static_cast<float>(visibleSize.y));

    int reverseRadius = max(ceilf(radius * vToH.x), ceilf(radius * vToH.y));

    int diam = reverseRadius * 2 + 1;

    int numBlocksPerVisible = diam * diam;
    int numValuesPerVisible = numBlocksPerVisible * hiddenSize.z;

    int blocksSize = numVisible * numBlocksPerVisible;
    int valuesSize = numVisible * numValuesPerVisible;

    valueIndices = IntBuffer(valuesSize, 0);

    columnRanges.resize(numVisible + 1);

    rowIndices.resize(blocksSize);
    int rowIndex = 0;

    // Initialize weight matrix
    for (int vx = 0; vx < visibleSize.x; vx++)
        for (int vy = 0; vy < visibleSize.y; vy++) {
            Int2 hiddenCenter = project(Int2(vx, vy), vToH);

            // Lower corner
            Int2 fieldLowerBound(hiddenCenter.x - reverseRadius, hiddenCenter.y - reverseRadius);

            // Bounds of receptive field, clamped to input size
            Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
            Int2 iterUpperBound(min(hiddenSize.x - 1, hiddenCenter.x + reverseRadius), min(hiddenSize.y - 1, hiddenCenter.y + reverseRadius));

            for (int vz = 0; vz < visibleSize.z; vz++) {
                Int3 visiblePosition(vx, vy, vz);

                int nonZeroInColumn = 0;

                for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                    for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                        rowIndices[rowIndex] = address2(Int2(ix, iy), Int2(hiddenSize.x, hiddenSize.y));

                        valueIndices[rowIndices[rowIndex]] = rowRanges[address3(Int3(ix, iy, 0), hiddenSize)];

                        rowIndex++;
                        
                        nonZeroInColumn++;
                    }

                columnRanges[address3(visiblePosition, visibleSize)] = nonZeroInColumn;
            }
        }

    // Optionally resize to fit
    //rowIndices.resize(rowIndex);

    // Convert columnRanges from counts to cumulative counts
    int offset = 0;

    for (int i = 0; i < numVisible; i++) {
        int temp = columnRanges[i];

        columnRanges[i] = offset;

        offset += temp;
    }

    columnRanges[numVisible] = offset;
}

int BlockSparseMatrix::count(
    int row
) const {
    int nextRow = row + 1;

    return rowRanges[nextRow] - rowRanges[row];
}

int BlockSparseMatrix::countT(
    int column
) const {
    int nextColumn = column + 1;

    return columnRanges[nextColumn] - columnRanges[column];
}

float BlockSparseMatrix::multiply(
    int row,
    const FloatBuffer &visibleValues
) const {
    float sum = 0.0f;

    int nextRow = row + 1;

    for (int j = rowRanges[row]; j < rowRanges[nextRow]; j++) {
        int start0 = columnIndices[j] * visibleSize.z;
        int start1 = j * visibleSize.z;

        for (int k = 0; k < visibleSize.z; k++) {
            float value = visibleValues[k + start0];

            sum += values[k + start1] * value;
        }
    }

    return sum;
}

float BlockSparseMatrix::multiplyT(
    int column,
    const FloatBuffer &hiddenValues
) const {
    float sum = 0.0f;

    int nextColumn = column + 1;

    for (int j = columnRanges[column]; j < columnRanges[nextColumn]; j++) {
        int start0 = rowIndices[j] * hiddenSize.z;
        int start1 = valueIndices[j] * hiddenSize.z;

        for (int k = 0; k < hiddenSize.z; k++) {
            float value = hiddenValues[k + start0];

            sum += values[k + start1] * value;
        }
    }

    return sum;
}

float BlockSparseMatrix::multiplyCIs(
    int row,
    const IntBuffer &visibleCIs
) const {
    float sum = 0.0f;

    int nextRow = row + 1;

    for (int j = rowRanges[row]; j < rowRanges[nextRow]; j++)
        sum += values[visibleCIs[columnIndices[j]] + j * visibleSize.z];

    return sum;
}

float BlockSparseMatrix::multiplyCIsT(
    int column,
    const IntBuffer &hiddenCIs
) const {
    float sum = 0.0f;

    int nextColumn = column + 1;

    for (int j = columnRanges[column]; j < columnRanges[nextColumn]; j++)
        sum += values[hiddenCIs[rowIndices[j]] + valueIndices[j] * hiddenSize.z];

    return sum;
}

void BlockSparseMatrix::deltaCIs(
    int row,
    const IntBuffer &visibleCIs,
    float delta
) {
    int nextRow = row + 1;

    for (int j = rowRanges[row]; j < rowRanges[nextRow]; j++)
        values[visibleCIs[columnIndices[j]] + j * visibleSize.z] += delta;
}

void BlockSparseMatrix::deltaCIsT(
    int column,
    const IntBuffer &hiddenCIs,
    float delta
) {
    int nextColumn = column + 1;

    for (int j = columnRanges[column]; j < columnRanges[nextColumn]; j++)
        values[hiddenCIs[rowIndices[j]] + valueIndices[j] * hiddenSize.z] += delta;
}

int BlockSparseMatrix::size() const {
    return 2 * sizeof(int) + 2 * sizeof(Int3) +
        values.size() * sizeof(float) + rowRanges.size() * sizeof(int) + columnIndices.size() * sizeof(int) +
        valueIndices.size() * sizeof(int) + columnRanges.size() * sizeof(int) + rowIndices.size() * sizeof(int);
}

void BlockSparseMatrix::write(
    StreamWriter &writer
) const {
    writer.write(reinterpret_cast<const void*>(&rows), sizeof(int));
    writer.write(reinterpret_cast<const void*>(&columns), sizeof(int));
    writer.write(reinterpret_cast<const void*>(&visibleSize), sizeof(Int3));
    writer.write(reinterpret_cast<const void*>(&hiddenSize), sizeof(Int3));

    int valuesSize = values.size();
    writer.write(reinterpret_cast<const void*>(&valuesSize), sizeof(int));
    writer.write(reinterpret_cast<const void*>(&values[0]), values.size() * sizeof(float));
    
    int rowRangesSize = rowRanges.size();
    writer.write(reinterpret_cast<const void*>(&rowRangesSize), sizeof(int));
    writer.write(reinterpret_cast<const void*>(&rowRanges[0]), rowRanges.size() * sizeof(int));

    int columnIndicesSize = columnIndices.size();
    writer.write(reinterpret_cast<const void*>(&columnIndicesSize), sizeof(int));
    writer.write(reinterpret_cast<const void*>(&columnIndices[0]), columnIndices.size() * sizeof(int));

    int valueIndicesSize = valueIndices.size();
    writer.write(reinterpret_cast<const void*>(&valueIndicesSize), sizeof(int));
    writer.write(reinterpret_cast<const void*>(&valueIndices[0]), valueIndices.size() * sizeof(int));

    int columnRangesSize = columnRanges.size();
    writer.write(reinterpret_cast<const void*>(&columnRangesSize), sizeof(int));
    writer.write(reinterpret_cast<const void*>(&columnRanges[0]), columnRanges.size() * sizeof(int));

    int rowIndicesSize = rowIndices.size();
    writer.write(reinterpret_cast<const void*>(&rowIndicesSize), sizeof(int));
    writer.write(reinterpret_cast<const void*>(&rowIndices[0]), rowIndices.size() * sizeof(int));
}

void BlockSparseMatrix::read(
    StreamReader &reader
) {
    reader.read(reinterpret_cast<void*>(&rows), sizeof(int));
    reader.read(reinterpret_cast<void*>(&columns), sizeof(int));
    reader.read(reinterpret_cast<void*>(&visibleSize), sizeof(Int3));
    reader.read(reinterpret_cast<void*>(&hiddenSize), sizeof(Int3));

    int valuesSize;
    reader.read(reinterpret_cast<void*>(&valuesSize), sizeof(int));
    values.resize(valuesSize);
    reader.read(reinterpret_cast<void*>(&values[0]), values.size() * sizeof(float));
    
    int rowRangesSize;
    reader.read(reinterpret_cast<void*>(&rowRangesSize), sizeof(int));
    rowRanges.resize(rowRangesSize);
    reader.read(reinterpret_cast<void*>(&rowRanges[0]), rowRanges.size() * sizeof(int));

    int columnIndicesSize;
    reader.read(reinterpret_cast<void*>(&columnIndicesSize), sizeof(int));
    columnIndices.resize(columnIndicesSize);
    reader.read(reinterpret_cast<void*>(&columnIndices[0]), columnIndices.size() * sizeof(int));

    int valueIndicesSize;
    reader.read(reinterpret_cast<void*>(&valueIndicesSize), sizeof(int));
    valueIndices.resize(valueIndicesSize);
    reader.read(reinterpret_cast<void*>(&valueIndices[0]), valueIndices.size() * sizeof(int));

    int columnRangesSize;
    reader.read(reinterpret_cast<void*>(&columnRangesSize), sizeof(int));
    columnRanges.resize(columnRangesSize);
    reader.read(reinterpret_cast<void*>(&columnRanges[0]), columnRanges.size() * sizeof(int));

    int rowIndicesSize;
    reader.read(reinterpret_cast<void*>(&rowIndicesSize), sizeof(int));
    rowIndices.resize(rowIndicesSize);
    reader.read(reinterpret_cast<void*>(&rowIndices[0]), rowIndices.size() * sizeof(int));
}
