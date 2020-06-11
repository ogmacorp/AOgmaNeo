// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseMatrix.h"

using namespace aon;

void SparseMatrix::initSMLocalRF(
    const Int3 &inSize,
    const Int3 &outSize,
    int radius
) {
    int numOut = outSize.x * outSize.y * outSize.z;

    // Projection constant
    Float2 outToIn = Float2(static_cast<float>(inSize.x) / static_cast<float>(outSize.x),
        static_cast<float>(inSize.y) / static_cast<float>(outSize.y));

    int diam = radius * 2 + 1;

    int numWeightsPerOutput = diam * diam * inSize.z;

    int weightsSize = numOut * numWeightsPerOutput;

    nonZeroValues.resize(weightsSize);

    rowRanges.resize(numOut + 1);

    columnIndices.resize(weightsSize);

    int index = 0;

    // Initialize weight matrix
    for (int ox = 0; ox < outSize.x; ox++)
        for (int oy = 0; oy < outSize.y; oy++) {
            Int2 visiblePositionCenter = project(Int2(ox, oy), outToIn);

            // Lower corner
            Int2 fieldLowerBound(visiblePositionCenter.x - radius, visiblePositionCenter.y - radius);

            // Bounds of receptive field, clamped to input size
            Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
            Int2 iterUpperBound(min(inSize.x - 1, visiblePositionCenter.x + radius), min(inSize.y - 1, visiblePositionCenter.y + radius));

            for (int oz = 0; oz < outSize.z; oz++) {
                Int3 outPos(ox, oy, oz);

                int nonZeroInRow = 0;

                for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                    for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                        for (int iz = 0; iz < inSize.z; iz++) {
                            Int3 inPos(ix, iy, iz);

                            int inIndex = address3(inPos, inSize);

                            nonZeroValues[index] = 0.0f;
                            columnIndices[index] = inIndex;

                            index++;
                            
                            nonZeroInRow++;
                        }
                    }

                rowRanges[address3(outPos, outSize)] = nonZeroInRow;
            }
        }

    nonZeroValues.resize(index);
    columnIndices.resize(index);

    // Convert rowRanges from counts to cumulative counts
    int offset = 0;

	for (int i = 0; i < numOut; i++) {
		int temp = rowRanges[i];

		rowRanges[i] = offset;

		offset += temp;
	}

    rowRanges[numOut] = offset;

    rows = numOut;
    columns = inSize.x * inSize.y * inSize.z;
}

void SparseMatrix::initT() {
	columnRanges.resize(columns + 1, 0);

	rowIndices.resize(nonZeroValues.size());

	nonZeroValueIndices.resize(nonZeroValues.size());

	// Pattern for T
	int nextIndex;

	for (int i = 0; i < rows; i = nextIndex) {
		nextIndex = i + 1;

		for (int j = rowRanges[i]; j < rowRanges[nextIndex]; j++)
			columnRanges[columnIndices[j]]++;
	}

	// Bring row range array in place using exclusive scan
	int offset = 0;

	for (int i = 0; i < columns; i++) {
		int temp = columnRanges[i];

		columnRanges[i] = offset;

		offset += temp;
	}

	columnRanges[columns] = offset;

	Array<int> columnOffsets = columnRanges;

	for (int i = 0; i < rows; i = nextIndex) {
		nextIndex = i + 1;

		for (int j = rowRanges[i]; j < rowRanges[nextIndex]; j++) {
			int colIndex = columnIndices[j];

			int nonZeroIndexT = columnOffsets[colIndex];

			rowIndices[nonZeroIndexT] = i;

			nonZeroValueIndices[nonZeroIndexT] = j;

			columnOffsets[colIndex]++;
		}
	}
}

float SparseMatrix::multiply(
	const FloatBuffer &in,
	int row
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++)
		sum += nonZeroValues[j] * in[columnIndices[j]];

	return sum;
}

int SparseMatrix::count(
	int row
) {
	int nextIndex = row + 1;
	
	return rowRanges[nextIndex] - rowRanges[row];
}

float SparseMatrix::multiplyT(
	const FloatBuffer &in,
	int column
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++)
		sum += nonZeroValues[nonZeroValueIndices[j]] * in[rowIndices[j]];

	return sum;
}

int SparseMatrix::countT(
	int column
) {
	int nextIndex = column + 1;
	
	return columnRanges[nextIndex] - columnRanges[column];
}

float SparseMatrix::multiplyOHVs(
	const ByteBuffer &nonZeroIndices,
	int row,
	int oneHotSize
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int jj = rowRanges[row]; jj < rowRanges[nextIndex]; jj += oneHotSize) {
		int j = jj + nonZeroIndices[columnIndices[jj] / oneHotSize];

		sum += nonZeroValues[j];
	}

	return sum;
}

float SparseMatrix::multiplyOHVsT(
	const ByteBuffer &nonZeroIndices,
	int column,
	int oneHotSize
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int jj = columnRanges[column]; jj < columnRanges[nextIndex]; jj += oneHotSize) {
		int j = jj + nonZeroIndices[rowIndices[jj] / oneHotSize];

		sum += nonZeroValues[nonZeroValueIndices[j]];
	}

	return sum;
}

void SparseMatrix::deltas(
	const FloatBuffer &in,
	float delta,
	int row
) {
	int nextIndex = row + 1;
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++)
		nonZeroValues[j] += delta * in[columnIndices[j]];
}

void SparseMatrix::deltasT(
	const FloatBuffer &in,
	float delta,
	int column
) {
	int nextIndex = column + 1;
	
	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++)
		nonZeroValues[nonZeroValueIndices[j]] += delta * in[rowIndices[j]];
}

void SparseMatrix::deltaOHVs(
	const ByteBuffer &nonZeroIndices,
	float delta,
	int row,
	int oneHotSize
) {
	int nextIndex = row + 1;

	for (int jj = rowRanges[row]; jj < rowRanges[nextIndex]; jj += oneHotSize) {
		int j = jj + nonZeroIndices[columnIndices[jj] / oneHotSize];

		nonZeroValues[j] += delta;
	}
}

void SparseMatrix::deltaOHVsT(
	const ByteBuffer &nonZeroIndices,
	float delta,
	int column,
	int oneHotSize
) {
	int nextIndex = column + 1;

	for (int jj = columnRanges[column]; jj < columnRanges[nextIndex]; jj += oneHotSize) {
		int j = jj + nonZeroIndices[rowIndices[jj] / oneHotSize];

		nonZeroValues[nonZeroValueIndices[j]] += delta;
	}
}

void SparseMatrix::deltaChangedOHVs(
	const ByteBuffer &nonZeroIndices,
	const ByteBuffer &nonZeroIndicesPrev,
	float delta,
	int row,
	int oneHotSize
) {
	int nextIndex = row + 1;

	for (int jj = rowRanges[row]; jj < rowRanges[nextIndex]; jj += oneHotSize) {
		int i = columnIndices[jj] / oneHotSize;

		if (nonZeroIndices[i] != nonZeroIndicesPrev[i]) {
			int j = jj + nonZeroIndices[i];

			nonZeroValues[j] += delta;
		}
	}
}

void SparseMatrix::deltaChangedOHVsT(
	const ByteBuffer &nonZeroIndices,
	const ByteBuffer &nonZeroIndicesPrev,
	float delta,
	int column,
	int oneHotSize
) {
	int nextIndex = column + 1;

	for (int jj = columnRanges[column]; jj < columnRanges[nextIndex]; jj += oneHotSize) {
		int i = rowIndices[jj] / oneHotSize;

		if (nonZeroIndices[i] != nonZeroIndicesPrev[i]) {
			int j = jj + nonZeroIndices[i];

			nonZeroValues[nonZeroValueIndices[j]] += delta;
		}
	}
}

void SparseMatrix::hebb(
	const FloatBuffer &in,
	int row,
	float alpha
) {
	int nextIndex = row + 1;
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++)
		nonZeroValues[j] += alpha * (in[columnIndices[j]] - nonZeroValues[j]);
}

void SparseMatrix::hebbT(
	const FloatBuffer &in,
	int column,
	float alpha
) {
	int nextIndex = column + 1;
	
	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++)
		nonZeroValues[nonZeroValueIndices[j]] += alpha * (in[rowIndices[j]] - nonZeroValues[nonZeroValueIndices[j]]);
}
