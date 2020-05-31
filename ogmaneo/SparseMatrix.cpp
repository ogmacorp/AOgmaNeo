// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseMatrix.h"

using namespace ogmaneo;

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
	const Array<float> &in,
	int row
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++)
		sum += nonZeroValues[j] * in[columnIndices[j]];

	return sum;
}

float SparseMatrix::distance2(
	const Array<float> &in,
	int row
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++) {
		float delta = in[columnIndices[j]] - nonZeroValues[j];

		sum += delta * delta;
	}

	return sum;
}

int SparseMatrix::count(
	int row
) {
	int nextIndex = row + 1;
	
	return rowRanges[nextIndex] - rowRanges[row];
}

float SparseMatrix::count(
	const Array<float> &in,
	int row
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++)
		sum += in[columnIndices[j]];

	return sum;
}

void SparseMatrix::fill(
	int row,
    float value
) {
	int nextIndex = row + 1;
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++)
		nonZeroValues[j] = value;
}

float SparseMatrix::total(
	int row
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++)
		sum += nonZeroValues[j];

	return sum;
}

float SparseMatrix::multiplyT(
	const Array<float> &in,
	int column
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++)
		sum += nonZeroValues[nonZeroValueIndices[j]] * in[rowIndices[j]];

	return sum;
}

float SparseMatrix::distance2T(
	const Array<float> &in,
	int column
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++) {
		float delta = in[rowIndices[j]] - nonZeroValues[nonZeroValueIndices[j]];
	
		sum += delta * delta;
	}

	return sum;
}

int SparseMatrix::countT(
	int column
) {
	int nextIndex = column + 1;
	
	return columnRanges[nextIndex] - columnRanges[column];
}

float SparseMatrix::countT(
	const Array<float> &in,
	int column
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++)
		sum += in[rowIndices[j]];

	return sum;
}

void SparseMatrix::fillT(
	int column,
    float value
) {
	int nextIndex = column + 1;
	
	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++)
		nonZeroValues[nonZeroValueIndices[j]] = value;
}

float SparseMatrix::totalT(
	int column
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++)
		sum += nonZeroValues[nonZeroValueIndices[j]];

	return sum;
}

float SparseMatrix::multiplyOHVs(
	const Array<int> &nonZeroIndices,
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
	const Array<int> &nonZeroIndices,
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

float SparseMatrix::multiplyOHVs(
	const Array<int> &nonZeroIndices,
	const Array<float> &nonZeroScalars,
	int row,
	int oneHotSize
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int jj = rowRanges[row]; jj < rowRanges[nextIndex]; jj += oneHotSize) {
		int i = columnIndices[jj] / oneHotSize;
		int j = jj + nonZeroIndices[i];

		sum += nonZeroValues[j] * nonZeroScalars[i];
	}

	return sum;
}

float SparseMatrix::multiplyOHVsT(
	const Array<int> &nonZeroIndices,
	const Array<float> &nonZeroScalars,
	int column,
	int oneHotSize
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int jj = columnRanges[column]; jj < columnRanges[nextIndex]; jj += oneHotSize) {
		int i = rowIndices[jj] / oneHotSize;
		int j = jj + nonZeroIndices[i];

		sum += nonZeroValues[nonZeroValueIndices[j]] * nonZeroScalars[i];
	}

	return sum;
}

float SparseMatrix::distance2OHVs(
	const Array<int> &nonZeroIndices,
	int row,
	int oneHotSize
) {
	float dist = 0.0f;

	int nextIndex = row + 1;
	
	for (int jj = rowRanges[row]; jj < rowRanges[nextIndex]; jj += oneHotSize) {
		int targetDJ = nonZeroIndices[columnIndices[jj] / oneHotSize];

		for (int dj = 0; dj < oneHotSize; dj++) {
			float delta = (dj == targetDJ ? 1.0f : 0.0f) - nonZeroValues[jj + dj];

			dist += delta * delta;
		}
	}

	return dist;
}

float SparseMatrix::distance2OHVsT(
	const Array<int> &nonZeroIndices,
	int column,
	int oneHotSize
) {
	float dist = 0.0f;

	int nextIndex = column + 1;
	
	for (int jj = columnRanges[column]; jj < columnRanges[nextIndex]; jj += oneHotSize) {
		int targetDJ = nonZeroIndices[rowIndices[jj] / oneHotSize];

		for (int dj = 0; dj < oneHotSize; dj++) {
			float delta = (dj == targetDJ ? 1.0f : 0.0f) - nonZeroValues[nonZeroValueIndices[jj + dj]];

			dist += delta * delta;
		}
	}

	return dist;
}

int SparseMatrix::countChangedOHVs(
	const Array<int> &nonZeroIndices,
	const Array<int> &nonZeroIndicesPrev,
	int row,
	int oneHotSize
) {
	int count = 0;

	int nextIndex = row + 1;
	
	for (int jj = rowRanges[row]; jj < rowRanges[nextIndex]; jj += oneHotSize) {
		int i = columnIndices[jj] / oneHotSize;

		if (nonZeroIndices[i] != nonZeroIndicesPrev[i])
			count++;
	}

	return count;
}

int SparseMatrix::countChangedOHVsT(
	const Array<int> &nonZeroIndices,
	const Array<int> &nonZeroIndicesPrev,
	int column,
	int oneHotSize
) {
	int count = 0;

	int nextIndex = column + 1;
	
	for (int jj = columnRanges[column]; jj < columnRanges[nextIndex]; jj += oneHotSize) {
		int i = rowIndices[jj] / oneHotSize;
		
		if (nonZeroIndices[i] != nonZeroIndicesPrev[i])
			count++;
	}

	return count;
}

float SparseMatrix::multiplyChangedOHVs(
	const Array<int> &nonZeroIndices,
	const Array<int> &nonZeroIndicesPrev,
	int row,
	int oneHotSize
) {
	float sum = 0.0f;

	int nextIndex = row + 1;
	
	for (int jj = rowRanges[row]; jj < rowRanges[nextIndex]; jj += oneHotSize) {
		int i = columnIndices[jj] / oneHotSize;

		if (nonZeroIndices[i] != nonZeroIndicesPrev[i]) {
			int j = jj + nonZeroIndices[i];

			sum += nonZeroValues[j];
		}
	}

	return sum;
}

float SparseMatrix::multiplyChangedOHVsT(
	const Array<int> &nonZeroIndices,
	const Array<int> &nonZeroIndicesPrev,
	int column,
	int oneHotSize
) {
	float sum = 0.0f;

	int nextIndex = column + 1;
	
	for (int jj = columnRanges[column]; jj < columnRanges[nextIndex]; jj += oneHotSize) {
		int i = rowIndices[jj] / oneHotSize;

		if (nonZeroIndices[i] != nonZeroIndicesPrev[i]) {
			int j = jj + nonZeroIndices[i];

			sum += nonZeroValues[nonZeroValueIndices[j]];
		}
	}

	return sum;
}

void SparseMatrix::deltas(
	const Array<float> &in,
	float delta,
	int row
) {
	int nextIndex = row + 1;
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++)
		nonZeroValues[j] += delta * in[columnIndices[j]];
}

void SparseMatrix::deltasT(
	const Array<float> &in,
	float delta,
	int column
) {
	int nextIndex = column + 1;
	
	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++)
		nonZeroValues[nonZeroValueIndices[j]] += delta * in[rowIndices[j]];
}

void SparseMatrix::deltaOHVs(
	const Array<int> &nonZeroIndices,
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
	const Array<int> &nonZeroIndices,
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

void SparseMatrix::deltaOHVs(
	const Array<int> &nonZeroIndices,
	const Array<float> &nonZeroScalars,
	float delta,
	int row,
	int oneHotSize
) {
	int nextIndex = row + 1;

	for (int jj = rowRanges[row]; jj < rowRanges[nextIndex]; jj += oneHotSize) {
		int i = columnIndices[jj] / oneHotSize;
		int j = jj + nonZeroIndices[i];

		nonZeroValues[j] += delta * nonZeroScalars[i];
	}
}

void SparseMatrix::deltaOHVsT(
	const Array<int> &nonZeroIndices,
	const Array<float> &nonZeroScalars,
	float delta,
	int column,
	int oneHotSize
) {
	int nextIndex = column + 1;

	for (int jj = columnRanges[column]; jj < columnRanges[nextIndex]; jj += oneHotSize) {
		int i = rowIndices[jj] / oneHotSize;
		int j = jj + nonZeroIndices[i];

		nonZeroValues[nonZeroValueIndices[j]] += delta * nonZeroScalars[i];
	}
}

void SparseMatrix::deltaChangedOHVs(
	const Array<int> &nonZeroIndices,
	const Array<int> &nonZeroIndicesPrev,
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
	const Array<int> &nonZeroIndices,
	const Array<int> &nonZeroIndicesPrev,
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

void SparseMatrix::deltaUsageOHVs(
	const Array<int> &nonZeroIndices,
	const Array<int> &nonZeroIndicesPrev,
	const Array<float> &usages,
	float delta,
	int row,
	int oneHotSize
) {
	int nextIndex = row + 1;

	for (int jj = rowRanges[row]; jj < rowRanges[nextIndex]; jj += oneHotSize) {
		int i = columnIndices[jj] / oneHotSize;

		if (nonZeroIndices[i] != nonZeroIndicesPrev[i]) {
			int j = jj + nonZeroIndices[i];

			nonZeroValues[j] += delta * usages[columnIndices[j]];
		}
	}
}

void SparseMatrix::deltaUsageOHVsT(
	const Array<int> &nonZeroIndices,
	const Array<int> &nonZeroIndicesPrev,
	const Array<float> &usages,
	float delta,
	int column,
	int oneHotSize
) {
	int nextIndex = column + 1;

	for (int jj = columnRanges[column]; jj < columnRanges[nextIndex]; jj += oneHotSize) {
		int i = rowIndices[jj] / oneHotSize;

		if (nonZeroIndices[i] != nonZeroIndicesPrev[i]) {
			int j = jj + nonZeroIndices[i];

			nonZeroValues[nonZeroValueIndices[j]] += delta * usages[rowIndices[j]];
		}
	}
}

void SparseMatrix::fillOHVs(
	const Array<int> &nonZeroIndices,
	int row,
	int oneHotSize,
	float value
) {
	int nextIndex = row + 1;

	for (int jj = rowRanges[row]; jj < rowRanges[nextIndex]; jj += oneHotSize) {
		int j = jj + nonZeroIndices[columnIndices[jj] / oneHotSize];

		nonZeroValues[j] = value;
	}
}

void SparseMatrix::fillOHVsT(
	const Array<int> &nonZeroIndices,
	int column,
	int oneHotSize,
	float value
) {
	int nextIndex = column + 1;

	for (int jj = columnRanges[column]; jj < columnRanges[nextIndex]; jj += oneHotSize) {
		int j = jj + nonZeroIndices[rowIndices[jj] / oneHotSize];

		nonZeroValues[nonZeroValueIndices[j]] = value;
	}
}

void SparseMatrix::deltaTracedOHVs(
	SparseMatrix &traces,
	float delta,
	int row,
	float traceDecay
) {
	int nextIndex = row + 1;

	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++) {
		nonZeroValues[j] += delta * traces.nonZeroValues[j];
		traces.nonZeroValues[j] *= traceDecay;
	}
}

void SparseMatrix::deltaTracedOHVsT(
	SparseMatrix &traces,
	float delta,
	int column,
	float traceDecay
) {
	int nextIndex = column + 1;

	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++) {
		nonZeroValues[nonZeroValueIndices[j]] += delta * traces.nonZeroValues[nonZeroValueIndices[j]];
		traces.nonZeroValues[nonZeroValueIndices[j]] *= traceDecay;
	}
}

void SparseMatrix::hebb(
	const Array<float> &in,
	int row,
	float alpha
) {
	int nextIndex = row + 1;
	
	for (int j = rowRanges[row]; j < rowRanges[nextIndex]; j++)
		nonZeroValues[j] += alpha * (in[columnIndices[j]] - nonZeroValues[j]);
}

void SparseMatrix::hebbT(
	const Array<float> &in,
	int column,
	float alpha
) {
	int nextIndex = column + 1;
	
	for (int j = columnRanges[column]; j < columnRanges[nextIndex]; j++)
		nonZeroValues[nonZeroValueIndices[j]] += alpha * (in[rowIndices[j]] - nonZeroValues[nonZeroValueIndices[j]]);
}

void SparseMatrix::hebbOHVs(
	const Array<int> &nonZeroIndices,
	int row,
	int oneHotSize,
	float alpha
) {
	int nextIndex = row + 1;
	
	for (int jj = rowRanges[row]; jj < rowRanges[nextIndex]; jj += oneHotSize) {
		int targetDJ = nonZeroIndices[columnIndices[jj] / oneHotSize];

		for (int dj = 0; dj < oneHotSize; dj++) {
			int j = jj + dj;

			float target = (dj == targetDJ ? 1.0f : 0.0f);

			nonZeroValues[j] += alpha * (target - nonZeroValues[j]);
		}
	}
}

void SparseMatrix::hebbOHVsT(
	const Array<int> &nonZeroIndices,
	int column,
	int oneHotSize,
	float alpha
) {
	int nextIndex = column + 1;
	
	for (int jj = columnRanges[column]; jj < columnRanges[nextIndex]; jj += oneHotSize) {
		int targetDJ = nonZeroIndices[rowIndices[jj] / oneHotSize];

		for (int dj = 0; dj < oneHotSize; dj++) {
			int j = jj + dj;

			float target = (dj == targetDJ ? 1.0f : 0.0f);

			nonZeroValues[nonZeroValueIndices[j]] += alpha * (target - nonZeroValues[nonZeroValueIndices[j]]);
		}
	}
}