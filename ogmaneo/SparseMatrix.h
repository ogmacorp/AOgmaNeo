// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Array.h"

namespace ogmaneo {
// Compressed sparse row (CSR) format
struct SparseMatrix {
	int rows, columns; // Dimensions

	Array<float> nonZeroValues;
	Array<int> rowRanges;
	Array<int> columnIndices;

	// Transpose
	Array<int> nonZeroValueIndices;
	Array<int> columnRanges;
	Array<int> rowIndices;

	// --- Init ---

	SparseMatrix()
	{}

	// Generate a transpose, must be called after the original has been created
	void initT();

	// --- Dense ---

	float multiply(
		const Array<float> &in,
		int row
	);

	float distance2(
		const Array<float> &in,
		int row
	);

	int count(
		int row
	);

	float count(
		const Array<float> &in,
		int row
	);

    void fill(
        int row,
        float value
    );

    float total(
        int row
    );

	float multiplyT(
		const Array<float> &in,
		int column
	);

	float distance2T(
		const Array<float> &in,
		int column
	);

	int countT(
		int column
	);

	float countT(
		const Array<float> &in,
		int column
	);

    void fillT(
        int column,
        float value
    );

    float totalT(
        int column
    );

	// --- One-Hot Vectors Operations ---

	float multiplyOHVs(
		const Array<int> &nonZeroIndices,
		int row,
		int oneHotSize
	);

	float multiplyOHVsT(
		const Array<int> &nonZeroIndices,
		int column,
		int oneHotSize
	);

	float multiplyOHVs(
		const Array<int> &nonZeroIndices,
		const Array<float> &nonZeroScalars,
		int row,
		int oneHotSize
	);

	float multiplyOHVsT(
		const Array<int> &nonZeroIndices,
		const Array<float> &nonZeroScalars,
		int column,
		int oneHotSize
	);

	float distance2OHVs(
		const Array<int> &nonZeroIndices,
		int row,
		int oneHotSize
	);

	float distance2OHVsT(
		const Array<int> &nonZeroIndices,
		int column,
		int oneHotSize
	);

	int countChangedOHVs(
		const Array<int> &nonZeroIndices,
		const Array<int> &nonZeroIndicesPrev,
		int row,
		int oneHotSize
	);

	int countChangedOHVsT(
		const Array<int> &nonZeroIndices,
		const Array<int> &nonZeroIndicesPrev,
		int column,
		int oneHotSize
	);

	float multiplyChangedOHVs(
		const Array<int> &nonZeroIndices,
		const Array<int> &nonZeroIndicesPrev,
		int row,
		int oneHotSize
	);

	float multiplyChangedOHVsT(
		const Array<int> &nonZeroIndices,
		const Array<int> &nonZeroIndicesPrev,
		int column,
		int oneHotSize
	);

	// --- Delta Rules ---

	void deltas(
		const Array<float> &in,
		float delta,
		int row
	);

	void deltasT(
		const Array<float> &in,
		float delta,
		int column
	);

	void deltaOHVs(
		const Array<int> &nonZeroIndices,
		float delta,
		int row,
		int oneHotSize
	);

	void deltaOHVsT(
		const Array<int> &nonZeroIndices,
		float delta,
		int column,
		int oneHotSize
	);

	void deltaOHVs(
		const Array<int> &nonZeroIndices,
		const Array<float> &nonZeroScalars,
		float delta,
		int row,
		int oneHotSize
	);

	void deltaOHVsT(
		const Array<int> &nonZeroIndices,
		const Array<float> &nonZeroScalars,
		float delta,
		int column,
		int oneHotSize
	);

	void deltaChangedOHVs(
		const Array<int> &nonZeroIndices,
		const Array<int> &nonZeroIndicesPrev,
		float delta,
		int row,
		int oneHotSize
	);

	void deltaChangedOHVsT(
		const Array<int> &nonZeroIndices,
		const Array<int> &nonZeroIndicesPrev,
		float delta,
		int column,
		int oneHotSize
	);

	void deltaUsageOHVs(
		const Array<int> &nonZeroIndices,
		const Array<int> &nonZeroIndicesPrev,
		const Array<float> &usages,
		float delta,
		int row,
		int oneHotSize
	);

	void deltaUsageOHVsT(
		const Array<int> &nonZeroIndices,
		const Array<int> &nonZeroIndicesPrev,
		const Array<float> &usages,
		float delta,
		int column,
		int oneHotSize
	);

	void fillOHVs(
		const Array<int> &nonZeroIndices,
		int row,
		int oneHotSize,
		float value
	);

	void fillOHVsT(
		const Array<int> &nonZeroIndices,
		int column,
		int oneHotSize,
		float value
	);

	void deltaTracedOHVs(
		SparseMatrix &traces,
		float delta,
		int row,
		float traceDecay
	);

	void deltaTracedOHVsT(
		SparseMatrix &traces,
		float delta,
		int column,
		float traceDecay
	);

	// --- Hebb Rules ---

	void hebb(
		const Array<float> &in,
		int row,
		float alpha
	);

	void hebbT(
		const Array<float> &in,
		int column,
		float alpha
	);

	void hebbOHVs(
		const Array<int> &nonZeroIndices,
		int row,
		int oneHotSize,
		float alpha
	);

	void hebbOHVsT(
		const Array<int> &nonZeroIndices,
		int column,
		int oneHotSize,
		float alpha
	);
};
} // namespace ogmaneo