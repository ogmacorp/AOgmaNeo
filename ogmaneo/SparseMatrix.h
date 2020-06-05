// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Helpers.h"

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

    // Sparse matrix init
    void initSMLocalRF(
        const Int3 &inSize, // Size of input field
        const Int3 &outSize, // Size of output field
        int radius // Radius of output onto input
    );

	// Generate a transpose, must be called after the original has been created
	void initT();

	// --- Dense ---

	float multiply(
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

	float multiplyT(
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

	// --- One-Hot Vectors Operations ---

	float multiplyOHVs(
		const ByteBuffer &nonZeroIndices,
		int row,
		int oneHotSize
	);

	float multiplyOHVsT(
		const ByteBuffer &nonZeroIndices,
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
		const ByteBuffer &nonZeroIndices,
		float delta,
		int row,
		int oneHotSize
	);

	void deltaOHVsT(
		const ByteBuffer &nonZeroIndices,
		float delta,
		int column,
		int oneHotSize
	);
};
} // namespace ogmaneo