// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Helpers.h"

using namespace ogmaneo;

float ogmaneo::expf(float x) {
    if (x > 0.0f) {
        float res = 1.0f;

        float p = x;

        for (int i = 0; i < expIters; i++) {
            res += p / expFactorials[i];

            p *= x;
        }

        return res;
    }

    float res = 1.0f;

    float p = -x;

    for (int i = 0; i < expIters; i++) {
        res += p / expFactorials[i];

        p *= -x;
    }

    return 1.0f / res;
}

void ogmaneo::copyInt(
    const IntBuffer* src,
    IntBuffer* dst
) {
    for (int i = 0; i < src->size(); i++)
        (*dst)[i] = (*src)[i];
}

void ogmaneo::copyFloat(
    const FloatBuffer* src,
    FloatBuffer* dst
) {
    for (int i = 0; i < src->size(); i++)
        (*dst)[i] = (*src)[i];
}

Array<IntBuffer*> ogmaneo::get(
    Array<IntBuffer> &v
) {
    Array<IntBuffer*> vp(v.size());

    for (int i = 0; i < v.size(); i++)
        vp[i] = &v[i];

    return vp;
}

Array<FloatBuffer*> ogmaneo::get(
    Array<FloatBuffer> &v
) {
    Array<FloatBuffer*> vp(v.size());

    for (int i = 0; i < v.size(); i++)
        vp[i] = &v[i];

    return vp;
}

Array<const IntBuffer*> ogmaneo::constGet(
    const Array<IntBuffer> &v
) {
    Array<const IntBuffer*> vp(v.size());

    for (int i = 0; i < v.size(); i++)
        vp[i] = &v[i];

    return vp;
}

Array<const FloatBuffer*> ogmaneo::constGet(
    const Array<FloatBuffer> &v
) {
    Array<const FloatBuffer*> vp(v.size());

    for (int i = 0; i < v.size(); i++)
        vp[i] = &v[i];

    return vp;
}

Array<IntBuffer*> ogmaneo::get(
    CircleBuffer<IntBuffer> &v
) {
    Array<IntBuffer*> vp(v.size());

    for (int i = 0; i < v.size(); i++)
        vp[i] = &v[i];

    return vp;
}

Array<FloatBuffer*> ogmaneo::get(
    CircleBuffer<FloatBuffer> &v
) {
    Array<FloatBuffer*> vp(v.size());

    for (int i = 0; i < v.size(); i++)
        vp[i] = &v[i];

    return vp;
}

Array<const IntBuffer*> ogmaneo::constGet(
    const CircleBuffer<IntBuffer> &v
) {
    Array<const IntBuffer*> vp(v.size());

    for (int i = 0; i < v.size(); i++)
        vp[i] = &v[i];

    return vp;
}

Array<const FloatBuffer*> ogmaneo::constGet(
    const CircleBuffer<FloatBuffer> &v
) {
    Array<const FloatBuffer*> vp(v.size());

    for (int i = 0; i < v.size(); i++)
        vp[i] = &v[i];

    return vp;
}

unsigned long ogmaneo::seed = 1234;

int ogmaneo::rand() {
    return MWC64X(&seed) % 100000;
}

float ogmaneo::randf() {
    return rand() / 99999.0f;
}

float ogmaneo::randf(float low, float high) {
    return low + (high - low) * randf();
}

void ogmaneo::initSMLocalRF(
    const Int3 &inSize,
    const Int3 &outSize,
    int radius,
    SparseMatrix &mat
) {
    int numOut = outSize.x * outSize.y * outSize.z;

    // Projection constant
    Float2 outToIn = Float2(static_cast<float>(inSize.x) / static_cast<float>(outSize.x),
        static_cast<float>(inSize.y) / static_cast<float>(outSize.y));

    int diam = radius * 2 + 1;

    int numWeightsPerOutput = diam * diam * inSize.z;

    int weightsSize = numOut * numWeightsPerOutput;

    mat.nonZeroValues.resize(weightsSize);

    mat.rowRanges.resize(numOut + 1);

    mat.columnIndices.resize(weightsSize);

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

                            mat.nonZeroValues[index] = 0.0f;
                            mat.columnIndices[index] = inIndex;

                            index++;
                            
                            nonZeroInRow++;
                        }
                    }

                mat.rowRanges[address3(outPos, outSize)] = nonZeroInRow;
            }
        }

    mat.nonZeroValues.resize(index);
    mat.columnIndices.resize(index);

    // Convert rowRanges from counts to cumulative counts
    int offset = 0;

	for (int i = 0; i < numOut; i++) {
		int temp = mat.rowRanges[i];

		mat.rowRanges[i] = offset;

		offset += temp;
	}

    mat.rowRanges[numOut] = offset;

    mat.rows = numOut;
    mat.columns = inSize.x * inSize.y * inSize.z;
}