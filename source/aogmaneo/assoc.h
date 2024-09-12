// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "vec.h"

namespace aon {
template<int S, int L>
class Associator {
private:
    static const int N = S * L;
    static const int N2 = N * N;

    Byte buffer[N * N];

public:
    Associator() {}

    Associator(
        int value
    ) {
        fill(value);
    }

    int &operator[](
        int index
    ) {
        assert(index >= 0 && index < N2);
        
        return buffer[index];
    }

    const int &operator[](
        int index
    ) const {
        assert(index >= 0 && index < N2);
        
        return buffer[index];
    }

    int &operator()(
        int r,
        int c
    ) {
        assert(r >= 0 && r < N);
        assert(c >= 0 && c < N);
        
        return buffer[c + r * N];
    }

    const int &operator()(
        int r,
        int c
    ) const {
        assert(r >= 0 && r < N);
        assert(c >= 0 && c < N);
        
        return buffer[c + r * N];
    }

    // number of segments
    int segments() const {
        return S;
    }

    // segment length
    int length() const {
        return L;
    }

    int vsize() const {
        return N;
    }

    // total size
    int size() const {
        return N2;
    }

    void fill(
        int value
    ) {
        for (int i = 0; i < N2; i++)
            buffer[i] = value;
    }
    
    Vec<S, L> operator*(
        const Vec<S, L> &other
    ) {
        Bundle<S, L> result;

        for (int r = 0; r < N; r++) {
            int start = r * N;

            int sum = 0;

            for (int i = 0; i < S; i++)
                sum += buffer[other[i] + start];

            result[r] = sum;
        }

        return result.thin();
    }

    void assoc(
        const Vec<S, L> v1, Vec<S, L> v2,
        int limit = 1024
    ) {
        for (int i = 0; i < S; i++) {
            int index1 = v1[i] + i * L;

            for (int j = 0; j < S; j++) {
                int index2 = v2[j] + j * L;

                int buffer_index = index2 + N * index1;

                buffer[buffer_index] = min(limit, buffer[buffer_index] + 1); 
            }
        }
    }
};
}
