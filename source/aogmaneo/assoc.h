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
class Assoc {
private:
    static const int N = S * L;

    Byte* buffer;

public:
    static const int C = N * (N + 1) / 2;

    Assoc()
    :
    buffer(nullptr)
    {}

    Assoc(
        Byte* buffer
    ) 
    :
    buffer(buffer)
    {}

    void set_from(
        Byte* buffer
    ) {
        this->buffer = buffer;
    }

    Byte &operator[](
        int index
    ) {
        assert(buffer != nullptr);
        assert(index >= 0 && index < C);
        
        return buffer[index];
    }

    const Byte &operator[](
        int index
    ) const {
        assert(buffer != nullptr);
        assert(index >= 0 && index < C);
        
        return buffer[index];
    }

    Byte &operator()(
        int r,
        int c
    ) {
        assert(buffer != nullptr);
        assert(r >= 0 && r < N);
        assert(c >= 0 && c < N);
        
        int index = c + r * (r + 1) / 2;

        if (c > r)
            index = r + c * (c + 1) / 2;

        assert(index >= 0 && index < C);

        return buffer[index];
    }

    const Byte &operator()(
        int r,
        int c
    ) const {
        assert(buffer != nullptr);
        assert(r >= 0 && r < N);
        assert(c >= 0 && c < N);
        
        int index;

        if (c > r)
            index = r + c * (c + 1) / 2;
        else
            index = c + r * (r + 1) / 2;

        assert(index >= 0 && index < C);

        return buffer[index];
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
        return C;
    }

    void fill(
        Byte value
    ) {
        for (int i = 0; i < C; i++)
            buffer[i] = value;
    }
    
    Vec<S, L> operator*(
        const Vec<S, L> &other
    ) const {
        assert(buffer != nullptr);

        Bundle<S, L> result;

        for (int r = 0; r < N; r++) {
            int sum = 0;

            for (int i = 0; i < S; i++)
                sum += this->operator()(r, other[i] + i * L);

            result[r] = sum;
        }

        return result.thin();
    }

    void assoc(
        const Vec<S, L> &other
    ) {
        assert(buffer != nullptr);

        for (int i = 0; i < S; i++) {
            int r = other[i] + i * L;

            for (int j = 0; j < S; j++) {
                int c = other[j] + j * L;

                this->operator()(r, c) = 1; 
            }
        }
    }
};
}
