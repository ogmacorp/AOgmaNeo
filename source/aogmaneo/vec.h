// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "helpers.h"

namespace aon {
template<int S, int L>
class Bundle;

template<int S, int L>
class Vec {
private:
    Byte buffer[S];

public:
    static const int N = S * L;

    Vec() {}

    Vec(
        Byte value
    ) {
        fill(value);
    }

    static Vec<S, L> randomized(
        unsigned long* state = &global_state
    ) {
        Vec<S, L> result;

        for (int i = 0; i < S; i++)
            result.buffer[i] = rand(state) % L;

        return result;
    };

    Byte &operator[](
        int index
    ) {
        assert(index >= 0 && index < S);
        
        return buffer[index];
    }

    const Byte &operator[](
        int index
    ) const {
        assert(index >= 0 && index < S);
        
        return buffer[index];
    }

    // number of segments
    static int segments() {
        return S;
    }

    // segment length
    static int length() {
        return L;
    }

    // total size
    static int size() {
        return N;
    }

    void fill(
        Byte value
    ) {
        for (int i = 0; i < S; i++)
            buffer[i] = value;
    }
    
    Vec<S, L> operator*(
        const Vec<S, L> &other
    ) const {
        Vec<S, L> result;

        for (int i = 0; i < S; i++)
            result[i] = (buffer[i] + other.buffer[i]) % L;

        return result;
    }

    const Vec<S, L> &operator*=(
        const Vec<S, L> &other
    ) {
        for (int i = 0; i < S; i++)
            buffer[i] = (buffer[i] + other.buffer[i]) % L; 

        return *this;
    }

    Vec<S, L> operator/(
        const Vec<S, L> &other
    ) const {
        Vec<S, L> result;

        for (int i = 0; i < S; i++)
            result[i] = (buffer[i] + L - other.buffer[i]) % L;

        return result;
    }

    const Vec<S, L> &operator/=(
        const Vec<S, L> &other
    ) {
        for (int i = 0; i < S; i++)
            buffer[i] = (buffer[i] + L - other.buffer[i]) % L; 

        return *this;
    }

    Bundle<S, L> operator+(
        const Vec<S, L> &other
    ) const {
        Bundle<S, L> result = 0;

        for (int i = 0; i < S; i++) {
            int start = i * L;

            result[buffer[i] + start]++;
            result[other.buffer[i] + start]++;
        }

        return result;
    }

    Bundle<S, L> operator*(
        float value
    ) const {
        Bundle<S, L> result = 0.0f;

        for (int i = 0; i < S; i++)
            result[buffer[i] + L * i] = value;

        return result;
    }

    Vec<S, L> permute(
        int shift = 1
    ) {
        Vec<S, L> result;

        for (int i = 0; i < S; i++)
            result[i] = buffer[(i + S + shift) % S];

        return result;
    }

    int dot(
        const Vec<S, L> &other
    ) const {
        int sum = 0;

        for (int i = 0; i < S; i++)
            sum += (buffer[i] == other.buffer[i]);

        return sum;
    }

    bool operator==(
        const Vec<S, L> &other
    ) const {
        for (int i = 0; i < S; i++)
            if (buffer[i] != other.buffer[i])
                return false;

        return true;
    }

    bool operator!=(
        const Vec<S, L> &other
    ) const {
        return !this->operator==(other);
    }
};

template<int S, int L>
class Bundle {
public:
    static const int N = S * L;

private:
    float buffer[N];

public:

    Bundle()
    {}

    Bundle(
        float value
    ) {
        fill(value);
    }

    const Bundle<S, L> &operator=(
        float value
    ) {
        fill(value);

        return *this;
    }

    float &operator[](
        int index
    ) {
        assert(index >= 0 && index < N);
        
        return buffer[index];
    }

    const float &operator[](
        int index
    ) const {
        assert(index >= 0 && index < N);
        
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

    // total size
    int size() const {
        return N;
    }

    void fill(
        float value
    ) {
        for (int i = 0; i < N; i++)
            buffer[i] = value;
    }

    Bundle<S, L> operator+(
        const Bundle<S, L> &other
    ) const {
        Bundle<S, L> result;

        for (int i = 0; i < N; i++)
            result[i] = buffer[i] + other.buffer[i];

        return result;
    }

    const Bundle<S, L> &operator+=(
        const Bundle<S, L> &other
    ) {
        for (int i = 0; i < N; i++)
            buffer[i] += other.buffer[i];

        return *this;
    }

    Bundle<S, L> operator+(
        const Vec<S, L> &other
    ) const {
        Bundle<S, L> result = 0;

        for (int i = 0; i < S; i++)
            result[other[i] + i * L]++;

        return result;
    }

    const Bundle<S, L> &operator+=(
        const Vec<S, L> &other
    ) {
        for (int i = 0; i < S; i++)
            buffer[other[i] + i * L]++;

        return *this;
    }

    Bundle<S, L> operator*(
        float value
    ) const {
        Bundle<S, L> result;

        for (int i = 0; i < N; i++)
            result[i] = buffer[i] * value;

        return result;
    }

    const Bundle<S, L> &operator*=(
        float value
    ) {
        for (int i = 0; i < N; i++)
            buffer[i] *= value;

        return *this;
    }

    Vec<S, L> thin() const {
        Vec<S, L> result;

        for (int i = 0; i < S; i++) {
            int start = i * L;

            float mv = 0.0f;
            int mi = 0;

            for (int j = 0; j < L; j++) {
                int index = j + start;

                if (buffer[index] > mv) {
                    mv = buffer[index];
                    mi = j;
                }
            }

            // check for multiple maxima and do context-dependent thinning in that case
            int index_sum = 0;
            int mcount = 0;

            for (int j = 0; j < L; j++) {
                int index = j + start;

                if (buffer[index] == mv) {
                    index_sum += j;
                    mcount++;
                }
            }

            if (mcount > 0) {
                // context-dependent thinning
                int p = index_sum % mcount;

                mcount = 0;

                for (int j = 0; j < L; j++) {
                    int index = j + start;

                    if (buffer[index] == mv) {
                        if (mcount == p) {
                            mi = j;

                            break;
                        }

                        mcount++;
                    }
                }
            }

            result[i] = mi;
        }

        return result;
    }

    bool operator==(
        const Bundle<S, L> &other
    ) const {
        for (int i = 0; i < N; i++)
            if (buffer[i] != other.buffer[i])
                return false;

        return true;
    }

    bool operator!=(
        const Bundle<S, L> &other
    ) const {
        return !this->operator==(other);
    }
};

template<int S, int L>
Bundle<S, L> operator+(
    const Vec<S, L> &vec,
    const Bundle<S, L> &bundle
) {
    Bundle<S, L> result = bundle;

    for (int i = 0; i < S; i++) {
        int start = i * L;

        result[vec[i] + start]++;
    }

    return result;
}

template<int S, int L>
Bundle<S, L> operator*(
    float value,
    const Bundle<S, L> &bundle
) {
    Bundle<S, L> result;

    const int N = Bundle<S, L>::N;

    for (int i = 0; i < N; i++)
        result[i] = bundle[i] * value;

    return result;
}

template<int S, int L>
Bundle<S, L> operator*(
    float value,
    const Vec<S, L> &vec
) {
    Bundle<S, L> result = 0.0f;

    for (int i = 0; i < S; i++)
        result[vec[i] + L * i] = value;

    return result;
}
}
