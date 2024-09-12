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
    static const int N = S * L;

    int buffer[S];

public:
    Vec() {}

    Vec(
        int value
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

    int &operator[](
        int index
    ) {
        assert(index >= 0 && index < S);
        
        return buffer[index];
    }

    const int &operator[](
        int index
    ) const {
        assert(index >= 0 && index < S);
        
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
        int value
    ) {
        for (int i = 0; i < S; i++)
            buffer[i] = value;
    }
    
    Vec<S, L> operator*(
        const Vec<S, L> &other
    ) {
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
    ) {
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
    ) {
        Bundle<S, L> result = 0;

        for (int i = 0; i < S; i++) {
            int start = i * L;

            result[buffer[i] + start]++;
            result[other.buffer[i] + start]++;
        }

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
};

template<int S, int L>
class Bundle {
private:
    static const int N = S * L;

    int buffer[N];

public:
    Bundle()
    {}

    Bundle(
        int value
    ) {
        fill(value);
    }

    const Bundle<S, L> &operator=(
        int value
    ) {
        fill(value);

        return *this;
    }

    int &operator[](
        int index
    ) {
        assert(index >= 0 && index < N);
        
        return buffer[index];
    }

    const int &operator[](
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
        int value
    ) {
        for (int i = 0; i < N; i++)
            buffer[i] = value;
    }
    
    Bundle<S, L> operator+(
        const Bundle<S, L> &other
    ) {
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
    ) {
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

    Vec<S, L> thin() const {
        Vec<S, L> result;

        for (int i = 0; i < S; i++) {
            int start = i * L;

            int mv = 0;
            int mi = 0;

            for (int j = 0; j < L; j++) {
                int index = j + start;

                if (buffer[index] > mv) {
                    mv = buffer[index];
                    mi = j;
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
};
}
