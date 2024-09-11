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
template<int S>
class Bundle;

template<int S>
class Vec {
private:
    static const int bs = (S + 7) / 8;
    static const int over_bits = 8 - (S % 8); // overflowing bit count

    Byte buffer[bs];

public:
    Vec() {}

    Vec(
        S_Byte value
    ) {
        fill(value);
    }

    static Vec<S> randomized(
        unsigned long* state = &global_state
    ) {
        Vec<S> result;

        for (int i = 0; i < bs; i++)
            result.buffer[i] = rand() % 256;

        return result;
    };

    void set(
        int index,
        S_Byte value
    ) {
        assert(index >= 0 && index < S);
        
        int byi = index / 8;
        int bi = index % 8;

        if (value > 0)
            buffer[byi] = buffer[byi] | (1 << bi);
        else
            buffer[byi] = buffer[byi] & (~(1 << bi));
    }

    S_Byte get(
        int index
    ) const {
        assert(index >= 0 && index < S);
        
        int byi = index / 8;
        int bi = index % 8;

        return ((buffer[byi] & (1 << bi)) != 0) * 2 - 1;
    }

    int size() const {
        return S;
    }

    // byte size
    int bsize() const {
        return bs;
    }

    void fill(
        S_Byte value
    ) {
        if (value > 0) {
            for (int i = 0; i < bs; i++)
                buffer[i] = 0xff;
        }
        else {
            for (int i = 0; i < bs; i++)
                buffer[i] = 0x0;
        }
    }
    
    Vec<S> operator*(
        const Vec<S> &other
    ) {
        Vec<S> result;

        for (int i = 0; i < bs; i++)
            result.buffer[i] = ~(buffer[i] ^ other.buffer[i]); 

        return result;
    }

    const Vec<S> &operator*=(
        const Vec<S> &other
    ) {
        for (int i = 0; i < bs; i++)
            buffer[i] = ~(buffer[i] ^ other.buffer[i]); 

        return *this;
    }

    Bundle<S> operator+(
        const Vec<S> &other
    ) {
        Bundle<S> result;

        for (int i = 0; i < bs; i++) {
            for (int j = 0; j < 8; j++) {
                int index = j + i * 8;

                if (index >= S)
                    return result;

                result[index] = ((((1 << j) & buffer[i]) != 0) + (((1 << j) & other.buffer[i]) != 0)) * 2 - 2;
            }
        }

        return result;
    }

    Vec<S> permute(
        int shift = 1
    ) {
        Vec<S> result;

        for (int i = 0; i < bs; i++)
            result.set(i) = get((i + S + shift) % S);

        return result;
    }

    int dot(
        const Vec<S> &other
    ) {
        int sum = 0;

        for (int i = 0; i < bs; i++) {
            for (int j = 0; j < 8; j++) {
                int index = j + i * 8;

                if (index >= S)
                    return sum;

                sum += ((((1 << j) & buffer[i]) != 0) * 2 - 1) * ((((1 << j) & other.buffer[i]) != 0) * 2 - 1);
            }
        }

        return sum;
    }
};

template<int S>
class Bundle {
private:
    int buffer[S];

public:
    Bundle() {
        fill(0);
    }

    Bundle(
        int value
    ) {
        fill(value);
    }

    const Bundle<S> &operator=(
        int value
    ) {
        fill(value);

        return *this;
    }

    void set(
        int index,
        int value
    ) const {
        assert(index >= 0 && index < S);
        
        buffer[index] = value;
    }

    int get(
        int index
    ) const {
        assert(index >= 0 && index < S);
        
        return buffer[index];
    }

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

    int size() const {
        return S;
    }

    void fill(
        int value
    ) {
        for (int i = 0; i < S; i++)
            buffer[i] = value;
    }
    
    Bundle<S> operator+(
        const Bundle<S> &other
    ) {
        Bundle<S> result;

        for (int i = 0; i < S; i++)
            result[i] = buffer[i] + other.buffer[i];

        return result;
    }

    const Bundle<S> &operator+=(
        const Bundle<S> &other
    ) {
        for (int i = 0; i < S; i++)
            buffer[i] += other.buffer[i];

        return *this;
    }

    Bundle<S> operator+(
        const Vec<S> &other
    ) {
        Bundle<S> result;

        for (int i = 0; i < S; i++)
            result[i] = buffer[i] + other.get(i);

        return result;
    }

    const Bundle<S> &operator+=(
        const Vec<S> &other
    ) {
        for (int i = 0; i < S; i++)
            buffer[i] += other.get(i);

        return *this;
    }

    Vec<S> thin() const {
        Vec<S> result;

        for (int i = 0; i < S; i++)
            result.set(i, (buffer[i] > 0) * 2 - 1); 

        return result;
    }
};
}
