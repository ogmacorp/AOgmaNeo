// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2025 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <assert.h>

namespace aon {
template<typename T>
class Array_View;

template<typename T>
class Array {
private:
    T* p;
    int s;

public:
    Array()
    :
    p(nullptr),
    s(0)
    {}

    Array(
        const Array<T> &other
    )
    :
    p(nullptr),
    s(0)
    {
        *this = other;
    }

    Array(
        int size
    )
    :
    p(nullptr),
    s(0)
    {
        resize(size);
    }

    Array(
        int size,
        T value
    )
    :
    p(nullptr),
    s(0)
    {
        resize(size, value);
    }

    ~Array() {
        if (p != nullptr)
            delete[] p;
    }

    Array<T> &operator=(
        const Array<T> &other
    ) {
        if (s != other.s) {
            if (p != nullptr)
                delete[] p;

            s = other.s;

            p = new T[static_cast<unsigned int>(s)];
        }

        for (int i = 0; i < s; i++)
            p[i] = other.p[i];
        
        return *this;
    }

    Array<T> &operator=(
        const Array_View<T> &other
    ) {
        if (s != other.s) {
            if (p != nullptr)
                delete[] p;

            s = other.s;

            p = new T[static_cast<unsigned int>(s)];
        }

        for (int i = 0; i < s; i++)
            p[i] = other.p[i];
        
        return *this;
    }

    void resize(
        int size
    ) {
        if (s == size)
            return;

        T* temp = p;
        int old_size = s;
        
        s = size;

        if (s != 0) {
            p = new T[static_cast<unsigned int>(s)];

            int ms = old_size < s ? old_size : s;

            for (int i = 0; i < ms; i++)
                p[i] = temp[i];
        }
        else
            p = nullptr;

        if (temp != nullptr)
            delete[] temp;
    }

    void resize(
        int size,
        T value
    ) {
        int old_s = s;

        resize(size);

        for (int i = old_s; i < s; i++)
            p[i] = value;
    }

    T &operator[](
        int index
    ) {
        assert(index >= 0 && index < s);

        return p[index];
    }

    const T &operator[](
        int index
    ) const {
        assert(index >= 0 && index < s);
        
        return p[index];
    }

    int size() const {
        return s;
    }

    void fill(
        T value
    ) {
        for (int i = 0; i < s; i++)
            p[i] = value;
    }

    friend Array_View<T>;
};

template<typename T>
class Array_View {
private:
    T* p;
    int s;

public:
    Array_View()
    :
    p(nullptr),
    s(0)
    {}

    Array_View(
        const Array_View<T> &other
    ) {
        *this = other;
    }

    Array_View(
        const Array<T> &other
    ) {
        *this = other;
    }

    Array_View<T> &operator=(
        const Array_View<T> &other
    ) {
        p = other.p;
        s = other.s;
        
        return *this;
    }

    Array_View<T> &operator=(
        const Array<T> &other
    ) {
        p = other.p;
        s = other.s;
        
        return *this;
    }

    T &operator[](
        int index
    ) {
        assert(index >= 0 && index < s);

        return p[index];
    }

    const T &operator[](
        int index
    ) const {
        assert(index >= 0 && index < s);
        
        return p[index];
    }

    int size() const {
        return s;
    }

    void fill(
        T value
    ) {
        for (int i = 0; i < s; i++)
            p[i] = value;
    }

    friend Array<T>;
};
}
