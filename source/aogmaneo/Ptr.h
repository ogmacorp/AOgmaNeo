// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

namespace aon {
template<class T>
class Ptr {
private:
    T* p;
    
public:
    Ptr()
    :
    p(nullptr)
    {}

    Ptr(
        const Ptr &other
    ) {
        p = nullptr;

        *this = other;
    }

    ~Ptr() {
        if (p != nullptr)
            delete p;
    }

    Ptr &operator=(
        const Ptr &other
    ) {
        if (p != nullptr)
            delete p;

        if (other.p != nullptr) {
            p = new T();

            *p = *other.p;
        }
        else
            p = nullptr;

        return *this;
    }

    Ptr &operator=(
        T* other
    ) {
        if (p != nullptr)
            delete p;

        p = other;

        return *this;
    }

    void make() {
        if (p != nullptr)
            delete p;
            
        p = new T();
    }

    T* get() {
        return p;
    }

    const T* get() const {
        return p;
    }

    T* operator->() {
        return p;
    }

    const T* operator->() const {
        return p;
    }

    T &operator*() {
        return *p;
    }

    const T &operator*() const {
        return *p;
    }

    bool operator()() const {
        return p;
    }

    bool operator==(
        const Ptr &other
    ) const {
        return p == other.p;
    }

    bool operator!=(
        const Ptr &other
    ) const {
        return p != other.p;
    }

    bool operator==(
        T* other
    ) const {
        return p == other;
    }

    bool operator!=(
        T* other
    ) const {
        return p != other;
    }
};
} // namespace aon
