// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Ptr.h"
#include "Array.h"
#include <omp.h>

namespace aon {
const int expIters = 10;
const float expFactorials[] = { 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800 };

float expf(float x);

inline float floorf(
    float x
) {
    return static_cast<int>(x);
}

inline float ceilf(
    float x
) {
    if (x > 0.0f)
        return (x - static_cast<int>(x)) > 0 ? static_cast<int>(x + 1) : static_cast<int>(x);

    return (x - static_cast<int>(x)) < 0 ? static_cast<int>(x - 1) : static_cast<int>(x);
}

inline int roundftoi(
    float x
) {
    if (x > 0.0f)
        return static_cast<int>(x + 0.5f);

    return -static_cast<int>(-x + 0.5f);
}

template <typename T>
T min(
    T left,
    T right
) {
    if (left < right)
        return left;
    
    return right;
}

template <typename T>
T max(
    T left,
    T right
) {
    if (left > right)
        return left;
    
    return right;
}

template <typename T>
T abs(
    T x
) {
    if (x >= 0)
        return x;

    return  -x;
}

template <typename T>
void swap(T &left, T &right) {
    T temp = left;
    left = right;
    right = temp;
}

inline void setNumThreads(int numThreads) {
    omp_set_num_threads(numThreads);
}

inline int getNumThreads() {
    return omp_get_num_threads();
}

// Vector types
template <typename T> 
struct Vec2 {
    T x, y;

    Vec2() {}

    Vec2(
        T x,
        T y
    )
    : x(x), y(y)
    {}
};

template <typename T> 
struct Vec3 {
    T x, y, z;
    T pad;

    Vec3()
    {}

    Vec3(
        T x,
        T y,
        T z
    )
    : x(x), y(y), z(z)
    {}
};

template <typename T> 
struct Vec4 {
    T x, y, z, w;

    Vec4()
    {}

    Vec4(
        T x,
        T y,
        T z,
        T w
    )
    : x(x), y(y), z(z), w(w)
    {}
};

// Some basic definitions
typedef Vec2<int> Int2;
typedef Vec3<int> Int3;
typedef Vec4<int> Int4;
typedef Vec2<float> Float2;
typedef Vec3<float> Float3;
typedef Vec4<float> Float4;

typedef Array<int> IntBuffer;
typedef Array<float> FloatBuffer;
typedef Array<unsigned char> ByteBuffer;

typedef unsigned char ColSize8;
typedef unsigned short ColSize16;
typedef unsigned int ColSize32;
typedef unsigned long ColSize64;

// --- Circular Buffer ---

template <typename T> 
struct CircleBuffer {
    Array<T> data;
    int start;

    CircleBuffer()
    :
    start(0)
    {}

    void resize(
        int size
    ) {
        data.resize(size);
    }

    void pushFront() {
        start--;

        if (start < 0)
            start += data.size();
    }

    T &front() {
        return data[start];
    }

    const T &front() const {
        return data[start];
    }

    T &back() {
        return data[(start + data.size() - 1) % data.size()];
    }

    const T &back() const {
        return data[(start + data.size() - 1) % data.size()];
    }

    T &operator[](int index) {
        return data[(start + index) % data.size()];
    }

    const T &operator[](int index) const {
        return data[(start + index) % data.size()];
    }

    int size() const {
        return data.size();
    }
};

// --- Bounds ---

// Bounds check from (0, 0) to upperBound
inline bool inBounds0(
    const Int2 &pos, // Position
    const Int2 &upperBound // Bottom-right corner
) {
    return pos.x >= 0 && pos.x < upperBound.x && pos.y >= 0 && pos.y < upperBound.y;
}

// Bounds check in range
inline bool inBounds(
    const Int2 &pos, // Position
    const Int2 &lowerBound, // Top-left corner
    const Int2 &upperBound // Bottom-right corner
) {
    return pos.x >= lowerBound.x && pos.x < upperBound.x && pos.y >= lowerBound.y && pos.y < upperBound.y;
}

// --- Projections ---

inline Int2 project(
    const Int2 &pos, // Position
    const Float2 &toScalars // Ratio of sizes
) {
    return Int2(pos.x * toScalars.x + 0.5f, pos.y * toScalars.y + 0.5f);
}

inline Int2 projectf(
    const Float2 &pos, // Position
    const Float2 &toScalars // Ratio of sizes
) {
    return Int2(pos.x * toScalars.x + 0.5f, pos.y * toScalars.y + 0.5f);
}

// --- Addressing ---

// Row-major
inline int address2(
    const Int2 &pos, // Position
    const Int2 &dims // Dimensions to ravel with
) {
    return pos.y + pos.x * dims.y;
}

inline int address3(
    const Int3 &pos, // Position
    const Int3 &dims // Dimensions to ravel with
) {
    return pos.z + dims.z * (pos.y + dims.y * pos.x);
}

inline int address4(
    const Int4 &pos, // Position
    const Int4 &dims // Dimensions to ravel with
) {
    return pos.w + dims.w * (pos.z + dims.z * (pos.y + dims.y * pos.x));
}

// --- Getters ---

template <typename T>
Array<Array<T>*> get(
    Array<Array<T>> &v
) {
    Array<T*> vp(v.size());

    for (int i = 0; i < v.size(); i++)
        vp[i] = &v[i];

    return vp;
}

template <typename T>
Array<const Array<T>*> constGet(
    const Array<Array<T>> &v
) {
    Array<const Array<T>*> vp(v.size());

    for (int i = 0; i < v.size(); i++)
        vp[i] = &v[i];

    return vp;
}

template <typename T>
Array<Array<T>*> get(
    CircleBuffer<Array<T>> &v
) {
    Array<T*> vp(v.size());

    for (int i = 0; i < v.size(); i++)
        vp[i] = &v[i];

    return vp;
}

template <typename T>
Array<const Array<T>*> constGet(
    const CircleBuffer<Array<T>> &v
) {
    Array<const Array<T>*> vp(v.size());

    for (int i = 0; i < v.size(); i++)
        vp[i] = &v[i];

    return vp;
}

// --- Noninearities ---

inline float sigmoid(
    float x
) {
    if (x < 0.0f) {
        float z = expf(x);

        return z / (1.0f + z);
    }
    
    return 1.0f / (1.0f + expf(-x));
}

// --- RNG ---

// From http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html

extern unsigned long globalState;

inline unsigned int MWC64X(
    unsigned long* state
) {
    unsigned int c = (*state) >> 32, x = (*state) & 0xffffffff;

    *state = x * ((unsigned long)4294883355u) + c;

    return x ^ c;
}

unsigned int rand(
    unsigned long* state = &globalState
);

float randf(
    unsigned long* state = &globalState
);

float randf(
    float low,
    float high,
    unsigned long* state = &globalState
);

template <typename T>
T randBits(
    unsigned long* state = &globalState
) {
    unsigned char numBytes = sizeof(T);

    T out = 0;

    for (unsigned char i = 0; i < numBytes; i++) {
        unsigned char b = rand(state) % 256;

        out = out | b;

        if (i < numBytes - 1)
            out <<= 8;
    }

    return out;
}

template <typename T>
T randBits(
    float ratio,
    unsigned long* state = &globalState
) {
    unsigned char numBits = sizeof(T) * 8;

    T out = 0;

    for (unsigned char i = 0; i < numBits; i++) {
        out = out | (randf(state) < ratio ? 1 : 0);

        if (i < numBits - 1)
            out <<= 1;
    }

    return out;
}

// --- Weight mutation ---

template <typename T>
T increase(
    T weight,
    unsigned char index
) {
    return weight | (1 << index);
}

template <typename T>
T decrease(
    T weight,
    unsigned char index
) {
    return weight & ~(1 << index);
}

template <typename T>
unsigned char countBits(
    T value
) {
    unsigned char numBits = sizeof(T) * 8;

    unsigned char c = 0;

    for (unsigned char i = 0; i < numBits; i++) {
        c += 1 & value;
        value >>= 1;
    }

    return c;
}

// --- Sorting ---

template <typename T>
int partition(Array<T> &arr, int low, int high) {
    T pivotVal = arr[high];

    int i = low;

    for (int j = low; j < high; j++) {
        if (arr[j] < pivotVal) {
            swap<T>(arr[i], arr[j]);
            i++;
        }
    }

    swap<T>(arr[i], arr[high]);

    return i;
}

// In-place
template <typename T>
void quicksort(Array<T> &arr, int low = 0, int high = -1) {
    if (arr.size() <= 1)
        return;

    if (high == -1)
        high = arr.size() - 1;
    else
        high--;

    Array<int> stack(high - low + 1);

    int top = 0;

    stack[top] = low;
    top++;
    stack[top] = high;

    while (top >= 0) {
        high = stack[top--];
        low = stack[top--];

        int pivotIndex = partition<T>(arr, low, high);

        if (pivotIndex - 1 > low) {
            top++;
            stack[top] = low;
            top++;
            stack[top] = pivotIndex - 1;
        }

        if (pivotIndex + 1 < high) {
            top++;
            stack[top] = pivotIndex + 1;
            top++;
            stack[top] = high;
        }
    }
}

// --- Serialization ---

class StreamWriter {
public:
    virtual ~StreamWriter() {}

    virtual void write(
        const void* data,
        int len
    ) = 0;
};

class StreamReader {
public:
    virtual ~StreamReader() {}

    virtual void read(
        void* data,
        int len
    ) = 0;
};
} // namespace aon