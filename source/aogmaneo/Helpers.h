// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Ptr.h"
#include "Array.h"

#ifdef USE_STD_MATH
#include <cmath>
#include <algorithm>
#endif

namespace aon {
const int expIters = 5;
const int logIters = 3;
const int sinIters = 5;
const float pi = 3.14159f;
const float pi2 = pi * 2.0f;
const float byteInv = 1.0f / 255.0f;

// Lookup table for weight influence (byte weights)
const float weightLookupTable[257] {
    0.000000, 0.501961, 0.669281, 0.752941, 0.803137, 0.836601, 0.860504, 0.878431, 0.892375, 0.903529, 0.912656, 0.920261, 0.926697, 0.932213, 0.936993, 0.941176, 
    0.944867, 0.948148, 0.951084, 0.953725, 0.956116, 0.958289, 0.960273, 0.962092, 0.963765, 0.965309, 0.966739, 0.968067, 0.969304, 0.970458, 0.971537, 0.972549, 
    0.973500, 0.974394, 0.975238, 0.976035, 0.976789, 0.977503, 0.978180, 0.978824, 0.979436, 0.980019, 0.980575, 0.981105, 0.981612, 0.982097, 0.982562, 0.983007, 
    0.983433, 0.983843, 0.984237, 0.984615, 0.984980, 0.985330, 0.985668, 0.985994, 0.986309, 0.986613, 0.986906, 0.987190, 0.987464, 0.987729, 0.987986, 0.988235, 
    0.988477, 0.988711, 0.988938, 0.989158, 0.989372, 0.989580, 0.989782, 0.989978, 0.990169, 0.990355, 0.990536, 0.990712, 0.990884, 0.991051, 0.991214, 0.991373, 
    0.991527, 0.991679, 0.991826, 0.991970, 0.992111, 0.992248, 0.992382, 0.992513, 0.992642, 0.992767, 0.992889, 0.993009, 0.993127, 0.993242, 0.993354, 0.993464, 
    0.993572, 0.993677, 0.993781, 0.993882, 0.993982, 0.994079, 0.994175, 0.994268, 0.994360, 0.994451, 0.994539, 0.994626, 0.994711, 0.994795, 0.994877, 0.994958, 
    0.995037, 0.995115, 0.995192, 0.995267, 0.995341, 0.995414, 0.995485, 0.995556, 0.995625, 0.995693, 0.995760, 0.995825, 0.995890, 0.995954, 0.996017, 0.996078, 
    0.996139, 0.996199, 0.996258, 0.996316, 0.996373, 0.996430, 0.996485, 0.996540, 0.996594, 0.996647, 0.996699, 0.996751, 0.996802, 0.996852, 0.996901, 0.996950, 
    0.996998, 0.997045, 0.997092, 0.997138, 0.997184, 0.997229, 0.997273, 0.997317, 0.997360, 0.997403, 0.997445, 0.997486, 0.997527, 0.997568, 0.997608, 0.997647, 
    0.997686, 0.997725, 0.997763, 0.997800, 0.997837, 0.997874, 0.997910, 0.997946, 0.997981, 0.998016, 0.998051, 0.998085, 0.998119, 0.998152, 0.998185, 0.998217, 
    0.998250, 0.998282, 0.998313, 0.998344, 0.998375, 0.998406, 0.998436, 0.998465, 0.998495, 0.998524, 0.998553, 0.998582, 0.998610, 0.998638, 0.998665, 0.998693, 
    0.998720, 0.998747, 0.998773, 0.998800, 0.998826, 0.998851, 0.998877, 0.998902, 0.998927, 0.998952, 0.998976, 0.999000, 0.999024, 0.999048, 0.999072, 0.999095, 
    0.999118, 0.999141, 0.999164, 0.999186, 0.999208, 0.999230, 0.999252, 0.999274, 0.999295, 0.999316, 0.999337, 0.999358, 0.999379, 0.999399, 0.999420, 0.999440, 
    0.999460, 0.999479, 0.999499, 0.999518, 0.999538, 0.999557, 0.999576, 0.999594, 0.999613, 0.999631, 0.999650, 0.999668, 0.999686, 0.999703, 0.999721, 0.999739, 
    0.999756, 0.999773, 0.999790, 0.999807, 0.999824, 0.999841, 0.999857, 0.999873, 0.999890, 0.999906, 0.999922, 0.999938, 0.999953, 0.999969, 0.999985, 1.000000,
    1.0 // Dummy
};

inline float weightLookup(float w) {
    assert(w >= 0.0f && w <= 1.0f);

    w *= 255.0f;

    int index = static_cast<int>(w);
    float interp = w - index;

    return weightLookupTable[index] * (1.0f - interp) + weightLookupTable[index + 1] * interp;
}

inline float modf(
    float x,
    float y
);

float expf(
    float x
);

float logf(
    float x
);

float sinf(
    float x
);

float sqrtf(
    float x
);

float powf(
    float x,
    float y
);

inline float ceilf(
    float x
) {
    if (x > 0.0f)
        return (x - static_cast<int>(x)) > 0.0f ? static_cast<int>(x + 1) : static_cast<int>(x);

    return (x - static_cast<int>(x)) < 0.0f ? static_cast<int>(x - 1) : static_cast<int>(x);
}

inline int roundf(
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
#ifdef USE_STD_MATH
    return std::min(left, right);
#else
    if (left < right)
        return left;
    
    return right;
#endif
}

template <typename T>
T max(
    T left,
    T right
) {
#ifdef USE_STD_MATH
    return std::max(left, right);
#else
    if (left > right)
        return left;
    
    return right;
#endif
}

template <typename T>
T abs(
    T x
) {
#ifdef USE_STD_MATH
    return std::abs(x);
#else
    if (x >= 0)
        return x;

    return  -x;
#endif
}

template <typename T>
void swap(
    T &left,
    T &right
) {
    T temp = left;
    left = right;
    right = temp;
}

// OpenMP stuff, does nothing if USE_OMP is not set
void setNumThreads(
    int numThreads
);

int getNumThreads();

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

typedef unsigned char Byte;
typedef signed char SByte;
typedef Array<Byte> ByteBuffer;
typedef Array<SByte> SByteBuffer;
typedef Array<int> IntBuffer;
typedef Array<float> FloatBuffer;

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

    T &operator[](
        int index
    ) {
        return data[(start + index) % data.size()];
    }

    const T &operator[](
        int index
    ) const {
        return data[(start + index) % data.size()];
    }

    int size() const {
        return data.size();
    }
};

// --- Bounds ---

// Bounds check from (0, 0) to upperBound
inline bool inBounds0(
    const Int2 &pos,
    const Int2 &upperBound
) {
    return pos.x >= 0 && pos.x < upperBound.x && pos.y >= 0 && pos.y < upperBound.y;
}

// Bounds check in range
inline bool inBounds(
    const Int2 &pos,
    const Int2 &lowerBound,
    const Int2 &upperBound
) {
    return pos.x >= lowerBound.x && pos.x < upperBound.x && pos.y >= lowerBound.y && pos.y < upperBound.y;
}

// --- Projections ---

inline Int2 project(
    const Int2 &pos, // Position
    const Float2 &toScalars // Ratio of sizes
) {
    return Int2((pos.x + 0.5f) * toScalars.x, (pos.y + 0.5f) * toScalars.y);
}

inline Int2 projectf(
    const Float2 &pos, // Position
    const Float2 &toScalars // Ratio of sizes
) {
    return Int2((pos.x + 0.5f) * toScalars.x, (pos.y + 0.5f) * toScalars.y);
}

Int2 minOverhang(
    const Int2 &pos,
    const Int2 &size,
    const Int2 &radii
);

inline Int2 minOverhang(
    const Int2 &pos,
    const Int2 &size,
    int radius
) {
    return minOverhang(pos, size, Int2(radius, radius));
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
#ifdef USE_STD_MATH
    return std::tanh(x * 0.5f) * 0.5f + 0.5f;
#else
    if (x < 0.0f) {
        float z = expf(x);

        return z / (1.0f + z);
    }
    
    return 1.0f / (1.0f + expf(-x));
#endif
}

inline float tanh(
    float x
) {
#ifdef USE_STD_MATH
    return std::tanh(x);
#else
    if (x < 0.0f) {
        float z = expf(2.0f * x);

        return (z - 1.0f) / (z + 1.0f);
    }

    float z = expf(-2.0f * x);

    return -(z - 1.0f) / (z + 1.0f);
#endif
}

// --- RNG ---

extern unsigned int globalState;

const unsigned int randMax = 0x00003fff;

unsigned int rand(
    unsigned int* state = &globalState
);

float randf(
    unsigned int* state = &globalState
);

float randf(
    float low,
    float high,
    unsigned int* state = &globalState
);

// --- Sorting ---

template <typename T>
int partition(
    Array<T> &arr,
    int low,
    int high
) {
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
void quicksort(
    Array<T> &arr,
    int low = 0,
    int high = -1
) {
    if (high == -1)
        high = arr.size() - 1;
    else
        high--;

    if (high - low <= 1)
        return;

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
