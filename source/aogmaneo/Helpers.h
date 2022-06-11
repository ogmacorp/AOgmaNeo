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
    -5.545177, -4.852030, -4.446565, -4.158883, -3.935740, -3.753418, -3.599267, -3.465736, -3.347953, -3.242592, -3.147282, -3.060271, -2.980228, -2.906120, -2.837127, -2.772589, 
    -2.711964, -2.654806, -2.600738, -2.549445, -2.500655, -2.454135, -2.409683, -2.367124, -2.326302, -2.287081, -2.249341, -2.212973, -2.177882, -2.143980, -2.111190, -2.079442, 
    -2.048670, -2.018817, -1.989829, -1.961659, -1.934260, -1.907591, -1.881616, -1.856298, -1.831605, -1.807508, -1.783977, -1.760988, -1.738515, -1.716536, -1.695030, -1.673976, 
    -1.653357, -1.633154, -1.613352, -1.593934, -1.574886, -1.556193, -1.537844, -1.519826, -1.502126, -1.484734, -1.467640, -1.450833, -1.434304, -1.418043, -1.402043, -1.386294, 
    -1.370790, -1.355523, -1.340485, -1.325670, -1.311071, -1.296682, -1.282498, -1.268511, -1.254718, -1.241112, -1.227689, -1.214444, -1.201372, -1.188469, -1.175730, -1.163151, 
    -1.150728, -1.138458, -1.126337, -1.114361, -1.102526, -1.090830, -1.079269, -1.067841, -1.056541, -1.045368, -1.034318, -1.023389, -1.012578, -1.001883, -0.991301, -0.980829, 
    -0.970466, -0.960210, -0.950058, -0.940007, -0.930057, -0.920205, -0.910448, -0.900787, -0.891217, -0.881738, -0.872349, -0.863046, -0.853830, -0.844697, -0.835647, -0.826679, 
    -0.817790, -0.808979, -0.800245, -0.791587, -0.783004, -0.774493, -0.766054, -0.757686, -0.749387, -0.741156, -0.732993, -0.724896, -0.716864, -0.708896, -0.700990, -0.693147, 
    -0.685365, -0.677643, -0.669980, -0.662376, -0.654828, -0.647338, -0.639903, -0.632523, -0.625197, -0.617924, -0.610704, -0.603535, -0.596418, -0.589350, -0.582333, -0.575364, 
    -0.568444, -0.561571, -0.554745, -0.547965, -0.541231, -0.534542, -0.527898, -0.521297, -0.514740, -0.508225, -0.501752, -0.495321, -0.488932, -0.482582, -0.476273, -0.470004, 
    -0.463773, -0.457581, -0.451427, -0.445311, -0.439232, -0.433190, -0.427184, -0.421213, -0.415279, -0.409379, -0.403514, -0.397683, -0.391886, -0.386122, -0.380391, -0.374693, 
    -0.369028, -0.363394, -0.357792, -0.352221, -0.346680, -0.341171, -0.335691, -0.330242, -0.324822, -0.319431, -0.314069, -0.308735, -0.303430, -0.298153, -0.292904, -0.287682, 
    -0.282487, -0.277319, -0.272178, -0.267063, -0.261974, -0.256910, -0.251873, -0.246860, -0.241873, -0.236910, -0.231971, -0.227057, -0.222167, -0.217301, -0.212459, -0.207639, 
    -0.202843, -0.198070, -0.193319, -0.188591, -0.183885, -0.179201, -0.174539, -0.169899, -0.165280, -0.160682, -0.156106, -0.151550, -0.147015, -0.142500, -0.138006, -0.133531, 
    -0.129077, -0.124642, -0.120227, -0.115832, -0.111455, -0.107098, -0.102760, -0.098440, -0.094139, -0.089856, -0.085592, -0.081346, -0.077117, -0.072907, -0.068714, -0.064539, 
    -0.060381, -0.056240, -0.052116, -0.048009, -0.043919, -0.039846, -0.035789, -0.031749, -0.027725, -0.023717, -0.019725, -0.015748, -0.011788, -0.007843, -0.003914, 0.000000,
    0.0 // Dummy
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
