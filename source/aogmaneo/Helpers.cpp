// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Helpers.h"

#ifdef USE_OMP
#include <omp.h>
#endif

using namespace aon;

float aon::modf(
    float x,
    float y
) {
    return x - static_cast<int>(x / y) * y;
}

float aon::expf(
    float x
) {
#ifdef USE_STD_MATH
    return std::exp(x);
#else
    if (x > 0.0f) {
        float p = x;
        int f = 1;

        float res = 1.0f + x;

        for (int n = 2; n <= expIters; n++) {
            p *= x;
            f *= n;

            res += p / f;
        }

        return res;
    }

    float p = -x;
    int f = 1;

    float res = 1.0f - x;

    for (int n = 2; n <= expIters; n++) {
        p *= -x;
        f *= n;

        res += p / f;
    }

    return 1.0f / res;
#endif
}

float aon::sinf(
    float x
) {
#ifdef USE_STD_MATH
    return std::sin(x);
#else
    x = modf(x, pi2);

    if (x < -pi)
        x += pi2;
    else if (x > pi)
        x -= pi2;

    float p = x;
    int f = 1;

    float res = x;

    for (int n = 1; n <= sinIters; n++) {
        p *= -x * x;

        int f1 = n * 2;

        f *= f1 * (f1 + 1);

        res += p / f;
    }

    return res;
#endif
}

float aon::sqrtf(
    float x
) {
#ifdef USE_STD_MATH
    return std::sqrt(x);
#else
    // Quake method
    union {
        float x;
        int i;
    } u;

    u.x = x;
    u.i = 0x5f3759df - (u.i >> 1);

    return x * u.x * (1.5f - 0.5f * x * u.x * u.x);
#endif
}

#ifdef USE_OMP
void aon::setNumThreads(
    int numThreads
) {
    omp_set_num_threads(numThreads);
}

int aon::getNumThreads() {
    return omp_get_num_threads();
}
#else
void aon::setNumThreads(
    int numThreads
) {}

int aon::getNumThreads() {
    return 0;
}
#endif

Int2 aon::minOverhang(
    const Int2 &pos,
    const Int2 &size,
    const Int2 &radii
) {
    Int2 newPos = pos;

    bool overhangPX = (newPos.x + radii.x >= size.x);
    bool overhangNX = (newPos.x - radii.x < 0);
    bool overhangPY = (newPos.y + radii.y >= size.y);
    bool overhangNY = (newPos.y - radii.y < 0);

    if (overhangPX && !overhangNX)
       newPos.x = size.x - 1 - radii.x;
    else if (overhangNX && !overhangPX)
       newPos.x = radii.x;

    if (overhangPY && !overhangNY)
       newPos.y = size.y - 1 - radii.y;
    else if (overhangNY && !overhangPY)
       newPos.y = radii.y;

    return newPos;
}

unsigned int aon::globalState = 123456;

unsigned int aon::rand(
    unsigned int* state
) {
    *state = ((*state) * 1103515245 + 12345) % (1u << 31);

    return (*state >> 16) & randMax;
}

float aon::randf(
    unsigned int* state
) {
    return static_cast<float>(rand(state)) / static_cast<float>(randMax);
}

float aon::randf(
    float low,
    float high,
    unsigned int* state
) {
    return low + (high - low) * randf(state);
}
