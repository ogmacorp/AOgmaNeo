// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Helpers.h"

#ifdef USE_OMP
#include <omp.h>
#endif

using namespace aon;

float aon::expf(
    float x
) {
    if (x > 0.0f) {
        float res = 1.0f;

        float p = x;

        for (int i = 0; i < expIters; i++) {
            res += p / expFactorials[i];

            p *= x;
        }

        return res;
    }

    float res = 1.0f;

    float p = -x;

    for (int i = 0; i < expIters; i++) {
        res += p / expFactorials[i];

        p *= -x;
    }

    return 1.0f / res;
}

// Quake method
float aon::sqrtf(
    float x
) {
    const float xHalf = 0.5f * x;
 
    union {
        float x;
        int i;
    } u;

    u.x = x;
    u.i = 0x5f3759df - (u.i >> 1);

    return x * u.x * (1.5f - xHalf * u.x * u.x);
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

unsigned long aon::globalState = 1234;

unsigned int aon::rand(
    unsigned long* state
) {
    return MWC64X(state);
}

float aon::randf(
    unsigned long* state
) {
    return (rand(state) % 100000) / 99999.0f;
}

float aon::randf(
    float low,
    float high,
    unsigned long* state
) {
    return low + (high - low) * randf(state);
}
