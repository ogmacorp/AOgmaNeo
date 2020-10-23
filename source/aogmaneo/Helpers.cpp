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

unsigned int aon::globalState = 1234;

unsigned int aon::rand(
    unsigned int* state
) {
    *state = (*state * 1103515245 + 12345) % (1u << 31);

    return *state & randMax;
}

float aon::randf(
    unsigned int* state
) {
    return static_cast<float>(rand(state)) / static_cast<float>(randMax - 1);
}

float aon::randf(
    float low,
    float high,
    unsigned int* state
) {
    return low + (high - low) * randf(state);
}
