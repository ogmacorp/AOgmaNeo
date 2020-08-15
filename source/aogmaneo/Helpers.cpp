// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Helpers.h"

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
float aon::sqrt(
    float x
) {
    const float xhalf = 0.5f*x;
    
    union
    {
        float x;
        int i;
    } u;

    u.x = x;
    u.i = 0x5f3759df - (u.i >> 1);

    return x * u.x * (1.5f - xhalf * u.x * u.x);
} 

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