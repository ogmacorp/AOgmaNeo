// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Helpers.h"

using namespace ogmaneo;

float ogmaneo::expf(float x) {
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

unsigned long ogmaneo::globalState = 1234;

unsigned int ogmaneo::rand(unsigned long* state) {
    return MWC64X(state);
}

float ogmaneo::randf(unsigned long* state) {
    return (rand(state) % 100000) / 99999.0f;
}

float ogmaneo::randf(float low, float high, unsigned long* state) {
    return low + (high - low) * randf(state);
}