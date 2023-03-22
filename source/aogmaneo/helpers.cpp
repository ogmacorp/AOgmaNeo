// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "helpers.h"

#ifdef use_omp
#include <omp.h>
#endif

using namespace aon;

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

        for (int n = 2; n <= exp_iters; n++) {
            p *= x;
            f *= n;

            res += p / f;
        }

        return res;
    }

    float p = -x;
    int f = 1;

    float res = 1.0f - x;

    for (int n = 2; n <= exp_iters; n++) {
        p *= -x;
        f *= n;

        res += p / f;
    }

    return 1.0f / res;
#endif
}

float aon::logf(
    float x
) {
#ifdef USE_STD_MATH
    return std::log(x);
#else
    if (x <= 0.0f)
        return -999999.0f;

    float res = 1.0f; // initial guess

    for (int n = 1; n <= log_iters; n++) {
        float ey = expf(res);

        res += 2.0f * (x - ey) / (x + ey);
    }

    return res;
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

    for (int n = 1; n <= sin_iters; n++) {
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
    // quake method
    union {
        float x;
        int i;
    } u;

    u.x = x;
    u.i = 0x5f3759df - (u.i >> 1);

    return x * u.x * (1.5f - 0.5f * x * u.x * u.x);
#endif
}

float aon::powf(
    float x,
    float y
) {
#ifdef USE_STD_MATH
    return std::pow(x, y);
#else
    return expf(y * logf(x));
#endif
}

#ifdef use_omp
void aon::set_num_threads(
    int num_threads
) {
    omp_set_num_threads(num_threads);
}

int aon::get_num_threads() {
    return omp_get_num_threads();
}
#else
void aon::set_num_threads(
    int num_threads
) {}

int aon::get_num_threads() {
    return 0;
}
#endif

Int2 aon::min_overhang(
    const Int2 &pos,
    const Int2 &size,
    const Int2 &radii
) {
    Int2 new_pos = pos;

    bool overhang_px = (new_pos.x + radii.x >= size.x);
    bool overhang_nx = (new_pos.x - radii.x < 0);
    bool overhang_py = (new_pos.y + radii.y >= size.y);
    bool overhang_ny = (new_pos.y - radii.y < 0);

    if (overhang_px && !overhang_nx)
       new_pos.x = size.x - 1 - radii.x;
    else if (overhang_nx && !overhang_px)
       new_pos.x = radii.x;

    if (overhang_py && !overhang_ny)
       new_pos.y = size.y - 1 - radii.y;
    else if (overhang_ny && !overhang_py)
       new_pos.y = radii.y;

    return new_pos;
}

unsigned int aon::global_state = 123456;

unsigned int aon::rand(
    unsigned int* state
) {
    *state = ((*state) * 1103515245 + 12345) % (1u << 31);

    return (*state >> 16) & rand_max;
}

float aon::randf(
    unsigned int* state
) {
    return static_cast<float>(rand(state)) / static_cast<float>(rand_max);
}

float aon::randf(
    float low,
    float high,
    unsigned int* state
) {
    return low + (high - low) * randf(state);
}
