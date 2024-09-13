// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "vec.h"

namespace aon {
template<int S, int L, int SH, int LH>
class Assoc {
private:
    static const int N = S * L;
    static const int NH = SH * LH;

    Byte* buffer;

public:
    static const int C = N * NH;

    Assoc()
    :
    buffer(nullptr)
    {}

    Assoc(
        Byte* buffer
    ) 
    :
    buffer(buffer)
    {}

    void set_from(
        Byte* buffer
    ) {
        this->buffer = buffer;
    }

    // number of segments
    int segments() const {
        return S;
    }

    // segment length
    int length() const {
        return L;
    }

    int vsize() const {
        return N;
    }

    // total size
    int size() const {
        return C;
    }

    void fill(
        Byte value
    ) {
        for (int i = 0; i < C; i++)
            buffer[i] = value;
    }
    
    Vec<S, L> operator*(
        const Vec<S, L> &other
    ) const {
        assert(buffer != nullptr);

        // activate hidden
        Vec<SH, LH> hidden;

        for (int hs = 0; hs < SH; hs++) {
            int max_index = 0;
            int max_sum = 0;

            for (int hl = 0; hl < LH; hl++) {
                int sum = 0;

                for (int vs = 0; vs < S; vs++)
                    sum += buffer[other[vs] + L * (vs + S * (hl + LH * hs))];

                if (sum > max_sum) {
                    max_sum = sum;
                    max_index = hl;
                }
            }

            hidden[hs] = max_index;
        }

        // reconstruct
        Vec<S, L> result;

        for (int vs = 0; vs < S; vs++) {
            int max_index = 0;
            int max_sum = 0;

            for (int vl = 0; vl < L; vl++) {
                int sum = 0;

                for (int hs = 0; hs < SH; hs++)
                    sum += buffer[vl + L * (vs + S * (hidden[hs] + LH * hs))];

                if (sum > max_sum) {
                    max_sum = sum;
                    max_index = vl;
                }
            }

            result[vs] = max_index;
        }

        return result;
    }

    void assoc(
        const Vec<S, L> &other,
        float lr,
        float variance,
        unsigned long* state = &global_state
    ) {
        assert(buffer != nullptr);

        // activate hidden
        Vec<SH, LH> hidden;

        for (int hs = 0; hs < SH; hs++) {
            int max_index = 0;
            int max_sum = 0;

            for (int hl = 0; hl < LH; hl++) {
                int sum = 0;

                for (int vs = 0; vs < S; vs++)
                    sum += buffer[other[vs] + L * (vs + S * (hl + LH * hs))];

                if (sum > max_sum) {
                    max_sum = sum;
                    max_index = hl;
                }
            }

            hidden[hs] = max_index;
        }

        // reconstruct
        Vec<S, L> result;
        Bundle<S, L> sums;

        const float scale = variance * sqrtf(1.0f / SH);

        for (int vs = 0; vs < S; vs++) {
            int max_index = 0;
            int max_sum = 0;

            for (int vl = 0; vl < L; vl++) {
                int sum = 0;

                for (int hs = 0; hs < SH; hs++)
                    sum += buffer[vl + L * (vs + S * (hidden[hs] + LH * hs))];

                sums[vl + L * vs] = sum;

                if (sum > max_sum) {
                    max_sum = sum;
                    max_index = vl;
                }
            }

            if (max_index != other[vs]) {
                for (int vl = 0; vl < L; vl++) {
                    float recon = expf(sums[vl + L * vs] * scale - 1.0f);

                    int delta = rand_roundf((vl == other[vs]) - recon, state);

                    for (int hs = 0; hs < SH; hs++) {
                        int index = vl + L * (vs + S * (hidden[hs] + LH * hs));

                        buffer[index] = min(255, max(0, buffer[index] + delta));
                    }
                }
            }
        }
    }
};
}
