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
template<int S, int L>
class Encoder;

template<int S, int L>
class Assoc {
private:
    static const int N = S * L;

    int HS;
    int HL;

    Byte* weights;
    int* hiddens;

public:
    int C;

    Assoc()
    :
    weights(nullptr),
    hiddens(nullptr),
    HS(0),
    HL(0),
    C(0)
    {}

    Assoc(
        int HS,
        int HL,
        Byte* weights,
        int* hiddens
    ) {
        set_from(HS, HL, weights, hiddens);
    }

    void set_from(
        int HS,
        int HL,
        Byte* weights,
        int* hiddens
    ) {
        this->HS = HS;
        this->HL = HL;

        int HN = HS * HL;
        C = N * HN;

        this->weights = weights;
        this->hiddens = hiddens;
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
    
    Vec<S, L> multiply(
        const Vec<S, L> &other,
        const typename Encoder<S, L>::Params &params
    ) const {
        assert(weights != nullptr);
        assert(hiddens != nullptr);

        // activate hidden
        for (int hs = 0; hs < HS; hs++) {
            int max_index = 0;
            int max_sum = 0;

            for (int hl = 0; hl < HL; hl++) {
                int sum = 0;

                for (int vs = 0; vs < S; vs++)
                    sum += weights[other[vs] + L * (vs + S * (hl + HL * hs))];

                if (sum > max_sum) {
                    max_sum = sum;
                    max_index = hl;
                }
            }

            hiddens[hs] = max_index;
        }

        // reconstruct
        Vec<S, L> result;

        for (int vs = 0; vs < S; vs++) {
            int max_index = 0;
            int max_sum = 0;

            for (int vl = 0; vl < L; vl++) {
                int sum = 0;

                for (int hs = 0; hs < HS; hs++)
                    sum += weights[vl + L * (vs + S * (hiddens[hs] + HL * hs))];

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
        const typename Encoder<S, L>::Params &params
    ) {
        assert(weights != nullptr);
        assert(hiddens != nullptr);

        // activate hidden
        for (int hs = 0; hs < HS; hs++) {
            int max_index = 0;
            int max_sum = 0;

            for (int hl = 0; hl < HL; hl++) {
                int sum = 0;

                for (int vs = 0; vs < S; vs++)
                    sum += weights[other[vs] + L * (vs + S * (hl + HL * hs))];

                if (sum > max_sum) {
                    max_sum = sum;
                    max_index = hl;
                }
            }

            hiddens[hs] = max_index;
        }

        // reconstruct
        const float scale = params.variance * sqrtf(1.0f / HS) / 127.0f;

        for (int vs = 0; vs < S; vs++) {
            for (int vl = 0; vl < L; vl++) {
                int sum = 0;

                for (int hs = 0; hs < HS; hs++)
                    sum += weights[vl + L * (vs + S * (hiddens[hs] + HL * hs))];

                float recon = max(0.0f, 1.0f + min(0, sum - 127 * HS) * scale);

                int delta = roundf(params.lr * 255.0f * ((vl == other[vs]) - recon));

                for (int hs = 0; hs < HS; hs++) {
                    int index = vl + L * (vs + S * (hiddens[hs] + HL * hs));

                    weights[index] = min(255, max(0, weights[index] + delta));
                }
            }
        }
    }
};
}
