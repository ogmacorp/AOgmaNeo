// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "helpers.h"
#include "vec.h"

namespace aon {
struct Layer_Params {
    int iters;

    Layer_Params()
    :
    iters(3)
    {}
};

// take 2 vectors and map to 1
template<int S, int L>
class Predictor {
private:
    int hidden_segments;
    int hidden_length;

    Byte_Buffer weights;
    Int_Buffer commits;

    Int_Buffer sums;

public:
    static const int N = S * L;

    Predictor()
    {}

    Predictor(
        int hidden_segments,
        int hidden_length
    ) {
        set_from(hidden_segments, hidden_length);
    }

    void set_from(
        int hidden_segments,
        int hidden_length
    ) {
        this->hidden_segments = hidden_segments;
        this->hidden_length = hidden_length;

        int num_hidden = hidden_segments * hidden_length;
        int num_weights = num_hidden * N * 2;

        weights = Byte_Buffer((num_weights + 7) / 8, 0);
        commits = Int_Buffer(hidden_segments, 0);
    }

    // number of segments
    static int segments() {
        return S;
    }

    // segment length
    static int length() {
        return L;
    }

    static int vsize() {
        return N;
    }

    Vec<S, L> predict(
        const Vec<S, L> &src,
        const Layer_Params &params
    ) {
        int num_hidden = hidden_segments * hidden_length;

        for (int i = 0; i < hidden_segments; i++)
            sums[i] = 0;

        for (int vs = 0; vs < S; vs++) {
            int sindex = src[vs] + L * vs;

            for (int hi = 0; hi < num_hidden; hi++) {
                int wi = hi + num_hidden * sindex;

                int byi = wi / 8;
                int bi = wi % 8;

                sums[hi] += ((weights[wi] & (1 << bi)) != 0);
            }
        }

        Vec<S, L> result;

        for (int hs = 0; hs < S; hs++) {
            int max_index = 0;
            int max_activation = limit_min;

            for (int hl = 0; hl < L; hl++) {
                int hi = hl + L * hs;

                float complemented = (
                float activation = 
                if (sums[hi] > max_activation) {
                    max_activation = sums[hi];
                    max_index = hl;
                }
            }

            result[hs] = max_index;
        }

        return result;
    }

    // reqires predict to have been called first
    void learn(
        const Vec<S, L> &src,
        const Vec<S, L> &pred,
        const Vec<S, L> &target,
        unsigned long* state,
        const Layer_Params &params
    ) {
        assert(data != nullptr);

        const float rate = params.lr * 255.0f;

        // update output weights
        for (int vs = 0; vs < S; vs++) {
            int sindex = src[vs] + L * vs;

            const int delta = rand_roundf(rate, state);

            for (int os = 0; os < S; os++) {
                if (target[os] == pred[os])
                    continue;

                int wi_target = (target[os] + L * os) + N * sindex;
                int wi_pred = (pred[os] + L * os) + N * sindex;

                weights[wi_target] = min(255, weights[wi_target] + delta);
                weights[wi_pred] = max(0, weights[wi_pred] - delta);
            }
        }
    }
};
}
