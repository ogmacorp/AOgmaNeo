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
class Predictor {
private:
    Byte* weights;
    float* hiddens;

public:
    static const int N = S * L;
    static const int C = N * N;

    Predictor()
    :
    weights(nullptr),
    hiddens(nullptr)
    {}

    Predictor(
        Byte* weights,
        int* hiddens
    ) {
        set_from(weights, hiddens);
    }

    void set_from(
        Byte* weights,
        int* hiddens
    ) {
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
        const Vec<S, L> &src,
        const typename Encoder<S, L>::Params &params
    ) const {
        assert(weights != nullptr);
        assert(hiddens != nullptr);

        Vec<S, L> result;

        // activate
        for (int hs = 0; hs < S; hs++) {
            int max_index = 0;
            int max_sum = 0;

            for (int hl = 0; hl < L; hl++) {
                int hindex = hl + L * hs;

                int sum = 0;

                for (int vs = 0; vs < S; vs++)
                    sum += weights[src[vs] + L * (vs + S * hindex)];

                if (sum > max_sum) {
                    max_sum = sum;
                    max_index = hl;
                }
            }

            result[hs] = max_index;
        }

        return result;
    }

    void learn(
        const Vec<S, L> &src,
        const Vec<S, L> &target,
        const typename Encoder<S, L>::Params &params
    ) {
        assert(weights != nullptr);
        assert(hiddens != nullptr);

        const float byte_inv = 1.0f / 255.0f;

        // activate hidden
        for (int hs = 0; hs < S; hs++) {
            float max_hidden = 0.0f;

            for (int hl = 0; hl < L; hl++) {
                int hindex = hl + L * hs;

                int sum = 0;

                for (int vs = 0; vs < S; vs++)
                    sum += weights[src[vs] + L * (vs + S * hindex)];

                hiddens[hindex] = static_cast<float>(sum) * byte_inv / S * params.scale;

                max_hidden = max(max_hidden, hiddens[hindex]);
            }

            // softmax
            float total = 0.0f;

            for (int hl = 0; hl < L; hl++) {
                int hindex = hl + L * hs;

                hiddens[hindex] = expf(hiddens[hindex] - max_hidden);

                total += hiddens[hindex];
            }

            float total_inv = 1.0f / max(limit_small, total);

            for (int hl = 0; hl < L; hl++) {
                int hindex = hl + L * hs;

                hiddens[hindex] *= total_inv;

                // learn on target
                float error = (hl == target[hs]) - hiddens[hindex];

                int delta = roundf(params.lr * 255.0f * error);

                for (int vs = 0; vs < S; vs++) {
                    int wi = src[vs] + L * (vs + S * hindex);

                    weights[wi] = min(255, max(0, weights[wi] + delta));
                }
            }
        }
    }
};
}