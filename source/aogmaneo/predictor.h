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
class Layer;

template<int S, int L>
class Predictor {
private:
    Byte* weights;
    float* dendrite_acts;
    float* output_acts;

    int D;
    int C;

public:
    static const int N = S * L;

    Predictor()
    :
    weights(nullptr),
    dendrite_acts(nullptr),
    output_acts(nullptr)
    {}

    Predictor(
        int D,
        Byte* weights,
        float* dendrite_acts,
        float* output_acts
    ) {
        set_from(D, weights, dendrite_acts, output_acts);
    }

    void set_from(
        int D,
        Byte* weights,
        float* dendrite_acts,
        float* output_acts
    ) {
        this->D = D;

        C = N * D * N;

        this->weights = weights;
        this->dendrite_acts = dendrite_acts;
        this->output_acts = output_acts;
    }

    // number of segments
    int segments() const {
        return S;
    }

    // segment length
    int length() const {
        return L;
    }

    int dendrites() const {
        return D;
    }

    int vsize() const {
        return N;
    }

    // total size
    int size() const {
        return C;
    }
    
    Vec<S, L> predict(
        const Vec<S, L> &src,
        const typename Layer<S, L>::Params &params
    ) const {
        assert(weights != nullptr);
        assert(dendrite_acts != nullptr);
        assert(output_acts != nullptr);

        Vec<S, L> result;

        const float rescale = params.scale * sqrtf(1.0f / S) / 127.0f;

        // activate
        for (int hs = 0; hs < S; hs++) {
            int max_index = 0;
            float max_activation = limit_min;

            for (int hl = 0; hl < L; hl++) {
                int hindex = hl + L * hs;

                float activation = 0.0f;

                for (int hd = 0; hd < D; hd++) {
                    int dindex = hd + D * hindex;

                    int sum = 0;

                    for (int vs = 0; vs < S; vs++)
                        sum += weights[src[vs] + L * (vs + S * dindex)];

                    float stim = (sum - S * 127) * rescale;

                    float dendrite_act = max(stim * params.leak, stim);

                    activation += dendrite_act * ((hd >= D / 2) * 2.0f - 1.0f);
                }

                if (activation > max_activation) {
                    max_activation = activation;
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
        const typename Layer<S, L>::Params &params
    ) {
        assert(weights != nullptr);
        assert(dendrite_acts != nullptr);
        assert(output_acts != nullptr);

        const float rescale = params.scale * sqrtf(1.0f / S) / 127.0f;

        // activate output
        for (int hs = 0; hs < S; hs++) {
            float max_activation = limit_min;

            for (int hl = 0; hl < L; hl++) {
                int hindex = hl + L * hs;

                float activation = 0.0f;

                for (int hd = 0; hd < D; hd++) {
                    int dindex = hd + D * hindex;

                    int sum = 0;

                    for (int vs = 0; vs < S; vs++)
                        sum += weights[src[vs] + L * (vs + S * dindex)];

                    float stim = (sum - S * 127) * rescale;

                    float dendrite_act = max(stim * params.leak, stim);

                    dendrite_acts[dindex] = dendrite_act;

                    activation += dendrite_act * ((hd >= D / 2) * 2.0f - 1.0f);
                }

                output_acts[hindex] = activation;

                max_activation = max(max_activation, activation);
            }

            // softmax
            float total = 0.0f;

            for (int hl = 0; hl < L; hl++) {
                int hindex = hl + L * hs;

                output_acts[hindex] = expf(output_acts[hindex] - max_activation);

                total += output_acts[hindex];
            }

            float total_inv = 1.0f / max(limit_small, total);

            for (int hl = 0; hl < L; hl++) {
                int hindex = hl + L * hs;

                output_acts[hindex] *= total_inv;

                // learn on target
                float error = (hl == target[hs]) - output_acts[hindex];

                for (int hd = 0; hd < D; hd++) {
                    int dindex = hd + D * hindex;

                    int delta = roundf(params.lr * 255.0f * error * ((hd >= D / 2) * 2.0f - 1.0f) * ((dendrite_acts[dindex] > 0.0f) * (1.0f - params.leak) + params.leak));

                    for (int vs = 0; vs < S; vs++) {
                        int wi = src[vs] + L * (vs + S * dindex);

                        weights[wi] = min(255, max(0, weights[wi] + delta));
                    }
                }
            }
        }
    }
};
}
