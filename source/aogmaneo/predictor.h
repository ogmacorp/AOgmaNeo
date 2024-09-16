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

// take 2 vectors and map to 1
template<int S, int L>
class Predictor {
private:
    Byte* weights;
    int* dendrite_sums;
    float* dendrite_acts;
    float* output_acts;
    int* dendrite_deltas;

    int D;
    int C;

public:
    static const int N = S * L;

    Predictor()
    :
    weights(nullptr),
    dendrite_acts(nullptr),
    output_acts(nullptr),
    dendrite_deltas(nullptr)
    {}

    Predictor(
        int D,
        Byte* weights,
        int* dendrite_sums,
        float* dendrite_acts,
        float* output_acts,
        int* dendrite_deltas
    ) {
        set_from(D, weights, dendrite_sums, dendrite_acts, output_acts, dendrite_deltas);
    }

    void set_from(
        int D,
        Byte* weights,
        int* dendrite_sums,
        float* dendrite_acts,
        float* output_acts,
        int* dendrite_deltas
    ) {
        this->D = D;

        C = 2 * N * D * N;

        this->weights = weights;
        this->dendrite_sums = dendrite_sums;
        this->dendrite_acts = dendrite_acts;
        this->output_acts = output_acts;
        this->dendrite_deltas = dendrite_deltas;
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
        const Vec<S, L> &src1,
        const Vec<S, L> &src2,
        const typename Layer<S, L>::Params &params
    ) const {
        assert(weights != nullptr);

        const int total_num_dendrites = N * D;

        for (int i = 0; i < total_num_dendrites; i++)
            dendrite_sums[i] = 0;

        for (int vs = 0; vs < S; vs++) {
            int sindex1 = src1[vs] + L * vs;
            int sindex2 = N + src2[vs] + L * vs;

            for (int hs = 0; hs < S; hs++) {
                for (int hl = 0; hl < L; hl++) {
                    int hindex = hl + L * hs;

                    for (int hd = 0; hd < D; hd++) {
                        int dindex = hd + D * hindex;

                        dendrite_sums[dindex] += weights[dindex + total_num_dendrites * sindex1];
                        dendrite_sums[dindex] += weights[dindex + total_num_dendrites * sindex2];
                    }
                }
            }
        }

        const float rescale_stim = params.scale * sqrtf(1.0f / S) / 127.0f;
        const float rescale_act = sqrtf(1.0f / D);

        Vec<S, L> result;

        for (int hs = 0; hs < S; hs++) {
            int max_index = 0;
            float max_activation = limit_min;

            for (int hl = 0; hl < L; hl++) {
                int hindex = hl + L * hs;

                float activation = 0.0f;

                for (int hd = 0; hd < D; hd++) {
                    int dindex = hd + D * hindex;

                    float stim = (dendrite_sums[dindex] - S * 127.0f) * rescale_stim;

                    dendrite_acts[dindex] = max(stim * params.leak, stim);

                    activation += dendrite_acts[dindex] * ((hd >= D / 2) * 2.0f - 1.0f);
                }

                activation * rescale_act;

                output_acts[hindex] = activation;

                if (activation > max_activation) {
                    max_activation = activation;
                    max_index = hl;
                }
            }

            result[hs] = max_index;
        }

        return result;
    }

    // reqires predict to have been called first
    void learn(
        const Vec<S, L> &src1,
        const Vec<S, L> &src2,
        const Vec<S, L> &target,
        unsigned long* state,
        const typename Layer<S, L>::Params &params
    ) {
        assert(weights != nullptr);

        const float rescale_stim = params.scale * sqrtf(1.0f / S) / 127.0f;
        const float rescale_act = sqrtf(1.0f / D);

        // learn from existing state (predict was called with the same src)
        for (int hs = 0; hs < S; hs++) {
            float max_activation = limit_min;

            for (int hl = 0; hl < L; hl++) {
                int hindex = hl + L * hs;

                max_activation = max(max_activation, output_acts[hindex]);
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

                    int delta = rand_roundf(params.lr * 255.0f * error * ((hd >= D / 2) * 2.0f - 1.0f) * ((dendrite_acts[dindex] > 0.0f) * (1.0f - params.leak) + params.leak), state);

                    dendrite_deltas[dindex] = delta;
                }
            }
        }

        const int total_num_dendrites = N * D;

        for (int vs = 0; vs < S; vs++) {
            int sindex1 = src1[vs] + L * vs;
            int sindex2 = N + src2[vs] + L * vs;

            for (int hs = 0; hs < S; hs++) {
                for (int hl = 0; hl < L; hl++) {
                    int hindex = hl + L * hs;

                    for (int hd = 0; hd < D; hd++) {
                        int dindex = hd + D * hindex;

                        int wi1 = dindex + total_num_dendrites * sindex1;
                        int wi2 = dindex + total_num_dendrites * sindex2;

                        weights[wi1] = min(255, max(0, weights[wi1] + dendrite_deltas[dindex]));
                        weights[wi2] = min(255, max(0, weights[wi2] + dendrite_deltas[dindex]));
                    }
                }
            }
        }
    }
};
}
