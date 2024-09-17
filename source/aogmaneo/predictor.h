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
    Byte* data;

    S_Byte* weights_ih; // input to hidden
    S_Byte* weights_ho; // hidden to output
    float* hidden_acts;
    float* output_acts;
    int* hidden_deltas;
    int* output_deltas;

    int num_hidden;

public:
    static const int N = S * L;

    Predictor()
    :
    data(nullptr)
    {}

    Predictor(
        int num_hidden,
        Byte* data
    ) {
        set_from(num_hidden, data);
    }

    void set_from(
        int num_hidden,
        Byte* data
    ) {
        this->num_input_vecs;
        this->num_hidden = num_hidden;
        this->data = data;

        int num_ih = 2 * S * L * num_hidden;
        int num_ho = num_hidden * S * L;

        int offset = 0;

        this->weights_ih = &data[offset];

        offset += num_ih * sizeof(S_Byte);

        this->weights_oh = &data[offset];

        offset += num_ho * sizeof(S_Byte);

        this->hidden_acts = &data[offset];

        offset += num_hidden * sizeof(float);

        this->output_acts = &data[offset];

        offset += S * L * sizeof(float);

        this->hidden_deltas = &data[offset];

        offset += num_hidden * sizeof(int);

        this->output_deltas = &data[offset];
    }

    void init_random() {
        for (int i = 0; i < num_hidden; i++)
            hidden_acts[i] = 0.0f;

        for (int i = 0; i < N; i++)
            output_acts[i] = 0.0f;

        int num_ih = 2 * S * L * num_hidden;
        int num_ho = num_hidden * S * L;

        for (int i = 0; i < num_ih; i++)
            weights_ih[i] = (rand() % init_weight_noisei) - init_weight_noisei / 2;

        for (int i = 0; i < num_ho; i++)
            weights_ho[i] = (rand() % init_weight_noisei) - init_weight_noisei / 2;
    }

    // number of segments
    int segments() const {
        return S;
    }

    // segment length
    int length() const {
        return L;
    }

    int get_num_hidden() const {
        return num_hidden;
    }

    int vsize() const {
        return N;
    }

    static int weights_size(
        int num_hidden
    ) {
        int num_ih = 2 * S * L * num_hidden;
        int num_ho = num_hidden * S * L;

        return (num_ih + num_ho) * sizeof(S_Byte);
    }

    static int data_size(
        int num_hidden
    ) {
        int num_ih = 2 * S * L * num_hidden;
        int num_ho = num_hidden * S * L;

        return (num_ih + num_ho) * sizeof(S_Byte) + (num_hidden + S * L) * (sizeof(float) + sizeof(int));
    }
    
    Vec<S, L> predict(
        const Vec<S, L> &src1,
        const Vec<S, L> &src2,
        const typename Layer<S, L>::Params &params
    ) const {
        assert(data != nullptr);

        for (int i = 0; i < num_hidden; i++)
            hidden_acts[i] = 0.0f;

        for (int i = 0; i < N; i++)
            output_acts[i] = 0.0f;

        for (int vs = 0; vs < S; vs++) {
            int sindex1 = src1[vs] + L * vs;
            int sindex2 = src2[vs] + L * (vs + S);

            for (int hi = 0; hi < num_hidden; hi++) {
                hidden_acts[hi] += weights_ih[hi + num_hidden * sindex1];
                hidden_acts[hi] += weights_ih[hi + num_hidden * sindex1];
            }
        }

        const float rescale_hidden = params.scale * sqrtf(1.0f / (2 * S)) / 127.0f;

        for (int hi = 0; hi < num_hidden; hi++) {
            hidden_acts[hi] = max(0.0f, hidden_acts[hi] * rescale_hidden); // ReLU

            // sum for output
            for (int oi = 0; oi < N; oi++)
                output_acts[oi] += weights_ho[oi + N * hi] * hidden_acts[hi];
        }

        const float rescale_output = params.scale * sqrtf(1.0f / num_hidden) / 127.0f;

        Vec<S, L> result;

        for (int os = 0; os < S; os++) {
            int max_index = 0;
            float max_activation = limit_min;

            for (int ol = 0; ol < L; ol++) {
                int oi = ol + L * os;

                output_acts[oi] *= rescale_output;
                
                if (output_acts[oi] > max_activation) {
                    max_activation = output_acts[oi];
                    max_index = ol;
                }
            }

            result[os] = max_index;
        }

        return result;
    }

    // reqires predict to have been called first
    void learn(
        const Vec<S, L> &src1,
        const Vec<S, L> &src2,
        const Vec<S, L> &pred,
        const Vec<S, L> &target,
        unsigned long* state,
        const typename Layer<S, L>::Params &params
    ) {
        assert(data != nullptr);

        const float rescale_error = params.scale * sqrtf(1.0f / N) / 127.0f;

        // backward
        for (int hi = 0; hi < num_hidden; hi++) {
            if (hidden_acts[hi] <= 0.0f) // ReLU gradient, can skip entirely if in linear portion
                continue;

            float error = 0.0f;

            for (int os = 0; os < S; os++) {
                error -= weights_ho[(pred[os] + L * os) + N * hi];
                error += weights_ho[(target[os] + L * os) + N * hi];
            }

            error *= rescale_error;

            int delta = rand_roundf(params.lr * 127.0f * error);

            for (int vs = 0; vs < S; vs++) {
                int sindex1 = src1[vs] + L * vs;
                int sindex2 = src2[vs] + L * (vs + S);

                int wi1 = hi + num_hidden * sindex1;
                int wi2 = hi + num_hidden * sindex2;

                weights_ih[wi1] = min(127, max(-127, weights_ih[wi1] + delta)); 
                weights_ih[wi2] = min(127, max(-127, weights_ih[wi2] + delta)); 
            }
        }

        // update output weights
        for (int os = 0; os < S; os++) {
            int delta_target = rand_roundf(params.lr * 127.0f);
            int delta_pred = -delta_target;

            for (int hi = 0; hi < num_hidden; hi++) {
                int wi_target = (os + L * target[os]) + N * hi;
                int wi_pred = (os + L * pred[os]) + N * hi;

                weights_ho[wi_target] = min(127, weights_ho[wi_target] + delta_target);
                weights_ho[wi_pred] = max(-127, weights_ho[wi_pred] + delta_pred);
            }
        }
    }
};
}
