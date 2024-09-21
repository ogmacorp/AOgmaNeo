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
struct Layer_Params {
    float lr;

    Layer_Params()
    :
    lr(0.01f)
    {}
};


// take 2 vectors and map to 1
template<int S, int L>
class Predictor {
private:
    Byte* data;

    S_Byte* weights;
    int* output_acts;

public:
    static const int N = S * L;

    Predictor()
    :
    data(nullptr)
    {}

    Predictor(
        Byte* data
    ) {
        set_from(data);
    }

    void set_from(
        Byte* data
    ) {
        this->data = data;

        int num_weights = N * N;

        int offset = 0;

        this->weights = reinterpret_cast<S_Byte*>(data + offset);

        offset += num_weights * sizeof(S_Byte);

        this->output_acts = reinterpret_cast<int*>(data + offset);
    }

    void init_random() {
        assert(data != nullptr);

        for (int i = 0; i < N; i++)
            output_acts[i] = 0.;

        int num_weights = N * N;

        for (int i = 0; i < num_weights; i++)
            weights[i] = (rand() % (init_weight_noisei + 1)) - init_weight_noisei / 2;
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

    static int weights_size() {
        return N * N;
    }

    static int data_size() {
        return N * N * sizeof(S_Byte) + N * sizeof(int);
    }
    
    Vec<S, L> predict(
        const Vec<S, L> &src,
        const Layer_Params &params
    ) const {
        assert(data != nullptr);

        for (int i = 0; i < N; i++)
            output_acts[i] = 0;

        for (int vs = 0; vs < S; vs++) {
            int sindex = src[vs] + L * vs;

            for (int oi = 0; oi < N; oi++)
                output_acts[oi] += weights[oi + N * sindex];
        }

        Vec<S, L> result;

        for (int os = 0; os < S; os++) {
            int max_index = 0;
            int max_activation = 0;

            for (int ol = 0; ol < L; ol++) {
                int oi = ol + L * os;

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
        const Vec<S, L> &src,
        const Vec<S, L> &pred,
        const Vec<S, L> &target,
        unsigned long* state,
        const Layer_Params &params
    ) {
        assert(data != nullptr);

        // update output weights
        for (int vs = 0; vs < S; vs++) {
            int sindex = src[vs] + L * vs;

            const int delta = rand_roundf(params.lr * 127.0f, state);

            for (int os = 0; os < S; os++) {
                if (target[os] == pred[os])
                    continue;

                int wi_target = (target[os] + L * os) + N * sindex;
                int wi_pred = (pred[os] + L * os) + N * sindex;

                weights[wi_target] = min(127, max(-127, weights[wi_target] + delta));
                weights[wi_pred] = min(127, max(-127, weights[wi_pred] - delta));
            }
        }
    }
};
}
