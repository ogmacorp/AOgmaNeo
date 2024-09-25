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
    int radius;
    Byte* data;

    Byte* weights;
    int* output_acts;

public:
    static const int N = S * L;

    Predictor()
    :
    data(nullptr)
    {}

    Predictor(
        int radius,
        Byte* data
    ) {
        set_from(radius, data);
    }

    void set_from(
        int radius,
        Byte* data
    ) {
        this->radius = radius;
        this->data = data;

        const int diam = radius * 2 + 1;
        const int num_weights = N * diam * L;

        this->weights = reinterpret_cast<Byte*>(data);
        this->output_acts = reinterpret_cast<int*>(data + num_weights * sizeof(Byte));
    }

    void init_random() {
        assert(data != nullptr);

        for (int i = 0; i < N; i++)
            output_acts[i] = 0;

        const int diam = radius * 2 + 1;
        const int num_weights = N * diam * L;

        for (int i = 0; i < num_weights; i++)
            weights[i] = 127 + (rand() % (init_weight_noisei + 1)) - init_weight_noisei / 2;
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

    static int data_size(
        int radius
    ) {
        const int diam = radius * 2 + 1;
        const int num_weights = N * diam * L;

        return num_weights * sizeof(Byte) + N * sizeof(int);
    }

    int get_radius() const {
        return radius;
    }
    
    Vec<S, L> predict(
        const Vec<S, L> &src,
        const Layer_Params &params
    ) const {
        assert(data != nullptr);

        const int diam = radius * 2 + 1;

        for (int i = 0; i < N; i++)
            output_acts[i] = 0;

        for (int os = 0; os < S; os++) {
            for (int dvs = -radius; dvs <= radius; dvs++) {
                int vs = os + dvs;

                // wrap
                if (vs < 0)
                    vs += S;
                else if (vs >= S)
                    vs -= S;

                int sindex = src[vs] + L * vs;

                for (int ol = 0; ol < L; ol++) {
                    int oi = ol + L * os;

                    output_acts[oi] += weights[ol + L * ((dvs + radius) + diam * sindex)];
                }
            }
        }

        Vec<S, L> result;

        for (int os = 0; os < S; os++) {
            int max_index = 0;
            int max_activation = limit_min;

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
        const Layer_Params &params
    ) {
        assert(data != nullptr);

        const int delta = ceilf(params.lr * 255.0f);

        const int diam = radius * 2 + 1;

        for (int os = 0; os < S; os++) {
            if (target[os] == pred[os])
                continue;

            for (int dvs = -radius; dvs <= radius; dvs++) {
                int vs = os + dvs;

                // wrap
                if (vs < 0)
                    vs += S;
                else if (vs >= S)
                    vs -= S;

                int sindex = src[vs] + L * vs;

                {
                    int ol = target[os];

                    int wi = ol + L * ((dvs + radius) + diam * sindex);

                    weights[wi] = min(255, weights[wi] + delta);
                }

                {
                    int ol = pred[os];

                    int wi = ol + L * ((dvs + radius) + diam * sindex);

                    weights[wi] = max(0, weights[wi] - delta);
                }
            }
        }
    }
};
}
