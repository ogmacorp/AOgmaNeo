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
    float choice;
    float vigilance;

    Layer_Params()
    :
    choice(0.0001f),
    vigilance(0.9f)
    {}
};

// take 2 vectors and map to 1
template<int S, int L>
class Predictor {
private:
    int hidden_segments;
    int hidden_length;

    Byte_Buffer weights_encode;
    Byte_Buffer weights_decode;
    Int_Buffer totals;

    Vec<S, L> hiddens;
    Vec<S, L> preds;

    int max_global_index;

    Int_Buffer sums;

public:
    static const int N = S * L;

    Predictor()
    {}

    Predictor(
        int hidden_segments,
        int hidden_length
    ) {
        init_random(hidden_segments, hidden_length);
    }

    void init_random(
        int hidden_segments,
        int hidden_length
    ) {
        this->hidden_segments = hidden_segments;
        this->hidden_length = hidden_length;

        int num_hidden = hidden_segments * hidden_length;
        int num_weights = num_hidden * N;

        weights_encode.resize(num_weights);
        weights_decode.resize(weights_encode.size());

        for (int i = 0; i < weights_encode.size(); i++) {
            weights_encode[i] = (rand() % init_weight_noisei);
            weights_decode[i] = 0;
        }

        totals = Int_Buffer(num_hidden);

        for (int hi = 0; hi < num_hidden; hi++) {
            int total = 0;

            for (int vi = 0; vi < N; vi++) {
                int wi = hi + num_hidden * vi;

                total += weights_encode[wi];
            }

            totals[hi] = total;
        }

        hiddens = 0;
        preds = 0;

        max_global_index = -1;

        sums.resize(num_hidden);
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
    
    int get_hidden_segments() const {
        return hidden_segments;
    }

    int get_hidden_length() const {
        return hidden_length;
    }

    Vec<S, L> step(
        const Vec<S, L> &inputs,
        const Vec<S, L> &targets,
        bool learn_enabled,
        const Layer_Params &params
    ) {
        int num_hidden = hidden_segments * hidden_length;

        // decoder learn
        if (learn_enabled && max_global_index != -1) { // check max_global_index to see that ran at least once
            int hi_max_global = hiddens[max_global_index] + hidden_length * max_global_index;

            for (int vs = 0; vs < S; vs++) {
                if (preds[vs] == targets[vs])
                    continue;

                int tindex = targets[vs] + L * vs;

                int wi = hi_max_global + num_hidden * tindex;

                weights_decode[wi] = min(255, weights_decode[wi] + 1);
            }
        }

        for (int i = 0; i < num_hidden; i++)
            sums[i] = 0;

        for (int vs = 0; vs < S; vs++) {
            int iindex = inputs[vs] + L * vs;

            for (int hs = 0; hs < hidden_segments; hs++) {
                for (int hl = 0; hl < hidden_length; hl++) {
                    int hi = hl + hidden_length * hs;

                    int wi = hi + num_hidden * iindex;

                    sums[hi] += weights_encode[wi];
                }
            }
        }

        hiddens = 0;

        max_global_index = 0;
        float max_global_activation = 0.0f;
        float max_global_match = 0.0f;

        const float byte_inv = 1.0f / 255.0f;

        for (int hs = 0; hs < hidden_segments; hs++) {
            int max_index = -1;
            float max_activation = 0.0f;

            int max_complete_index = 0;
            float max_complete_activation = 0.0f;

            for (int hl = 0; hl < hidden_length; hl++) {
                int hi = hl + hidden_length * hs;

                float sum = sums[hi] * byte_inv;
                float total = totals[hi] * byte_inv;

                float complemented = (N - total) - (S - sum);

                float match = complemented / (S * (L - 1));

                float activation = complemented / (params.choice + N - total);

                if (activation > max_activation && match >= params.vigilance) {
                    max_activation = activation;
                    max_index = hl;
                }

                if (activation > max_complete_activation) {
                    max_complete_activation = activation;
                    max_complete_index = hl;
                }
            }

            hiddens[hs] = (max_index == -1 ? max_complete_index : max_index);

            float global_activation = (max_index == -1 ? 0.0f : max_complete_activation);

            if (global_activation > max_global_activation) {
                max_global_activation = global_activation;
                max_global_index = hs;
            }
        }

        // next prediction
        for (int vs = 0; vs < S; vs++) {
            int max_index = 0;
            int max_activation = 0;

            for (int vl = 0; vl < L; vl++) {
                int vi = vl + L * vs;

                int sum = 0;

                for (int hs = 0; hs < hidden_segments; hs++) {
                    int hi = hiddens[hs] + hidden_length * hs;

                    int wi = hi + num_hidden * vi;

                    sum += weights_decode[wi];
                }

                if (sum > max_activation) {
                    max_activation = sum;
                    max_index = vl;
                }
            }

            preds[vs] = max_index;
        }

        // learn encoder
        if (learn_enabled) {
            int hi_max_global = hiddens[max_global_index] + hidden_length * max_global_index;

            for (int vs = 0; vs < S; vs++) {
                int iindex = inputs[vs] + L * vs;

                int wi = hi_max_global + num_hidden * iindex;

                Byte w_old = weights_encode[wi];

                weights_encode[wi] = 255;

                totals[hi_max_global] += weights_encode[wi] - w_old;
            }
        }

        return preds;
    }

    // serialization
    long size() const { // returns size in Bytes
        return 2 * sizeof(int) + 2 * weights_encode.size() * sizeof(Byte) + totals.size() * sizeof(int) + 2 * sizeof(Vec<S, L>) + sizeof(int);
    }

    long weights_size() const { // returns size of weights in Bytes
        return weights_encode.size() * sizeof(Byte);
    }

    void write(
        Stream_Writer &writer
    ) const {
        writer.write(&hidden_segments, sizeof(int));
        writer.write(&hidden_length, sizeof(int));

        writer.write(&weights_encode[0], weights_encode.size() * sizeof(Byte));
        writer.write(&weights_decode[0], weights_decode.size() * sizeof(Byte));
        writer.write(&totals[0], totals.size() * sizeof(int));

        writer.write(&hiddens, sizeof(Vec<S, L>));
        writer.write(&preds, sizeof(Vec<S, L>));

        writer.write(&max_global_index, sizeof(int));
    }

    void read(
        Stream_Reader &reader
    ) {
        reader.read(&hidden_segments, sizeof(int));
        reader.read(&hidden_length, sizeof(int));

        int num_hidden = hidden_segments * hidden_length;
        int num_weights = num_hidden * N * 2;

        weights_encode.resize(num_weights);
        weights_decode.resize(num_weights);
        totals.resize(num_hidden);

        reader.read(&weights_encode[0], weights_encode.size() * sizeof(Byte));
        reader.read(&weights_decode[0], weights_decode.size() * sizeof(Byte));
        reader.read(&totals[0], totals.size() * sizeof(int));

        reader.read(&hiddens, sizeof(Vec<S, L>));
        reader.read(&preds, sizeof(Vec<S, L>));

        reader.read(&max_global_index, sizeof(int));

        sums.resize(num_hidden);
    }

    const Byte_Buffer &get_weights_encode() const {
        return weights_encode;
    }

    const Byte_Buffer &get_weights_decode() const {
        return weights_decode;
    }

    const Vec<S, L> &get_hiddens() const {
        return hiddens;
    }
};
}
