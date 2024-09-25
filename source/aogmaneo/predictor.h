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
    float vigilance_low;
    float vigilance_high;

    Layer_Params()
    :
    choice(0.0001f),
    vigilance_low(0.9f),
    vigilance_high(0.95f)
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
    Int_Buffer hidden_max_indices;
    Int_Buffer totals;
    Int_Buffer commits;
    int global_commits;

    int max_global_index;

    Int_Buffer sums;

    Vec<S, L> hidden;

public:
    static const int N = S * L;

    Predictor()
    {}

    Predictor(
        int hidden_segments,
        int hidden_length
    ) {
        init(hidden_segments, hidden_length);
    }

    void init(
        int hidden_segments,
        int hidden_length
    ) {
        this->hidden_segments = hidden_segments;
        this->hidden_length = hidden_length;

        int num_hidden = hidden_segments * hidden_length;
        int num_weights = num_hidden * N;

        weights_encode = Byte_Buffer((num_weights + 7) / 8, 0);
        weights_decode = Byte_Buffer((num_weights + 7) / 8, 0);
        hidden_max_indices = Int_Buffer(hidden_segments, -1);
        totals = Int_Buffer(num_hidden, 0);
        commits = Int_Buffer(hidden_segments, 0);

        global_commits = 0;
        max_global_index = 0;

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

    Vec<S, L> predict(
        const Vec<S, L> &inputs,
        const Layer_Params &params
    ) {
        int num_hidden = hidden_segments * hidden_length;

        for (int i = 0; i < num_hidden; i++)
            sums[i] = 0;

        for (int vs = 0; vs < S; vs++) {
            int iindex = inputs[vs] + L * vs;

            for (int hs = 0; hs < global_commits; hs++) {
                for (int hl = 0; hl < commits[hs]; hl++) {
                    int hi = hl + hidden_length * hs;

                    int wi = hi + num_hidden * iindex;

                    int byi = wi / 8;
                    int bi = wi % 8;

                    sums[hi] += ((weights_encode[byi] & (1 << bi)) != 0);
                }
            }
        }

        hidden = 0;

        int max_global_index = 0;
        float max_global_activation = 0.0f;
        float max_global_match = 0.0f;

        for (int hs = 0; hs < global_commits; hs++) {
            int max_index = -1;
            float max_activation = 0.0f;
            float max_match = 0.0f;

            int max_complete_index = 0;
            float max_complete_activation = 0.0f;

            for (int hl = 0; hl < commits[hs]; hl++) {
                int hi = hl + hidden_length * hs;

                float complemented = (N - totals[hi]) - (S - sums[hi]);

                float match = complemented / (S * (L - 1));

                float activation = complemented / (params.choice + N - totals[hi]);

                if (activation > max_activation && match >= params.vigilance_high) {
                    max_activation = activation;
                    max_match = match;
                    max_index = hl;
                }

                if (activation > max_complete_activation) {
                    max_complete_activation = activation;
                    max_complete_index = hl;
                }
            }

            hidden_max_indices[hs] = max_index;
            hidden[hs] = (max_index == -1 ? max_complete_index : max_index);

            if (max_complete_activation > max_global_activation) {
                max_global_activation = max_complete_activation;
                max_global_match = max_match;
                max_global_index = hs;
            }
        }

        if (max_global_match < params.vigilance_low && global_commits < hidden_segments) {
            max_global_index = global_commits;
            global_commits++;
        }

        if (hidden_max_indices[max_global_index] == -1 && commits[max_global_index] < hidden_length) {
            hidden[max_global_index] = commits[max_global_index];
            commits[max_global_index]++;
        }

        // reconstruct
        Vec<S, L> result;

        for (int vs = 0; vs < S; vs++) {
            int max_index = 0;
            int max_activation = 0;

            for (int vl = 0; vl < L; vl++) {
                int vi = vl + L * vs;

                int sum = 0;

                for (int hs = 0; hs < global_commits; hs++) {
                    if (commits[hs] == 0)
                        continue;

                    int hi = hidden[hs] + hidden_length * hs;

                    int wi = hi + num_hidden * vi;

                    int byi = wi / 8;
                    int bi = wi % 8;

                    sum += ((weights_decode[byi] & (1 << bi)) != 0);
                }

                if (sum > max_activation) {
                    max_activation = sum;
                    max_index = vl;
                }
            }

            result[vs] = max_index;
        }

        return result;
    }

    // reqires predict to have been called first
    void learn(
        const Vec<S, L> &inputs,
        const Vec<S, L> &preds,
        const Vec<S, L> &targets,
        const Layer_Params &params
    ) {
        int num_hidden = hidden_segments * hidden_length;

        for (int vs = 0; vs < S; vs++) {
            // encoder
            {
                int iindex = inputs[vs] + L * vs;

                int hi = hidden[max_global_index] + hidden_length * max_global_index;

                int wi = hi + num_hidden * iindex;

                int byi = wi / 8;
                int bi = wi % 8;

                bool w_old = ((weights_encode[byi] & (1 << bi)) != 0);

                if (!w_old) {
                    weights_encode[byi] |= (1 << bi);

                    totals[hi]++;
                }
            }

            // decoder
            if (preds[vs] != targets[vs]) {
                int tindex = targets[vs] + L * vs;

                for (int hs = 0; hs < hidden_segments; hs++) {
                    if (commits[hs] == 0)
                        continue;

                    int hi = hidden[hs] + hidden_length * hs;

                    int wi = hi + num_hidden * tindex;

                    int byi = wi / 8;
                    int bi = wi % 8;

                    weights_decode[byi] |= (1 << bi);
                }
            }
        }
    }

    // serialization
    long size() const { // returns size in Bytes
        return 2 * sizeof(int) + 2 * weights_encode.size() * sizeof(Byte) + hidden_max_indices.size() * sizeof(int) + totals.size() * sizeof(int) + commits.size() * sizeof(int) + 2 * sizeof(int);
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
        writer.write(&hidden_max_indices[0], hidden_max_indices.size() * sizeof(int));
        writer.write(&totals[0], totals.size() * sizeof(int));
        writer.write(&commits[0], commits.size() * sizeof(int));

        writer.write(&global_commits, sizeof(int));
        writer.write(&max_global_index, sizeof(int));
    }

    void read(
        Stream_Reader &reader
    ) {
        reader.read(&hidden_segments, sizeof(int));
        reader.read(&hidden_length, sizeof(int));

        int num_hidden = hidden_segments * hidden_length;
        int num_weights = num_hidden * N * 2;

        weights_encode.resize((num_weights + 7) / 8);
        weights_decode.resize((num_weights + 7) / 8);
        hidden_max_indices.resize(hidden_segments);
        totals.resize(num_hidden);
        commits.resize(num_hidden);

        reader.read(&weights_encode[0], weights_encode.size() * sizeof(Byte));
        reader.read(&weights_decode[0], weights_decode.size() * sizeof(Byte));
        reader.read(&hidden_max_indices[0], hidden_max_indices.size() * sizeof(int));
        reader.read(&totals[0], totals.size() * sizeof(int));
        reader.read(&commits[0], commits.size() * sizeof(int));

        reader.read(&global_commits, sizeof(int));
        reader.read(&max_global_index, sizeof(int));

        sums.resize(num_hidden);
    }

    const Byte_Buffer &get_weights_encode() const {
        return weights_encode;
    }

    const Byte_Buffer &get_weights_decode() const {
        return weights_decode;
    }

    const Vec<S, L> &get_hidden() const {
        return hidden;
    }

    const Int_Buffer &get_commits() const {
        return commits;
    }

    int get_global_commits() const {
        return global_commits;
    }
};
}
