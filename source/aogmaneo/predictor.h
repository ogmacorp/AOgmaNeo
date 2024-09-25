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
#include <iostream>

namespace aon {
struct Layer_Params {
    float choice;
    float vigilance_low;
    float vigilance_high;

    Layer_Params()
    :
    choice(0.0001f),
    vigilance_low(0.97f),
    vigilance_high(0.97f)
    {}
};

// take 2 vectors and map to 1
template<int S, int L>
class Predictor {
private:
    int hidden_segments;
    int hidden_length;

    Byte_Buffer weights;
    Int_Buffer hidden_max_indices;
    Int_Buffer totals_src;
    Int_Buffer totals_pred;
    Int_Buffer commits;
    int global_commits;

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
        int num_weights = num_hidden * N * 2;

        weights = Byte_Buffer((num_weights + 7) / 8, 0);
        hidden_max_indices = Int_Buffer(hidden_segments, -1);
        totals_src = Int_Buffer(num_hidden, 0);
        totals_pred = Int_Buffer(num_hidden, 0);
        commits = Int_Buffer(hidden_segments, 0);

        global_commits = 0;

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
        const Vec<S, L> &src,
        const Layer_Params &params
    ) {
        int num_hidden = hidden_segments * hidden_length;

        for (int i = 0; i < num_hidden; i++)
            sums[i] = 0;

        for (int vs = 0; vs < S; vs++) {
            int sindex = src[vs] + L * vs;

            for (int hs = 0; hs < global_commits; hs++) {
                for (int hl = 0; hl < commits[hs]; hl++) {
                    int hi = hl + hidden_length * hs;

                    int swi = hi + num_hidden * sindex;

                    int sbyi = swi / 8;
                    int sbi = swi % 8;

                    sums[hi] += ((weights[sbyi] & (1 << sbi)) != 0);
                }
            }
        }

        hidden = 0;

        for (int hs = 0; hs < global_commits; hs++) {
            int max_index = -1;
            float max_activation = 0.0f;

            int max_complete_index = 0;
            float max_complete_activation = 0.0f;

            for (int hl = 0; hl < commits[hs]; hl++) {
                int hi = hl + hidden_length * hs;

                float complemented = (N - totals_src[hi]) - (S - sums[hi]);

                float match = complemented / (S * (L - 1));

                float activation = complemented / (params.choice + N - totals_src[hi]);

                if (activation > max_activation && match >= params.vigilance_high) {
                    max_activation = activation;
                    max_index = hl;
                }

                if (activation > max_complete_activation) {
                    max_complete_activation = activation;
                    max_complete_index = hl;
                }
            }

            hidden[hs] = (max_index == -1 ? max_complete_index : max_index);
        }

        // reconstruct
        Vec<S, L> result;

        for (int vs = 0; vs < S; vs++) {
            int sindex = src[vs] + L * vs;

            int max_index = 0;
            int max_activation = 0;

            for (int vl = 0; vl < L; vl++) {
                int tvi = vl + L * vs + N; // + N to shift to prediction weights

                int sum = 0;

                for (int hs = 0; hs < global_commits; hs++) {
                    int hi = hidden[hs] + hidden_length * hs;

                    int twi = hi + num_hidden * tvi;

                    int tbyi = twi / 8;
                    int tbi = twi % 8;

                    sum += ((weights[tbyi] & (1 << tbi)) != 0);
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
        const Vec<S, L> &src,
        const Vec<S, L> &target,
        const Layer_Params &params
    ) {
        int num_hidden = hidden_segments * hidden_length;

        for (int i = 0; i < num_hidden; i++)
            sums[i] = 0;

        for (int vs = 0; vs < S; vs++) {
            int sindex = src[vs] + L * vs;
            int tindex = target[vs] + L * vs + N;

            for (int hs = 0; hs < global_commits; hs++) {
                for (int hl = 0; hl < commits[hs]; hl++) {
                    int hi = hl + hidden_length * hs;

                    int swi = hi + num_hidden * sindex;
                    int twi = hi + num_hidden * tindex;

                    int sbyi = swi / 8;
                    int sbi = swi % 8;

                    int tbyi = twi / 8;
                    int tbi = twi % 8;

                    sums[hi] += ((weights[sbyi] & (1 << sbi)) != 0);
                    sums[hi] += ((weights[tbyi] & (1 << tbi)) != 0);
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

            int max_complete_index = 0;
            float max_complete_activation = 0.0f;
            float max_complete_match = 0.0f;

            for (int hl = 0; hl < commits[hs]; hl++) {
                int hi = hl + hidden_length * hs;

                float complemented = (N * 2 - (totals_src[hi] + totals_pred[hi])) - (S * 2 - sums[hi]);

                float match = complemented / (S * 2 * (L - 1));

                float activation = complemented / (params.choice + N * 2 - (totals_src[hi] + totals_pred[hi]));

                if (activation > max_activation && match >= params.vigilance_high) {
                    max_activation = activation;
                    max_index = hl;
                }

                if (activation > max_complete_activation) {
                    max_complete_activation = activation;
                    max_complete_index = hl;
                }

                max_complete_match = max(max_complete_match, match);
            }

            hidden_max_indices[hs] = max_index;
            hidden[hs] = (max_index == -1 ? max_complete_index : max_index);

            float global_activation = max_complete_activation;

            if (global_activation > max_global_activation) {
                max_global_activation = global_activation;
                max_global_match = max_complete_match;
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

        for (int vs = 0; vs < S; vs++) {
            int sindex = src[vs] + L * vs;
            int tindex = target[vs] + L * vs + N;

            int hi = hidden[max_global_index] + hidden_length * max_global_index;

            int swi = hi + num_hidden * sindex;
            int twi = hi + num_hidden * tindex;

            int sbyi = swi / 8;
            int sbi = swi % 8;

            int tbyi = twi / 8;
            int tbi = twi % 8;

            bool sw_old = ((weights[sbyi] & (1 << sbi)) != 0);
            bool tw_old = ((weights[tbyi] & (1 << tbi)) != 0);

            if (!sw_old) {
                weights[sbyi] |= (1 << sbi);

                totals_src[hi]++;
            }

            if (!tw_old) {
                weights[tbyi] |= (1 << tbi);

                totals_pred[hi]++;
            }
        }
    }

    // serialization
    long size() const { // returns size in Bytes
        return 2 * sizeof(int) + weights.size() * sizeof(Byte) + hidden_max_indices.size() * sizeof(int) + 2 * totals_src.size() * sizeof(int) + commits.size() * sizeof(int) + sizeof(int);
    }

    long weights_size() const { // returns size of weights in Bytes
        return weights.size() * sizeof(Byte);
    }

    void write(
        Stream_Writer &writer
    ) const {
        writer.write(&hidden_segments, sizeof(int));
        writer.write(&hidden_length, sizeof(int));

        writer.write(&weights[0], weights.size() * sizeof(Byte));
        writer.write(&hidden_max_indices[0], hidden_max_indices.size() * sizeof(int));
        writer.write(&totals_src[0], totals_src.size() * sizeof(int));
        writer.write(&totals_pred[0], totals_pred.size() * sizeof(int));
        writer.write(&commits[0], commits.size() * sizeof(int));

        writer.write(&global_commits, sizeof(int));
    }

    void read(
        Stream_Reader &reader
    ) {
        reader.read(&hidden_segments, sizeof(int));
        reader.read(&hidden_length, sizeof(int));

        int num_hidden = hidden_segments * hidden_length;
        int num_weights = num_hidden * N * 2;

        weights.resize((num_weights + 7) / 8);
        hidden_max_indices.resize(hidden_segments);
        totals_src.resize(num_hidden);
        totals_pred.resize(num_hidden);
        commits.resize(num_hidden);

        reader.read(&weights[0], weights.size() * sizeof(Byte));
        reader.read(&hidden_max_indices[0], hidden_max_indices.size() * sizeof(int));
        reader.read(&totals_src[0], totals_src.size() * sizeof(int));
        reader.read(&totals_pred[0], totals_pred.size() * sizeof(int));
        reader.read(&commits[0], commits.size() * sizeof(int));

        reader.read(&global_commits, sizeof(int));

        sums.resize(num_hidden);
    }

    const Byte_Buffer &get_weights() const {
        return weights;
    }

    const Vec<S, L> &get_hidden() const {
        return hidden;
    }
};
}
