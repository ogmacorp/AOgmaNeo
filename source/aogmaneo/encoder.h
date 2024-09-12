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
template<int S>
class Encoder {
public:
    // helper for implementing XXT in resonator
    class Resonator_Mat {
    private:
        static const int ss = S * (S + 1) / 2; // swap - 1 instead of + 1 for ignoring diagonal

        int buffer[ss];

    public:
        Resonator_Mat()
        {}

        Resonator_Mat(
            const Array<Vec<S>> &codes,
            int codes_start,
            int num_codes
        ) {
            set_from(codes, codes_start, num_codes);
        }

        void set_from(
            const Array<Vec<S>> &codes,
            int codes_start,
            int num_codes
        ) {
            for (int r = 0; r < S; r++) {
                int start = r * (r + 1) / 2;

                for (int c = 0; c <= r; c++) {
                    int index = c + start;

                    int sum = 0;

                    for (int c2 = 0; c2 < num_codes; c2++)
                        sum += codes[codes_start + c2].get(r) * codes[codes_start + c2].get(c);

                    buffer[index] = sum;
                }
            }
        }

        Bundle<S> operator*(
            const Vec<S> &v
        ) {
            Bundle<S> result = 0;

            for (int r = 0; r < S; r++) {
                int start = r * (r + 1) / 2;

                // lower triangle, duplicated into upper triangle as well
                for (int c = 0; c < r; c++) { // ignore diagonal for now
                    int index = c + start;
                    
                    result[c] += buffer[index] * v.get(r); // lower triangle (original)
                    result[r] += buffer[index] * v.get(c); // upper triangle (duplicate)
                }

                // now do diagonal (not duplicated)
                result[r] += buffer[r + start] * v.get(r);
            }

            return result;
        }
    };

    // visible layer descriptor
    struct Visible_Layer_Desc {
        Int4 size; // size of input

        int radius; // radius onto input

        // defaults
        Visible_Layer_Desc()
        :
        size(4, 4, 4, 16),
        radius(2)
        {}
    };

    // visible layer
    struct Visible_Layer {
        Int_Buffer input_cis;

        Array<Vec<S>> visible_code_vecs;

        Array<Vec<S>> visible_pos_vecs; // positional encodings

        Array<Vec<S>> input_vecs;

        Array<Bundle<S>> hidden_sums;

        Int_Buffer recon_cis;

        Array<Vec<S>> recon_factors;
        Array<Vec<S>> recon_factors_temp;

        Array<Resonator_Mat> visible_mats;

        bool use_input;
        bool up_to_date;
        
        Visible_Layer()
        :
        use_input(false),
        up_to_date(false)
        {}
    };

    struct Params {
        float lr; // code learning rate
        int resonate_iters;
        int sync_radius;

        Params()
        :
        lr(0.1f),
        resonate_iters(32),
        sync_radius(1)
        {}
    };

private:
    Int4 hidden_size; // size of hidden/output layer

    Int_Buffer hidden_cis;

    Array<Vec<S>> hidden_vecs;

    Array<Vec<S>> hidden_factors;
    Array<Vec<S>> hidden_factors_temp;

    // weights are ping-pong buffered
    Float_Buffer hidden_learn_vecs_ping;
    Float_Buffer hidden_learn_vecs_pong;

    Float_Buffer_View hidden_learn_vecs_read;
    Float_Buffer_View hidden_learn_vecs_write;
    bool ping;

    Array<Vec<S>> hidden_code_vecs;

    // visible layers and associated descriptors
    Array<Visible_Layer> visible_layers;
    Array<Visible_Layer_Desc> visible_layer_descs;

    Array<Resonator_Mat> hidden_mats;

    // --- kernels ---
    
    void bind_inputs(
        const Int2 &column_pos,
        int vli
    ) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        assert(vl.use_input);

        int visible_column_index = address2(column_pos, Int2(vld.size.x, vld.size.y));

        vl.input_vecs[visible_column_index] = vl.visible_pos_vecs[visible_column_index];

        for (int fi = 0; fi < vld.size.z; fi++) {
            int visible_features_index = fi + vld.size.z * visible_column_index;

            int in_ci = vl.input_cis[visible_features_index];

            vl.input_vecs[visible_column_index] *= vl.visible_code_vecs[in_ci + vld.size.w * fi];
        }
    }

    void forward(
        const Int2 &column_pos,
        bool learn_enabled,
        const Params &params
    ) {
        int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

        Bundle<S> sum = 0;

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];

            if (!vl.use_input)
                continue;

            if (!vl.up_to_date) {
                int diam = vld.radius * 2 + 1;

                // projection
                Float2 h_to_v = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hidden_size.x),
                    static_cast<float>(vld.size.y) / static_cast<float>(hidden_size.y));

                Int2 visible_center = project(column_pos, h_to_v);

                // lower corner
                Int2 field_lower_bound(visible_center.x - vld.radius, visible_center.y - vld.radius);

                // bounds of receptive field, clamped to input size
                Int2 iter_lower_bound(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
                Int2 iter_upper_bound(min(vld.size.x - 1, visible_center.x + vld.radius), min(vld.size.y - 1, visible_center.y + vld.radius));

                Bundle<S> sub_sum = 0;

                for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                    for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                        int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                        sub_sum += vl.input_vecs[visible_column_index];
                    }

                vl.hidden_sums[hidden_column_index] = sub_sum;
            }

            sum += vl.hidden_sums[hidden_column_index];
        }

        Vec<S> hidden_vec = sum.thin();

        // initialize factors to superpositions
        for (int fi = 0; fi < hidden_size.z; fi++) {
            int hidden_features_index = fi + hidden_size.z * hidden_column_index;

            int hidden_cells_start = hidden_size.w * hidden_features_index;

            Bundle<S> init_factor = 0;

            for (int hc = 0; hc < hidden_size.w; hc++) {
                int hidden_cell_index = hc + hidden_cells_start;

                init_factor += hidden_code_vecs[hidden_cell_index];
            }

            hidden_factors[hidden_features_index] = init_factor.thin();
        }

        // resonate
        for (int it = 0; it < params.resonate_iters; it++) {
            for (int fi = 0; fi < hidden_size.z; fi++) {
                int hidden_features_index = fi + hidden_size.z * hidden_column_index;

                Vec<S> temp = hidden_vec;

                // unbind other feature estimates
                for (int ofi = 0; ofi < hidden_size.z; ofi++) {
                    if (ofi == fi)
                        continue;

                    int other_hidden_features_index = ofi + hidden_size.z * hidden_column_index;

                    temp *= hidden_factors[other_hidden_features_index];
                }

                // constrain to codebook
                hidden_factors_temp[hidden_features_index] = (hidden_mats[hidden_features_index] * temp).thin();
            }

            // double buffer
            for (int fi = 0; fi < hidden_size.z; fi++) {
                int hidden_features_index = fi + hidden_size.z * hidden_column_index;

                hidden_factors[hidden_features_index] = hidden_factors_temp[hidden_features_index];
            }
        }

        // clean up
        for (int fi = 0; fi < hidden_size.z; fi++) {
            int hidden_features_index = fi + hidden_size.z * hidden_column_index;

            // find similarity to code
            int max_index = 0;
            int max_similarity = limit_min;

            int hidden_cells_start = hidden_size.w * hidden_features_index;

            for (int hc = 0; hc < hidden_size.w; hc++) {
                int hidden_cell_index = hc + hidden_cells_start;

                int similarity = hidden_factors[hidden_features_index].dot(hidden_code_vecs[hidden_cell_index]);

                if (similarity > max_similarity) {
                    max_similarity = similarity;
                    max_index = hc;
                }
            }

            hidden_cis[hidden_features_index] = max_index;
        }

        if (learn_enabled) {
            for (int fi = 0; fi < hidden_size.z; fi++) {
                int hidden_features_index = fi + hidden_size.z * hidden_column_index;

                int hidden_cell_index = hidden_cis[hidden_features_index] + hidden_features_index * hidden_size.w;

                // move learned code towards vec
                for (int i = 0; i < S; i++) {
                    int index = i + S * hidden_cell_index;

                    // in-place, so can modify read
                    hidden_learn_vecs_read[index] += params.lr * (hidden_factors[hidden_features_index].get(i) - hidden_learn_vecs_read[index]);
                }
            }
        }

        // get final complete cleaned-up vector
        hidden_vec = 1; // set to identity

        // find final hidden vec
        for (int fi = 0; fi < hidden_size.z; fi++) {
            int hidden_features_index = fi + hidden_size.z * hidden_column_index;

            hidden_vec *= hidden_code_vecs[hidden_cis[hidden_features_index] + hidden_features_index * hidden_size.w];//hidden_factors[hidden_features_index];
        }

        hidden_vecs[hidden_column_index] = hidden_vec;
    }

    void local_sync(
        const Int2 &column_pos,
        const Params &params
    ) {
        int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));
        
        // clear write area
        for (int fi = 0; fi < hidden_size.z; fi++) {
            int hidden_features_index = fi + hidden_size.z * hidden_column_index;

            for (int hc = 0; hc < hidden_size.w; hc++) {
                int hidden_cell_index = hc + hidden_features_index * hidden_size.w;

                for (int i = 0; i < S; i++) {
                    int index = i + S * hidden_cell_index;

                    hidden_learn_vecs_write[index] = 0.0f;
                }
            }
        }

        // merge weights
        int count = 0;

        for (int dcx = -params.sync_radius; dcx <= params.sync_radius; dcx++)
            for (int dcy = -params.sync_radius; dcy <= params.sync_radius; dcy++) {
                Int2 other_column_pos(column_pos.x + dcx, column_pos.y + dcy);

                if (in_bounds0(other_column_pos, Int2(hidden_size.x, hidden_size.y))) {
                    int other_hidden_column_index = address2(other_column_pos, Int2(hidden_size.x, hidden_size.y));

                    for (int fi = 0; fi < hidden_size.z; fi++) {
                        int hidden_features_index = fi + hidden_size.z * hidden_column_index;
                        int other_hidden_features_index = fi + hidden_size.z * other_hidden_column_index;

                        for (int hc = 0; hc < hidden_size.w; hc++) {
                            int hidden_cell_index = hc + hidden_features_index * hidden_size.w;
                            int other_hidden_cell_index = hc + other_hidden_features_index * hidden_size.w;

                            for (int i = 0; i < S; i++) {
                                int index = i + S * hidden_cell_index;
                                int other_index = i + S * other_hidden_cell_index;

                                hidden_learn_vecs_write[index] += hidden_learn_vecs_read[other_index];
                            }
                        }
                    }

                    count++;
                }
            }

        float scale = 1.0f / max(1, count);

        // scale
        for (int fi = 0; fi < hidden_size.z; fi++) {
            int hidden_features_index = fi + hidden_size.z * hidden_column_index;

            for (int hc = 0; hc < hidden_size.w; hc++) {
                int hidden_cell_index = hc + hidden_features_index * hidden_size.w;

                // move learned code towards actual vec
                for (int i = 0; i < S; i++) {
                    int index = i + S * hidden_cell_index;

                    hidden_learn_vecs_write[index] *= scale;

                    // update discrete code vecs
                    hidden_code_vecs[hidden_cell_index].set(i, (hidden_learn_vecs_write[index] > 0.0f) * 2 - 1);
                }
            }

            // update matrix
            hidden_mats[hidden_features_index].set_from(hidden_code_vecs, hidden_features_index * hidden_size.w, hidden_size.w);
        }
    }

    void reconstruct(
        const Int2 &column_pos,
        int vli,
        const Params &params
    ) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        int diam = vld.radius * 2 + 1;

        int visible_column_index = address2(column_pos, Int2(vld.size.x, vld.size.y));

        int visible_cells_start = vld.size.z * visible_column_index;

        // projection
        Float2 v_to_h = Float2(static_cast<float>(hidden_size.x) / static_cast<float>(vld.size.x),
            static_cast<float>(hidden_size.y) / static_cast<float>(vld.size.y));

        Float2 h_to_v = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hidden_size.x),
            static_cast<float>(vld.size.y) / static_cast<float>(hidden_size.y));

        Int2 reverse_radii(ceilf(v_to_h.x * (vld.radius * 2 + 1) * 0.5f), ceilf(v_to_h.y * (vld.radius * 2 + 1) * 0.5f));

        Int2 hidden_center = project(column_pos, v_to_h);

        // lower corner
        Int2 field_lower_bound(hidden_center.x - reverse_radii.x, hidden_center.y - reverse_radii.y);

        // bounds of receptive field, clamped to input size
        Int2 iter_lower_bound(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
        Int2 iter_upper_bound(min(hidden_size.x - 1, hidden_center.x + reverse_radii.x), min(hidden_size.y - 1, hidden_center.y + reverse_radii.y));

        Bundle<S> sum = 0;

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                Int2 hidden_pos = Int2(ix, iy);

                int hidden_column_index = address2(hidden_pos, Int2(hidden_size.x, hidden_size.y));

                Int2 visible_center = project(hidden_pos, h_to_v);

                if (in_bounds(column_pos, Int2(visible_center.x - vld.radius, visible_center.y - vld.radius), Int2(visible_center.x + vld.radius + 1, visible_center.y + vld.radius + 1)))
                    sum += hidden_vecs[hidden_column_index];
            }

        // thin and unbind position
        Vec<S> visible_vec = sum.thin() * vl.visible_pos_vecs[visible_column_index];

        // initialize factors to superpositions
        for (int fi = 0; fi < vld.size.z; fi++) {
            int visible_features_index = fi + vld.size.z * visible_column_index;

            Bundle<S> init_factor = 0;

            for (int vc = 0; vc < vld.size.w; vc++)
                init_factor += vl.visible_code_vecs[vc + vld.size.w * fi];

            vl.recon_factors[visible_features_index] = init_factor.thin();
        }

        // resonate
        for (int it = 0; it < params.resonate_iters; it++) {
            for (int fi = 0; fi < vld.size.z; fi++) {
                int visible_features_index = fi + vld.size.z * visible_column_index;

                Vec<S> temp = visible_vec;

                // unbind other feature estimates
                for (int ofi = 0; ofi < vld.size.z; ofi++) {
                    if (ofi == fi)
                        continue;

                    int other_visible_features_index = ofi + vld.size.z * visible_column_index;

                    temp *= vl.recon_factors[other_visible_features_index];
                }

                // constrain to codebook
                vl.recon_factors_temp[visible_features_index] = (vl.visible_mats[fi] * temp).thin();
            }

            // double buffer
            for (int fi = 0; fi < vld.size.z; fi++) {
                int visible_features_index = fi + vld.size.z * visible_column_index;

                vl.recon_factors[visible_features_index] = vl.recon_factors_temp[visible_features_index];
            }
        }

        for (int fi = 0; fi < vld.size.z; fi++) {
            int visible_features_index = fi + vld.size.z * visible_column_index;

            // find similarity to code
            int max_index = 0;
            int max_similarity = limit_min;

            for (int vc = 0; vc < vld.size.w; vc++) {
                int similarity = vl.recon_factors[visible_features_index].dot(vl.visible_code_vecs[vc + vld.size.w * fi]);

                if (similarity > max_similarity) {
                    max_similarity = similarity;
                    max_index = vc;
                }
            }

            // set to most similar code
            vl.recon_cis[visible_features_index] = max_index;
        }
    }

public:
    // create a sparse coding layer with random initialization
    void init_random(
        const Int4 &hidden_size, // hidden/output size
        float positional_scale, // positional encoding scale
        const Array<Visible_Layer_Desc> &visible_layer_descs // descriptors for visible layers
    ) {
        this->visible_layer_descs = visible_layer_descs;

        this->hidden_size = hidden_size;

        visible_layers.resize(visible_layer_descs.size());

        // pre-compute dimensions
        int num_hidden_columns = hidden_size.x * hidden_size.y;
        int num_hidden_features = num_hidden_columns * hidden_size.z;

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

            int num_visible_columns = vld.size.x * vld.size.y;
            int num_visible_features = num_visible_columns * vld.size.z;

            vl.visible_code_vecs.resize(vld.size.z * vld.size.w);

            for (int i = 0; i < vl.visible_code_vecs.size(); i++)
                vl.visible_code_vecs[i] = Vec<S>::randomized();

            // generate temporary positional matrix
            Float_Buffer embedding(S * 3);

            for (int i = 0; i < embedding.size(); i++)
                embedding[i] = rand_normalf() * positional_scale;

            vl.visible_pos_vecs.resize(num_visible_columns);

            for (int x = 0; x < vld.size.x; x++)
                for (int y = 0; y < vld.size.y; y++) {
                    int visible_column_index = y + x * vld.size.y;

                    for (int i = 0; i < S; i++)
                        vl.visible_pos_vecs[visible_column_index].set(i, (cosf(embedding[i * 3] * x + embedding[i * 3 + 1] * y + embedding[i * 3 + 2]) > 0.0f) * 2 - 1);
                }

            vl.input_cis = Int_Buffer(num_visible_features, 0);

            vl.recon_cis = Int_Buffer(num_visible_features, 0);

            vl.recon_factors.resize(num_visible_features);
            vl.recon_factors_temp.resize(vl.recon_factors.size());

            vl.input_vecs.resize(num_visible_columns);

            vl.hidden_sums.resize(num_hidden_columns);

            vl.visible_mats.resize(vld.size.z);

            for (int i = 0; i < vld.size.z; i++)
                vl.visible_mats[i].set_from(vl.visible_code_vecs, i * vld.size.w, vld.size.w);
        }

        hidden_cis = Int_Buffer(num_hidden_features, 0);

        hidden_learn_vecs_ping.resize(num_hidden_features * hidden_size.w * S);

        // set first part, then copy to all other parts (initialize equal in all columns)
        int num_codes_per_column = hidden_size.z * hidden_size.w;
        int num_code_elements_per_column = num_codes_per_column * S;

        for (int i = 0; i < num_code_elements_per_column; i++)
            hidden_learn_vecs_ping[i] = randf(-init_weight_noisef, init_weight_noisef);

        for (int i = 1; i < num_hidden_columns; i++) {
            for (int j = 0; j < num_code_elements_per_column; j++)
                hidden_learn_vecs_ping[j + i * num_code_elements_per_column] = hidden_learn_vecs_ping[j];
        }

        // set up ping pong
        hidden_learn_vecs_pong.resize(hidden_learn_vecs_ping.size());

        hidden_learn_vecs_read = hidden_learn_vecs_ping;
        hidden_learn_vecs_write = hidden_learn_vecs_pong;
        ping = true;

        hidden_code_vecs.resize(num_hidden_features * hidden_size.w);

        for (int i = 0; i < hidden_code_vecs.size(); i++) {
            for (int j = 0; j < S; j++)
                hidden_code_vecs[i].set(j, (hidden_learn_vecs_read[j + i * S] > 0.0f) * 2 - 1);
        }

        hidden_vecs.resize(num_hidden_columns);

        hidden_factors.resize(num_hidden_features);
        hidden_factors_temp.resize(hidden_factors.size());

        // initialize resonator matrices
        hidden_mats.resize(num_hidden_features);

        for (int i = 0; i < num_hidden_features; i++)
            hidden_mats[i].set_from(hidden_code_vecs, i * hidden_size.w, hidden_size.w);
    }

    void set_ignore(
        int vli
    ) {
        visible_layers[vli].use_input = false;
    }

    void set_input_cis(
        int vli,
        Int_Buffer_View input_cis
    ) {
        assert(input_cis.size() == visible_layers[vli].input_cis.size());

        visible_layers[vli].use_input = true;
        visible_layers[vli].up_to_date = false;
        visible_layers[vli].input_cis = input_cis;
    }

    void step(
        bool learn_enabled, // whether to learn
        const Params &params // parameters
    ) {
        int num_hidden_columns = hidden_size.x * hidden_size.y;

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];

            if (!vl.use_input || vl.up_to_date)
                continue;

            int num_visible_columns = vld.size.x * vld.size.y;

            PARALLEL_FOR
            for (int i = 0; i < num_visible_columns; i++)
                bind_inputs(Int2(i / vld.size.y, i % vld.size.y), vli);
        }

        PARALLEL_FOR
        for (int i = 0; i < num_hidden_columns; i++)
            forward(Int2(i / hidden_size.y, i % hidden_size.y), learn_enabled, params);

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];

            if (vl.use_input)
                vl.up_to_date = true;
        }
        
        // synchronize
        PARALLEL_FOR
        for (int i = 0; i < num_hidden_columns; i++)
            local_sync(Int2(i / hidden_size.y, i % hidden_size.y), params);

        // update ping-pong
        ping = !ping;

        if (ping) {
            hidden_learn_vecs_read = hidden_learn_vecs_ping;
            hidden_learn_vecs_write = hidden_learn_vecs_pong;
        }
        else {
            hidden_learn_vecs_read = hidden_learn_vecs_pong;
            hidden_learn_vecs_write = hidden_learn_vecs_ping;
        }
    }

    void reconstruct(
        int vli, // visible layer index to reconstruct
        const Params &params // parameters
    ) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;

        PARALLEL_FOR
        for (int i = 0; i < num_visible_columns; i++)
            reconstruct(Int2(i / vld.size.y, i % vld.size.y), vli, params);
    }

    void clear_state() {
        hidden_cis.fill(0);

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];

            vl.recon_cis.fill(0);
        }
    }

    // serialization
    long size() const { // returns size in Bytes
        long size = sizeof(Int3) + hidden_cis.size() * sizeof(int) + hidden_learn_vecs_ping.size() * sizeof(float) + sizeof(int);

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            const Visible_Layer &vl = visible_layers[vli];

            size += sizeof(Visible_Layer_Desc) + vl.visible_code_vecs.size() * sizeof(Vec<S>) + vl.visible_pos_vecs.size() * sizeof(Vec<S>) + vl.recon_cis.size() * sizeof(int) + sizeof(float);
        }

        return size;
    }

    long state_size() const { // returns size of state in Bytes
        long size = hidden_cis.size() * sizeof(int);

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            const Visible_Layer &vl = visible_layers[vli];

            size += vl.recon_cis.size() * sizeof(int);
        }

        return size;
    }

    long weights_size() const { // returns size of weights in Bytes
        return hidden_learn_vecs_ping.size() * sizeof(float);
    }

    void write(
        Stream_Writer &writer
    ) const {
        writer.write(&hidden_size, sizeof(Int3));

        writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));
        writer.write(&hidden_learn_vecs_read[0], hidden_learn_vecs_read.size() * sizeof(float));

        int num_visible_layers = visible_layers.size();

        writer.write(&num_visible_layers, sizeof(int));
        
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            const Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];

            writer.write(&vld, sizeof(Visible_Layer_Desc));

            writer.write(&vl.visible_code_vecs[0], vl.visible_code_vecs.size() * sizeof(Vec<S>));
            writer.write(&vl.visible_pos_vecs[0], vl.visible_pos_vecs.size() * sizeof(Vec<S>));

            writer.write(&vl.recon_cis[0], vl.recon_cis.size() * sizeof(int));
        }
    }

    void read(
        Stream_Reader &reader
    ) {
        reader.read(&hidden_size, sizeof(Int3));

        int num_hidden_columns = hidden_size.x * hidden_size.y;
        int num_hidden_features = num_hidden_columns * hidden_size.z;

        hidden_cis.resize(num_hidden_features);
        hidden_learn_vecs_ping.resize(num_hidden_features * hidden_size.w * S);

        reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));

        reader.read(&hidden_learn_vecs_ping[0], hidden_learn_vecs_ping.size() * sizeof(float));

        hidden_learn_vecs_pong.resize(hidden_learn_vecs_ping.size());

        hidden_learn_vecs_read = hidden_learn_vecs_ping;
        hidden_learn_vecs_write = hidden_learn_vecs_pong;
        ping = true;

        hidden_code_vecs.resize(num_hidden_features * hidden_size.w);

        for (int i = 0; i < hidden_code_vecs.size(); i++) {
            for (int j = 0; j < S; j++)
                hidden_code_vecs[i].set(j, (hidden_learn_vecs_read[j + i * S] > 0.0f) * 2 - 1);
        }

        // initialize resonator matrices
        hidden_mats.resize(num_hidden_features);

        for (int i = 0; i < num_hidden_features; i++)
            hidden_mats[i].set_from(hidden_code_vecs, i * hidden_size.w, hidden_size.w);

        hidden_vecs.resize(num_hidden_columns);

        hidden_factors.resize(num_hidden_features);
        hidden_factors_temp.resize(hidden_factors.size());

        int num_visible_layers = visible_layers.size();

        reader.read(&num_visible_layers, sizeof(int));

        visible_layers.resize(num_visible_layers);
        visible_layer_descs.resize(num_visible_layers);
        
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            Visible_Layer_Desc &vld = visible_layer_descs[vli];

            reader.read(&vld, sizeof(Visible_Layer_Desc));

            int num_visible_columns = vld.size.x * vld.size.y;
            int num_visible_features = num_visible_columns * vld.size.z;

            vl.visible_code_vecs.resize(vld.size.z * vld.size.w);
            vl.visible_pos_vecs.resize(num_visible_columns);

            reader.read(&vl.visible_code_vecs[0], vl.visible_code_vecs.size() * sizeof(Vec<S>));
            reader.read(&vl.visible_pos_vecs[0], vl.visible_pos_vecs.size() * sizeof(Vec<S>));

            vl.input_cis = Int_Buffer(num_visible_features, 0);

            vl.recon_cis.resize(num_visible_features);

            reader.read(&vl.recon_cis[0], vl.recon_cis.size() * sizeof(int));

            vl.recon_factors.resize(num_visible_features);
            vl.recon_factors_temp.resize(vl.recon_factors.size());

            vl.input_vecs.resize(num_visible_columns);

            vl.hidden_sums.resize(num_hidden_columns);

            vl.visible_mats.resize(vld.size.z);

            for (int i = 0; i < vld.size.z; i++)
                vl.visible_mats[i].set_from(vl.visible_code_vecs, i * vld.size.w, vld.size.w);
        }
    }

    void write_state(
        Stream_Writer &writer
    ) const {
        writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            const Visible_Layer &vl = visible_layers[vli];

            writer.write(&vl.recon_cis[0], vl.recon_cis.size() * sizeof(int));
        }
    }

    void read_state(
        Stream_Reader &reader
    ) {
        reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];

            reader.read(&vl.recon_cis[0], vl.recon_cis.size() * sizeof(int));
        }
    }

    void write_weights(
        Stream_Writer &writer
    ) const {
        writer.write(&hidden_learn_vecs_read[0], hidden_learn_vecs_read.size() * sizeof(float));
    }

    void read_weights(
        Stream_Reader &reader
    ) {
        reader.read(&hidden_learn_vecs_read[0], hidden_learn_vecs_read.size() * sizeof(float));
    }

    // get the number of visible layers
    int get_num_visible_layers() const {
        return visible_layers.size();
    }

    // get a visible layer
    Visible_Layer &get_visible_layer(
        int i // index of visible layer
    ) {
        return visible_layers[i];
    }

    // get a visible layer
    const Visible_Layer &get_visible_layer(
        int i // index of visible layer
    ) const {
        return visible_layers[i];
    }

    // get a visible layer descriptor
    const Visible_Layer_Desc &get_visible_layer_desc(
        int i // index of visible layer
    ) const {
        return visible_layer_descs[i];
    }

    // get the hidden states
    const Int_Buffer &get_hidden_cis() const {
        return hidden_cis;
    }

    // get the hidden size
    const Int4 &get_hidden_size() const {
        return hidden_size;
    }
};
}
