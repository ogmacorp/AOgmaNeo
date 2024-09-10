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

        Int_Buffer recon_cis;

        bool use_input;
        bool up_to_date;
        
        float importance;

        Visible_Layer()
        :
        use_input(false),
        up_to_date(false),
        importance(1.0f)
        {}
    };

    struct Params {
        float lr; // code learning rate
        int resonate_iters;

        Params()
        :
        lr(0.01f),
        resonate_iters(16)
        {}
    };

private:
    Int4 hidden_size; // size of hidden/output layer

    Int_Buffer hidden_cis;

    Array<Vec<S>> hidden_vecs;

    Float_Buffer hidden_learn_vecs;
    Array<Vec<S>> hidden_code_vecs;

    // visible layers and associated descriptors
    Array<Visible_Layer> visible_layers;
    Array<Visible_Layer_Desc> visible_layer_descs;

    // --- helpers ---

    Bundle<S> multiply_a_at(
        const Array<Vec<S>> &codes,
        int codes_start, // start index into codebook
        Vec<S> v, // vector to multiply
        int num_codes
    ) {
        Bundle<S> vd = 0;

        // accumulate from above diagonal (symmetric)
        for (int c = 0; c < num_codes; c++) {
            for (int c2 = 0; c2 < c; c2++) { // only compute up to diagonal
                int a_at_elem = codes[codes_start + c].dot(codes[codes_start + c2]);

                // exploit symmetry
                for (int r = 0; r < S; r++) {
                    // upper triangle
                    vd[r] += a_at_elem * v.get(r);

                    // lower triangle
                    int r_comp = S - r;

                    vd[r_comp] += a_at_elem * v.get(r_comp);
                }
            }
        }

        // accumlate from diagonal
        for (int c = 0; c < num_codes; c++) {
            int a_at_elem = codes[codes_start + c].dot(codes[codes_start + c]);

            for (int r = 0; r < S; r++)
                vd[r] += a_at_elem * v.get(r);
        }

        return vd;
    }
    
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

            vl.input_vecs[visible_column_index] *= vl.visible_code_vecs[in_ci + vld.size.w * visible_features_index];
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

                for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                    for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                        int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                        Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                        sum += vl.input_vecs[visible_column_index];
                    }
            }
        }

        Vec<S> hidden_vec = sum.thin();

        // resonate
        for (int it = 0; it < params.resonate_iters; it++) {
            for (int fi = 0; fi < hidden_size.z; fi++) {
                // compute a self-correlation vector per feature
                int hidden_features_index = fi + hidden_size.z * hidden_column_index;

                int hidden_cells_start = hidden_size.w * hidden_features_index;

                Vec<S> temp = hidden_vec;

                // bind other feature estimates
                for (int ofi = 0; ofi < hidden_size.z; ofi++) {
                    if (ofi == fi)
                        continue;

                    int other_hidden_features_index = ofi + hidden_size.z * hidden_column_index;

                    temp *= hidden_code_vecs[hidden_cis[other_hidden_features_index] + other_hidden_features_index * hidden_size.w];
                }

                temp = multiply_a_at(hidden_code_vecs, hidden_size.w * hidden_features_index).thin() * temp;

                // find similarity to code
                int max_index = 0;
                int max_similarity = limit_min;

                for (int hc = 0; hc < hidden_size.w; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int similarity = temp.dot(hidden_code_vecs[hidden_cell_index]);

                    if (similarity > max_similarity) {
                        max_similarity = similarity;
                        max_index = hc;
                    }
                }

                // set to most similar code
                hidden_cis[hidden_features_index] = max_index;
            }
        }

        hidden_vec = 1; // set to identity

        // find final hidden vec
        for (int fi = 0; fi < hidden_size.z; fi++) {
            int hidden_features_index = fi + hidden_size.z * hidden_column_index;

            hidden_vec *= hidden_code_vecs[hidden_cis[hidden_features_index] + hidden_features_index * hidden_size.w];
        }

        hidden_vecs[hidden_column_index] = hidden_vec;
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

                if (in_bounds(column_pos, Int2(visible_center.x - vld.radius, visible_center.y - vld.radius), Int2(visible_center.x + vld.radius + 1, visible_center.y + vld.radius + 1))) {
                    Int2 offset(column_pos.x - visible_center.x + vld.radius, column_pos.y - visible_center.y + vld.radius);

                    sum += hidden_vecs[hidden_column_index];
                }
            }

        Vec<S> visible_vec = sum.thin();

        // resonate
        for (int it = 0; it < params.resonate_iters; it++) {
            for (int fi = 0; fi < vld.size.z; fi++) {
                // compute a self-correlation vector per feature
                int visible_features_index = fi + vld.size.z * visible_column_index;

                int visible_cells_start = vld.size.w * visible_features_index;

                Vec<S> temp = visible_vec;

                // bind other feature estimates
                for (int ofi = 0; ofi < vld.size.z; ofi++) {
                    if (ofi == fi)
                        continue;

                    int other_visible_features_index = ofi + vld.size.z * visible_column_index;

                    temp *= vld.visible_code_vecs[vl.recon_cis[other_visible_features_index] + other_visible_features_index * vld.size.w];
                }

                temp = multiply_a_at(vl.visible_code_vecs, vld.size.w * visible_features_index).thin() * temp;

                // find similarity to code
                int max_index = 0;
                int max_similarity = limit_min;

                for (int vc = 0; vc < vld.size.w; vc++) {
                    int visible_cell_index = vc + visible_cells_start;

                    int similarity = temp.dot(vld.visible_code_vecs[visible_cell_index]);

                    if (similarity > max_similarity) {
                        max_similarity = similarity;
                        max_index = vc;
                    }
                }

                // set to most similar code
                vl.recon_cis[visible_features_index] = max_index;
            }
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

            vl.visible_code_vecs.resize(num_hidden_features * vld.size.w);

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

                    int visible_vecs_start = S * visible_column_index;

                    for (int i = 0; i < S; i++) {
                        int visible_vec_index = i + visible_vecs_start;

                        vl.visible_pos_vecs[visible_column_index].set(i, (cosf(embedding[visible_vec_index * 3] * x + embedding[visible_vec_index * 3 + 1] * y + embedding[visible_vec_index * 3 + 2]) > 0.0f) * 2 - 1);
                    }
                }

            vl.input_cis = Int_Buffer(num_visible_columns, 0);
            vl.recon_cis = Int_Buffer(num_visible_columns, 0);

            vl.input_vecs.resize(num_visible_columns);
        }

        hidden_cis = Int_Buffer(num_hidden_columns, 0);

        hidden_code_vecs.resize(num_hidden_features * hidden_size.w);

        for (int i = 0; i < hidden_code_vecs.size(); i++)
            hidden_code_vecs[i] = Vec<S>::randomized();

        hidden_vecs.resize(num_hidden_columns);
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
    );

    void reconstruct(
        int vli, // visible layer index to reconstruct
        const Params &params // parameters
    );

    void clear_state();

    // serialization
    long size() const; // returns size in Bytes
    long state_size() const; // returns size of state in Bytes
    long weights_size() const; // returns size of weights in Bytes

    void write(
        Stream_Writer &writer
    ) const;

    void read(
        Stream_Reader &reader
    );

    void write_state(
        Stream_Writer &writer
    ) const;

    void read_state(
        Stream_Reader &reader
    );

    void write_weights(
        Stream_Writer &writer
    ) const;

    void read_weights(
        Stream_Reader &reader
    );

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
