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
#include "assoc.h"

namespace aon {
template<int S, int L>
class Encoder {
public:
    // visible layer descriptor
    struct Visible_Layer_Desc {
        Int2 size; // size of input

        int radius; // radius onto input

        // defaults
        Visible_Layer_Desc()
        :
        size(4, 4),
        radius(2)
        {}
    };

    // visible layer
    struct Visible_Layer {
        Array<Vec<S, L>> visible_pos_vecs; // positional encodings
        Array<Vec<S, L>> input_vecs; // incoming input
        Array<Vec<S, L>> visible_vecs; // input with bound position
        Array<Vec<S, L>> recon_vecs; // reconstructed input

        Array<Bundle<S, L>> hidden_sums; // sum cache

        bool use_input;
        bool up_to_date;
        
        Visible_Layer()
        :
        use_input(false),
        up_to_date(false)
        {}
    };

    struct Params {
        int clean_iters;
        int assoc_limit;

        Params()
        :
        clean_iters(32),
        assoc_limit(1024)
        {}
    };

private:
    Int2 hidden_size; // size of hidden/output layer

    Array<Vec<S, L>> hidden_vecs;

    // visible layers and associated descriptors
    Array<Visible_Layer> visible_layers;
    Array<Visible_Layer_Desc> visible_layer_descs;

    Int_Buffer hidden_assoc_buffer;
    Array<Assoc<S, L>> hidden_assocs;

    // --- kernels ---
    
    void bind_inputs(
        const Int2 &column_pos,
        int vli
    ) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        assert(vl.use_input);

        int visible_column_index = address2(column_pos, Int2(vld.size.x, vld.size.y));

        vl.visible_vecs[visible_column_index] = vl.visible_pos_vecs[visible_column_index] * vl.input_vecs[visible_column_index];
    }

    void forward(
        const Int2 &column_pos,
        bool learn_enabled,
        const Params &params
    ) {
        int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

        Bundle<S, L> sum = 0;

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

                Bundle<S, L> sub_sum = 0;

                for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                    for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                        int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                        sub_sum += vl.visible_vecs[visible_column_index];
                    }

                vl.hidden_sums[hidden_column_index] = sub_sum;
            }

            sum += vl.hidden_sums[hidden_column_index];
        }

        Vec<S, L> hidden_vec = sum.thin();

        // clean up
        Vec<S, L> hidden_vec_clean = hidden_vec;

        for (int it = 0; it < params.clean_iters; it++)
            hidden_vec_clean = hidden_assocs[hidden_column_index] * hidden_vec_clean;

        hidden_vecs[hidden_column_index] = hidden_vec_clean;

        if (learn_enabled)
            hidden_assocs[hidden_column_index].assoc(hidden_vec, params.assoc_limit);
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

        Bundle<S, L> sum = 0;

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                Int2 hidden_pos = Int2(ix, iy);

                int hidden_column_index = address2(hidden_pos, Int2(hidden_size.x, hidden_size.y));

                Int2 visible_center = project(hidden_pos, h_to_v);

                if (in_bounds(column_pos, Int2(visible_center.x - vld.radius, visible_center.y - vld.radius), Int2(visible_center.x + vld.radius + 1, visible_center.y + vld.radius + 1)))
                    sum += hidden_vecs[hidden_column_index];
            }

        // thin and unbind position
        Vec<S, L> visible_vec = sum.thin() * vl.visible_pos_vecs[visible_column_index];

        vl.recon_vecs[visible_column_index] = visible_vec;
    }

public:
    // create a sparse coding layer with random initialization
    void init_random(
        const Int2 &hidden_size, // hidden/output size
        float positional_scale, // positional encoding scale
        const Array<Visible_Layer_Desc> &visible_layer_descs // descriptors for visible layers
    ) {
        this->visible_layer_descs = visible_layer_descs;

        this->hidden_size = hidden_size;

        visible_layers.resize(visible_layer_descs.size());

        // pre-compute dimensions
        int num_hidden_columns = hidden_size.x * hidden_size.y;

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

            int num_visible_columns = vld.size.x * vld.size.y;

            // generate temporary positional matrix
            Float_Buffer embedding(S * 3);

            for (int i = 0; i < embedding.size(); i++)
                embedding[i] = rand_normalf() * positional_scale;

            vl.visible_pos_vecs.resize(num_visible_columns);

            for (int x = 0; x < vld.size.x; x++)
                for (int y = 0; y < vld.size.y; y++) {
                    int visible_column_index = y + x * vld.size.y;

                    for (int i = 0; i < S; i++)
                        vl.visible_pos_vecs[visible_column_index][i] = static_cast<int>(modf(embedding[i * 3] * x + embedding[i * 3 + 1] * y + embedding[i * 3 + 2], 1.0f) * (L - 1) + 0.5f);
                }

            vl.input_vecs.resize(num_visible_columns);
            vl.visible_vecs.resize(num_visible_columns);
            vl.recon_vecs.resize(num_visible_columns, 0);

            vl.hidden_sums.resize(num_hidden_columns);
        }

        hidden_vecs.resize(num_hidden_columns, 0);

        hidden_assoc_buffer.resize(num_hidden_columns * Assoc<S, L>::C);

        hidden_assocs.resize(num_hidden_columns);

        for (int i = 0; i < num_hidden_columns; i++)
            hidden_assocs[i].set_from(&hidden_assoc_buffer[i * Assoc<S, L>::C]);
    }

    void set_ignore(
        int vli
    ) {
        visible_layers[vli].use_input = false;
    }

    void set_input_vecs(
        int vli,
        Array_View<Vec<S, L>> input_vecs
    ) {
        assert(input_vecs.size() == visible_layers[vli].input_vecs.size());

        visible_layers[vli].use_input = true;
        visible_layers[vli].up_to_date = false;
        visible_layers[vli].input_vecs = input_vecs;
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
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];

            vl.recon_vecs.fill(0);
        }

        hidden_vecs.fill(0);
    }

    // serialization
    long size() const { // returns size in Bytes
        long size = sizeof(Int2) + sizeof(int);

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            const Visible_Layer &vl = visible_layers[vli];

            size += sizeof(Visible_Layer_Desc) + vl.visible_pos_vecs.size () * sizeof(Vec<S, L>) + vl.recon_vecs.size() * sizeof(Vec<S, L>);
        }

        size += hidden_vecs.size() * sizeof(Vec<S, L>);

        size += hidden_assoc_buffer.size() * sizeof(int);

        return size;
    }

    long state_size() const { // returns size of state in Bytes
        long size = 0;

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            const Visible_Layer &vl = visible_layers[vli];

            size += vl.recon_vecs.size() * sizeof(Vec<S, L>);
        }

        size += hidden_vecs.size() * sizeof(Vec<S, L>);

        return size;
    }

    long weights_size() const { // returns size of weights in Bytes
        return hidden_assoc_buffer.size() * sizeof(int);
    }

    void write(
        Stream_Writer &writer
    ) const {
        writer.write(&hidden_size, sizeof(Int2));

        int num_visible_layers = visible_layers.size();

        writer.write(&num_visible_layers, sizeof(int));
        
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            const Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];

            writer.write(&vld, sizeof(Visible_Layer_Desc));

            writer.write(&vl.visible_pos_vecs[0], vl.visible_pos_vecs.size() * sizeof(Vec<S, L>));
            writer.write(&vl.recon_vecs[0], vl.recon_vecs.size() * sizeof(Vec<S, L>));
        }

        writer.write(&hidden_vecs[0], hidden_vecs.size() * sizeof(Vec<S, L>));

        writer.write(&hidden_assoc_buffer[0], hidden_assoc_buffer.size() * sizeof(int));
    }

    void read(
        Stream_Reader &reader
    ) {
        reader.read(&hidden_size, sizeof(Int2));

        int num_hidden_columns = hidden_size.x * hidden_size.y;

        int num_visible_layers = visible_layers.size();

        reader.read(&num_visible_layers, sizeof(int));

        visible_layers.resize(num_visible_layers);
        visible_layer_descs.resize(num_visible_layers);
        
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            Visible_Layer_Desc &vld = visible_layer_descs[vli];

            reader.read(&vld, sizeof(Visible_Layer_Desc));

            int num_visible_columns = vld.size.x * vld.size.y;

            vl.visible_pos_vecs.resize(num_visible_columns);
            vl.input_vecs.resize(num_visible_columns);
            vl.visible_vecs.resize(num_visible_columns);
            vl.recon_vecs.resize(num_visible_columns);

            reader.read(&vl.visible_pos_vecs[0], vl.visible_pos_vecs.size() * sizeof(Vec<S, L>));
            reader.read(&vl.recon_vecs[0], vl.recon_vecs.size() * sizeof(Vec<S, L>));
        }

        reader.read(&hidden_vecs[0], hidden_vecs.size() * sizeof(Vec<S, L>));

        hidden_assoc_buffer.resize(num_hidden_columns * Assoc<S, L>::C);

        reader.read(&hidden_assoc_buffer[0], hidden_assoc_buffer.size() * sizeof(int));

        hidden_assocs.resize(num_hidden_columns);

        for (int i = 0; i < num_hidden_columns; i++)
            hidden_assocs[i].set_from(&hidden_assoc_buffer[i * Assoc<S, L>::C]);
    }

    void write_state(
        Stream_Writer &writer
    ) const {
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            const Visible_Layer &vl = visible_layers[vli];

            writer.write(&vl.recon_vecs[0], vl.recon_vecs.size() * sizeof(Vec<S, L>));
        }

        writer.write(&hidden_vecs[0], hidden_vecs.size() * sizeof(Vec<S, L>));
    }

    void read_state(
        Stream_Reader &reader
    ) {
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];

            reader.read(&vl.recon_vecs[0], vl.recon_vecs.size() * sizeof(Vec<S, L>));
        }

        reader.read(&hidden_vecs[0], hidden_vecs.size() * sizeof(Vec<S, L>));
    }

    void write_weights(
        Stream_Writer &writer
    ) const {
        writer.write(&hidden_assoc_buffer[0], hidden_assoc_buffer.size() * sizeof(int));
    }

    void read_weights(
        Stream_Reader &reader
    ) {
        reader.read(&hidden_assoc_buffer[0], hidden_assoc_buffer.size() * sizeof(int));
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
    const Array<Vec<S, L>> &get_hidden_vecs() const {
        return hidden_vecs;
    }

    // get the hidden size
    const Int2 &get_hidden_size() const {
        return hidden_size;
    }
};
}
