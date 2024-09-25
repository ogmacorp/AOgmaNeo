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
#include "predictor.h"

namespace aon {
template<int S, int L>
class Layer {
public:
    // visible layer descriptor
    struct Visible_Layer_Desc {
        Int2 size; // size of input

        int radius; // radius onto input

        float positional_scale;

        Byte predictable;

        // defaults
        Visible_Layer_Desc()
        :
        size(4, 4),
        radius(2),
        positional_scale(1.0f),
        predictable(false)
        {}
    };

    // visible layer
    struct Visible_Layer {
        Array<Vec<S, L>> visible_pos_vecs; // positional encodings
        Array<Vec<S, L>> visible_vecs; // input with bound position
        Array<Vec<S, L>> pred_vecs; // reconstructed input
    };

private:
    Int2 hidden_size; // size of hidden/output layer

    Array<Vec<S, L>> hidden_vecs_all;
    Array<Vec<S, L>> hidden_vecs_pred;
    Array<Vec<S, L>> hidden_vecs_pred_next;

    // visible layers and associated descriptors
    Array<Visible_Layer> visible_layers;
    Array<Visible_Layer_Desc> visible_layer_descs;

    Array<Predictor<S, L>> predictors;

    // --- kernels ---
    
    void bind_inputs(
        const Int2 &column_pos,
        Array_View<Vec<S, L>> input_vecs,
        int vli
    ) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        int visible_column_index = address2(column_pos, Int2(vld.size.x, vld.size.y));

        vl.visible_vecs[visible_column_index] = vl.visible_pos_vecs[visible_column_index] * input_vecs[visible_column_index];
    }

    void forward(
        const Int2 &column_pos
    ) {
        int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

        Bundle<S, L> sum_all = 0;
        Bundle<S, L> sum_pred = 0;

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];

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

                    sum_all += vl.visible_vecs[visible_column_index];

                    if (vld.predictable)
                        sum_pred += vl.visible_vecs[visible_column_index];
                }
        }

        Vec<S, L> hidden_vec_all = sum_all.thin();
        Vec<S, L> hidden_vec_pred = sum_pred.thin();

        hidden_vecs_all[hidden_column_index] = (hidden_vec_all + hidden_vec_all * hidden_vecs_all[hidden_column_index]).thin(); // recurrent bind
        hidden_vecs_pred[hidden_column_index] = hidden_vec_pred;
    }

    void predict(
        const Int2 &column_pos,
        Array_View<Vec<S, L>> feedback_vecs,
        bool learn_enabled,
        const Layer_Params &params
    ) {
        int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

        Vec<S, L> pred_input_vec = hidden_vecs_all[hidden_column_index];

        if (feedback_vecs.size() != 0)
            pred_input_vec = (feedback_vecs[hidden_column_index] + pred_input_vec).thin();

        Vec<S, L> hidden_vec_pred_next = predictors[hidden_column_index].step(pred_input_vec, hidden_vecs_pred[hidden_column_index], learn_enabled, params);

        hidden_vecs_pred_next[hidden_column_index] = hidden_vec_pred_next;
    }

    void backward(
        const Int2 &column_pos,
        int vli
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
                    sum += hidden_vecs_pred_next[hidden_column_index];
            }

        // thin and unbind position
        vl.pred_vecs[visible_column_index] = sum.thin() / vl.visible_pos_vecs[visible_column_index];
    }

public:
    // create a sparse coding layer with random initialization
    void init_random(
        const Int2 &hidden_size, // hidden/output size
        int hidden_segments,
        int hidden_length,
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
                embedding[i] = rand_normalf() * vld.positional_scale;

            vl.visible_pos_vecs.resize(num_visible_columns);

            const float vld_size_x_inv = 1.0f / vld.size.x;
            const float vld_size_y_inv = 1.0f / vld.size.y;

            for (int x = 0; x < vld.size.x; x++)
                for (int y = 0; y < vld.size.y; y++) {
                    int visible_column_index = y + x * vld.size.y;

                    for (int i = 0; i < S; i++) {
                        float f = embedding[i * 3] * x * vld_size_x_inv + embedding[i * 3 + 1] * y * vld_size_y_inv + embedding[i * 3 + 2];

                        if (f < 0.0f)
                            f = 1.0f - modf(-f, 1.0f);
                        else
                            f = modf(f, 1.0f);

                        vl.visible_pos_vecs[visible_column_index][i] = static_cast<int>(f * L);
                    }
                }

            vl.visible_vecs.resize(num_visible_columns);
            vl.pred_vecs = Array<Vec<S, L>>(num_visible_columns, 0);
        }

        hidden_vecs_all = Array<Vec<S, L>>(num_hidden_columns, 0);
        hidden_vecs_pred = Array<Vec<S, L>>(num_hidden_columns, 0);
        hidden_vecs_pred_next = Array<Vec<S, L>>(num_hidden_columns, 0);

        predictors.resize(num_hidden_columns);

        for (int i = 0; i < num_hidden_columns; i++)
            predictors[i].init(hidden_segments, hidden_length);
    }

    void forward(
        Array<Array_View<Vec<S, L>>> input_vecs
    ) {
        int num_hidden_columns = hidden_size.x * hidden_size.y;

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];

            int num_visible_columns = vld.size.x * vld.size.y;

            PARALLEL_FOR
            for (int i = 0; i < num_visible_columns; i++)
                bind_inputs(Int2(i / vld.size.y, i % vld.size.y), input_vecs[vli], vli);
        }

        PARALLEL_FOR
        for (int i = 0; i < num_hidden_columns; i++)
            forward(Int2(i / hidden_size.y, i % hidden_size.y));
    }

    void predict(
        Array_View<Vec<S, L>> feedback_vecs, // can be empty
        bool learn_enabled, // whether to learn
        const Layer_Params &params // parameters
    ) {
        int num_hidden_columns = hidden_size.x * hidden_size.y;

        PARALLEL_FOR
        for (int i = 0; i < num_hidden_columns; i++)
            predict(Int2(i / hidden_size.y, i % hidden_size.y), feedback_vecs, learn_enabled, params);
    }

    void backward(
        int vli // visible layer index to reconstruct
    ) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;

        PARALLEL_FOR
        for (int i = 0; i < num_visible_columns; i++)
            backward(Int2(i / vld.size.y, i % vld.size.y), vli);
    }

    void clear_state() {
        hidden_vecs_all.fill(0);
    }

    // serialization
    long size() const { // returns size in Bytes
        long size = sizeof(Int2) + sizeof(int);

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            const Visible_Layer &vl = visible_layers[vli];

            size += sizeof(Visible_Layer_Desc) + vl.visible_pos_vecs.size() * sizeof(Vec<S, L>) + vl.pred_vecs.size() * sizeof(Vec<S, L>);
        }

        size += 3 * hidden_vecs_all.size() * sizeof(Vec<S, L>);

        size += predictors.size() * predictors[0].size();

        return size;
    }

    long state_size() const { // returns size of state in Bytes
        return hidden_vecs_all.size() * sizeof(Vec<S, L>);
    }

    long weights_size() const { // returns size of weights in Bytes
        return predictors.size() * predictors[0].weights_size();
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
            writer.write(&vl.pred_vecs[0], vl.pred_vecs.size() * sizeof(Vec<S, L>));
        }

        writer.write(&hidden_vecs_all[0], hidden_vecs_all.size() * sizeof(Vec<S, L>));
        writer.write(&hidden_vecs_pred[0], hidden_vecs_pred.size() * sizeof(Vec<S, L>));
        writer.write(&hidden_vecs_pred_next[0], hidden_vecs_pred_next.size() * sizeof(Vec<S, L>));

        for (int i = 0; i < predictors.size(); i++)
            predictors[i].write(writer);
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
            vl.visible_vecs.resize(num_visible_columns);
            vl.pred_vecs.resize(num_visible_columns);

            reader.read(&vl.visible_pos_vecs[0], vl.visible_pos_vecs.size() * sizeof(Vec<S, L>));
            reader.read(&vl.pred_vecs[0], vl.pred_vecs.size() * sizeof(Vec<S, L>));
        }

        hidden_vecs_all.resize(num_hidden_columns);
        hidden_vecs_pred.resize(num_hidden_columns);
        hidden_vecs_pred_next.resize(num_hidden_columns);

        reader.read(&hidden_vecs_all[0], hidden_vecs_all.size() * sizeof(Vec<S, L>));
        reader.read(&hidden_vecs_pred[0], hidden_vecs_pred.size() * sizeof(Vec<S, L>));
        reader.read(&hidden_vecs_pred_next[0], hidden_vecs_pred_next.size() * sizeof(Vec<S, L>));

        predictors.resize(num_hidden_columns);

        for (int i = 0; i < num_hidden_columns; i++)
            predictors[i].read(reader);
    }

    void write_state(
        Stream_Writer &writer
    ) const {
        writer.write(&hidden_vecs_all[0], hidden_vecs_all.size() * sizeof(Vec<S, L>));
    }

    void read_state(
        Stream_Reader &reader
    ) {
        reader.read(&hidden_vecs_all[0], hidden_vecs_all.size() * sizeof(Vec<S, L>));
    }

    void write_weights(
        Stream_Writer &writer
    ) const {
        for (int i = 0; i < predictors.size(); i++)
            predictors[i].write(writer);
    }

    void read_weights(
        Stream_Reader &reader
    ) {
        for (int i = 0; i < predictors.size(); i++)
            predictors[i].read(reader);
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
        return hidden_vecs_all;
    }

    // get the hidden size
    const Int2 &get_hidden_size() const {
        return hidden_size;
    }
};
}
