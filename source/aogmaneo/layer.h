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
        Array<Vec<S, L>> visible_vecs; // input with bound position
        Array<Vec<S, L>> pred_vecs; // reconstructed input
    };

    struct Params {
        int clean_iters;
        float scale;
        float lr;

        Params()
        :
        clean_iters(1),
        scale(4.0f),
        lr(0.04f)
        {}
    };

private:
    Int2 hidden_size; // size of hidden/output layer

    Array<Vec<S, L>> hidden_vecs;
    Array<Vec<S, L>> hidden_vecs_prev;
    Array<Vec<S, L>> hidden_vecs_pred;
    Array<Vec<S, L>> hidden_vecs_pred_fb; // with feedback

    // visible layers and associated descriptors
    Array<Visible_Layer> visible_layers;
    Array<Visible_Layer_Desc> visible_layer_descs;

    Byte_Buffer predictor_weights;
    Float_Buffer predictor_hiddens;
    Array<Predictor<S * 2, L, S, L>> predictors;

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
        const Int2 &column_pos,
        bool learn_enabled,
        const Params &params
    ) {
        int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

        Bundle<S, L> sum = 0;

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

                    sum += vl.visible_vecs[visible_column_index];
                }
        }

        hidden_vecs_prev[hidden_column_index] = hidden_vecs[hidden_column_index];

        Vec<S, L> hidden_vec = sum.thin();

        hidden_vecs[hidden_column_index] = hidden_vec;

        // clean up
        Vec<S, L> hidden_vec_pred = predictors[hidden_column_index].multiply(hidden_vec, params);

        hidden_vecs_pred[hidden_column_index] = hidden_vec_pred;
    }

    void add_feedback(
        const Int2 &column_pos,
        Array_View<Vec<S, L>> feedback_vecs,
        bool learn_enabled,
        const Params &params
    ) {
        int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

        hidden_vecs_pred_fb[hidden_column_index] = (hidden_vecs_pred[hidden_column_index] + feedback_vecs[hidden_column_index]).thin();

        if (learn_enabled)
            predictors[hidden_column_index].learn(hidden_vecs_prev[hidden_column_index], hidden_vecs[hidden_column_index], params);
    }

    void backward(
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
                    sum += hidden_vecs_pred_fb[hidden_column_index];
            }

        // thin and unbind position
        Vec<S, L> visible_vec = sum.thin() / vl.visible_pos_vecs[visible_column_index];

        vl.pred_vecs[visible_column_index] = visible_vec;
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

                    for (int i = 0; i < S; i++) {
                        float f = modf(embedding[i * 3] * x + embedding[i * 3 + 1] * y + embedding[i * 3 + 2], 1.0f);

                        if (f < 0.0f)
                            f += 1.0f;

                        vl.visible_pos_vecs[visible_column_index][i] = static_cast<int>(f * (L - 1) + 0.5f);
                    }
                }

            vl.visible_vecs.resize(num_visible_columns);

            vl.pred_vecs = Array<Vec<S, L>>(num_visible_columns, 0);
        }

        hidden_vecs = Array<Vec<S, L>>(num_hidden_columns, 0);
        hidden_vecs_pred = Array<Vec<S, L>>(num_hidden_columns, 0);

        hidden_vecs_pred_fb.resize(num_hidden_columns);

        predictor_weights.resize(num_hidden_columns * Predictor<S, L>::C);

        for (int i = 0; i < predictor_weights.size(); i++)
            predictor_weights[i] = 127 + (rand() % init_weight_noisei) - init_weight_noisei / 2;

        predictor_hiddens.resize(num_hidden_columns * Predictor<S, L>::N);

        predictors.resize(num_hidden_columns);

        for (int i = 0; i < num_hidden_columns; i++)
            predictors[i].set_from(&predictor_weights[i * Predictor<S, L>::C], &predictor_hiddens[i * Predictor<S, L>::N]);
    }

    void forward(
        Array<Array_View<Vec<S, L>>> input_vecs,
        bool learn_enabled, // whether to learn
        const Params &params // parameters
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
            forward(Int2(i / hidden_size.y, i % hidden_size.y), learn_enabled, params);

        hidden_vecs_pred_fb = hidden_vecs_pred; // copy in case no feedback is added
    }

    void add_feedback(
        Array_View<Vec<S, L>> feedback_vecs
    ) {
        int num_hidden_columns = hidden_size.x * hidden_size.y;

        PARALLEL_FOR
        for (int i = 0; i < num_hidden_columns; i++)
            add_feedback(Int2(i / hidden_size.y, i % hidden_size.y), feedback_vecs);
    }

    void backward(
        int vli, // visible layer index to reconstruct
        const Params &params // parameters
    ) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;

        PARALLEL_FOR
        for (int i = 0; i < num_visible_columns; i++)
            backward(Int2(i / vld.size.y, i % vld.size.y), vli, params);
    }

    void clear_state() {
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];

            vl.pred_vecs.fill(0);
        }

        hidden_vecs.fill(0);
        hidden_vecs_pred.fill(0);
    }

    // serialization
    long size() const { // returns size in Bytes
        long size = sizeof(Int2) + sizeof(int);

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            const Visible_Layer &vl = visible_layers[vli];

            size += sizeof(Visible_Layer_Desc) + vl.visible_pos_vecs.size () * sizeof(Vec<S, L>) + vl.pred_vecs.size() * sizeof(Vec<S, L>);
        }

        size += 2 * hidden_vecs.size() * sizeof(Vec<S, L>);

        size += predictor_weights.size() * sizeof(Byte);

        return size;
    }

    long state_size() const { // returns size of state in Bytes
        long size = 0;

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            const Visible_Layer &vl = visible_layers[vli];

            size += vl.pred_vecs.size() * sizeof(Vec<S, L>);
        }

        size += 2 * hidden_vecs.size() * sizeof(Vec<S, L>);

        return size;
    }

    long weights_size() const { // returns size of weights in Bytes
        return predictor_weights.size() * sizeof(Byte);
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

        writer.write(&hidden_vecs[0], hidden_vecs.size() * sizeof(Vec<S, L>));
        writer.write(&hidden_vecs_pred[0], hidden_vecs_pred.size() * sizeof(Vec<S, L>));

        writer.write(&num_visible_layers, sizeof(int));

        writer.write(&predictor_weights[0], predictor_weights.size() * sizeof(Byte));
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

        reader.read(&hidden_vecs[0], hidden_vecs.size() * sizeof(Vec<S, L>));
        reader.read(&hidden_vecs_pred[0], hidden_vecs_pred.size() * sizeof(Vec<S, L>));

        hidden_vecs_pred_fb.resize(num_hidden_columns);

        predictor_weights.resize(num_hidden_columns * Predictor<S, L>::C);

        reader.read(&predictor_weights[0], predictor_weights.size() * sizeof(Byte));

        predictor_hiddens.resize(num_hidden_columns * Predictor<S, L>::N);

        predictors.resize(num_hidden_columns);

        for (int i = 0; i < num_hidden_columns; i++)
            predictors[i].set_from(&predictor_weights[i * Predictor<S, L>::C], &predictor_hiddens[i * Predictor<S, L>::N]);
    }

    void write_state(
        Stream_Writer &writer
    ) const {
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            const Visible_Layer &vl = visible_layers[vli];

            writer.write(&vl.pred_vecs[0], vl.pred_vecs.size() * sizeof(Vec<S, L>));
        }

        writer.write(&hidden_vecs[0], hidden_vecs.size() * sizeof(Vec<S, L>));
        writer.write(&hidden_vecs_pred[0], hidden_vecs_pred.size() * sizeof(Vec<S, L>));
    }

    void read_state(
        Stream_Reader &reader
    ) {
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];

            reader.read(&vl.pred_vecs[0], vl.pred_vecs.size() * sizeof(Vec<S, L>));
        }

        reader.read(&hidden_vecs[0], hidden_vecs.size() * sizeof(Vec<S, L>));
        reader.read(&hidden_vecs_pred[0], hidden_vecs_pred.size() * sizeof(Vec<S, L>));
    }

    void write_weights(
        Stream_Writer &writer
    ) const {
        writer.write(&predictor_weights[0], predictor_weights.size() * sizeof(Byte));
    }

    void read_weights(
        Stream_Reader &reader
    ) {
        reader.read(&predictor_weights[0], predictor_weights.size() * sizeof(Byte));
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
