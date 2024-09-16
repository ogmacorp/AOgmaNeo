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
        Array<Vec<S, L>> pos_vecs; // positional encodings
        Array<Vec<S, L>> visible_vecs; // input with bound position
        Array<Vec<S, L>> pred_vecs; // reconstructed input
    };

    struct Params {
        float scale;
        float lr;
        float leak;

        Params()
        :
        scale(8.0f),
        lr(0.05f),
        leak(0.01f)
        {}
    };

private:
    Int2 hidden_size; // size of hidden/output layer
    int D; // number of dendrites per cell in predictor

    Array<Vec<S, L>> hidden_pos_vecs; // positional encodings

    Array<Vec<S, L>> hidden_vecs;
    Array<Vec<S, L>> hidden_vecs_pred;
    Array<Vec<S, L>> hidden_vecs_prev;

    // visible layers and associated descriptors
    Array<Visible_Layer> visible_layers;
    Array<Visible_Layer_Desc> visible_layer_descs;

    Byte_Buffer predictor_weights;
    Float_Buffer predictor_dendrite_acts;
    Float_Buffer predictor_output_acts;
    Array<Predictor<S, L>> predictors;

    // --- kernels ---

    void forward(
        const Int2 &column_pos,
        const Array<Array_View<Vec<S, L>>> &input_vecs
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

                    Int2 offset(column_pos.x - visible_center.x + vld.radius, column_pos.y - visible_center.y + vld.radius);

                    int wi = offset.y + diam * (offset.x + diam * hidden_column_index);

                    sum += input_vecs[vli][visible_column_index] * vl.pos_vecs[wi];
                }
        }

        Vec<S, L> hidden_vec = sum.thin();

        hidden_vecs[hidden_column_index] = hidden_vec;
    }

    void predict(
        const Int2 &column_pos,
        Array_View<Vec<S, L>> feedback_vecs,
        bool learn_enabled,
        unsigned long* state,
        const Params &params
    ) {
        int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

        if (learn_enabled)
            predictors[hidden_column_index].learn(hidden_vecs_prev[hidden_column_index], hidden_vecs[hidden_column_index], state, params);

        Vec<S, L> pred_input_vec;

        if (feedback_vecs.size() == 0)
            pred_input_vec = hidden_vecs[hidden_column_index];
        else
            pred_input_vec = (hidden_vecs[hidden_column_index] + feedback_vecs[hidden_column_index]).thin();

        Vec<S, L> hidden_vec_pred = predictors[hidden_column_index].predict(pred_input_vec, params);

        hidden_vecs_pred[hidden_column_index] = hidden_vec_pred;

        hidden_vecs_prev[hidden_column_index] = pred_input_vec;
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

                if (in_bounds(column_pos, Int2(visible_center.x - vld.radius, visible_center.y - vld.radius), Int2(visible_center.x + vld.radius + 1, visible_center.y + vld.radius + 1))) {
                    Int2 offset(column_pos.x - visible_center.x + vld.radius, column_pos.y - visible_center.y + vld.radius);

                    int wi = offset.y + diam * (offset.x + diam * hidden_column_index);

                    sum += hidden_vecs_pred[hidden_column_index] / vl.pos_vecs[wi];
                }
            }

        // thin and unbind position
        vl.pred_vecs[visible_column_index] = sum.thin();
    }

public:
    // create a sparse coding layer with random initialization
    void init_random(
        const Int2 &hidden_size, // hidden/output size
        int D,
        const Array<Visible_Layer_Desc> &visible_layer_descs // descriptors for visible layers
    ) {
        this->visible_layer_descs = visible_layer_descs;

        this->hidden_size = hidden_size;
        this->D = D;

        visible_layers.resize(visible_layer_descs.size());

        // pre-compute dimensions
        int num_hidden_columns = hidden_size.x * hidden_size.y;

        const float L_inv = 1.0f / L;

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

            int num_visible_columns = vld.size.x * vld.size.y;

            int diam = vld.radius * 2 + 1;
            int area = diam * diam;

            vl.pos_vecs.resize(num_hidden_columns * area);

            for (int i = 0; i < vl.pos_vecs.size(); i++)
                vl.pos_vecs[i] = Vec<S, L>::randomized(); // "weights"

            vl.visible_vecs.resize(num_visible_columns);
            vl.pred_vecs = Array<Vec<S, L>>(num_visible_columns, 0);
        }

        hidden_vecs = Array<Vec<S, L>>(num_hidden_columns, 0);
        hidden_vecs_pred = Array<Vec<S, L>>(num_hidden_columns, 0);
        hidden_vecs_prev = Array<Vec<S, L>>(num_hidden_columns, 0);

        int C = Predictor<S, L>::N * D * Predictor<S, L>::N;

        predictor_weights.resize(num_hidden_columns * C);

        for (int i = 0; i < predictor_weights.size(); i++)
            predictor_weights[i] = 127 + (rand() % init_weight_noisei) - init_weight_noisei / 2;

        predictor_dendrite_acts.resize(num_hidden_columns * Predictor<S, L>::N * D);
        predictor_output_acts.resize(num_hidden_columns * Predictor<S, L>::N);

        predictors.resize(num_hidden_columns);

        for (int i = 0; i < num_hidden_columns; i++)
            predictors[i].set_from(D, &predictor_weights[i * C], &predictor_dendrite_acts[i * Predictor<S, L>::N * D], &predictor_output_acts[i * Predictor<S, L>::N]);
    }

    void forward(
        const Array<Array_View<Vec<S, L>>> &input_vecs
    ) {
        int num_hidden_columns = hidden_size.x * hidden_size.y;

        PARALLEL_FOR
        for (int i = 0; i < num_hidden_columns; i++)
            forward(Int2(i / hidden_size.y, i % hidden_size.y), input_vecs);
    }

    void predict(
        Array_View<Vec<S, L>> feedback_vecs, // can be empty
        bool learn_enabled, // whether to learn
        const Params &params // parameters
    ) {
        int num_hidden_columns = hidden_size.x * hidden_size.y;

        unsigned int base_state = rand();

        PARALLEL_FOR
        for (int i = 0; i < num_hidden_columns; i++) {
            unsigned long state = rand_get_state(base_state + i * rand_subseed_offset);

            predict(Int2(i / hidden_size.y, i % hidden_size.y), feedback_vecs, learn_enabled, &state, params);
        }
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
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];

            vl.pred_vecs.fill(0);
        }

        hidden_vecs.fill(0);
        hidden_vecs_pred.fill(0);
        hidden_vecs_prev.fill(0);
    }

    // serialization
    long size() const { // returns size in Bytes
        long size = sizeof(Int2) + 2 * sizeof(int);

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            const Visible_Layer &vl = visible_layers[vli];

            size += sizeof(Visible_Layer_Desc) + vl.pos_vecs.size() * sizeof(Vec<S, L>) + vl.pred_vecs.size() * sizeof(Vec<S, L>);
        }

        size += 3 * hidden_vecs.size() * sizeof(Vec<S, L>);

        size += predictor_weights.size() * sizeof(Byte);

        return size;
    }

    long state_size() const { // returns size of state in Bytes
        long size = 0;

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            const Visible_Layer &vl = visible_layers[vli];

            size += vl.pred_vecs.size() * sizeof(Vec<S, L>);
        }

        size += 3 * hidden_vecs.size() * sizeof(Vec<S, L>);

        return size;
    }

    long weights_size() const { // returns size of weights in Bytes
        return predictor_weights.size() * sizeof(Byte);
    }

    void write(
        Stream_Writer &writer
    ) const {
        writer.write(&hidden_size, sizeof(Int2));
        writer.write(&D, sizeof(int));

        int num_visible_layers = visible_layers.size();

        writer.write(&num_visible_layers, sizeof(int));
        
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            const Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];

            writer.write(&vld, sizeof(Visible_Layer_Desc));

            writer.write(&vl.pos_vecs[0], vl.pos_vecs.size() * sizeof(Vec<S, L>));

            writer.write(&vl.pred_vecs[0], vl.pred_vecs.size() * sizeof(Vec<S, L>));
        }

        writer.write(&hidden_pos_vecs[0], hidden_pos_vecs.size() * sizeof(Vec<S, L>));

        writer.write(&hidden_vecs[0], hidden_vecs.size() * sizeof(Vec<S, L>));
        writer.write(&hidden_vecs_pred[0], hidden_vecs_pred.size() * sizeof(Vec<S, L>));
        writer.write(&hidden_vecs_prev[0], hidden_vecs_prev.size() * sizeof(Vec<S, L>));

        writer.write(&predictor_weights[0], predictor_weights.size() * sizeof(Byte));
    }

    void read(
        Stream_Reader &reader
    ) {
        reader.read(&hidden_size, sizeof(Int2));
        reader.read(&D, sizeof(int));

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

            int diam = vld.radius * 2 + 1;
            int area = diam * diam;

            vl.pos_vecs.resize(num_hidden_columns * area);

            reader.read(&vl.pos_vecs[0], vl.pos_vecs.size() * sizeof(Vec<S, L>));
                                                       //
            vl.visible_vecs.resize(num_visible_columns);
            vl.pred_vecs.resize(num_visible_columns);

            reader.read(&vl.pred_vecs[0], vl.pred_vecs.size() * sizeof(Vec<S, L>));
        }

        hidden_vecs.resize(num_hidden_columns);
        hidden_vecs_pred.resize(num_hidden_columns);
        hidden_vecs_prev.resize(num_hidden_columns);

        reader.read(&hidden_vecs[0], hidden_vecs.size() * sizeof(Vec<S, L>));
        reader.read(&hidden_vecs_pred[0], hidden_vecs_pred.size() * sizeof(Vec<S, L>));
        reader.read(&hidden_vecs_prev[0], hidden_vecs_prev.size() * sizeof(Vec<S, L>));

        int C = Predictor<S, L>::N * D * Predictor<S, L>::N;

        predictor_weights.resize(num_hidden_columns * C);

        reader.read(&predictor_weights[0], predictor_weights.size() * sizeof(Byte));

        predictor_dendrite_acts.resize(num_hidden_columns * Predictor<S, L>::N * D);
        predictor_output_acts.resize(num_hidden_columns * Predictor<S, L>::N);

        predictors.resize(num_hidden_columns);

        for (int i = 0; i < num_hidden_columns; i++)
            predictors[i].set_from(D, &predictor_weights[i * C], &predictor_dendrite_acts[i * Predictor<S, L>::N * D], &predictor_output_acts[i * Predictor<S, L>::N]);
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
        writer.write(&hidden_vecs_prev[0], hidden_vecs_prev.size() * sizeof(Vec<S, L>));
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
        reader.read(&hidden_vecs_prev[0], hidden_vecs_prev.size() * sizeof(Vec<S, L>));
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
