// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "encoder.h"
#include <iostream>

using namespace aon;

void Encoder::forward(
    const Int2 &column_pos,
    const Array<const Int_Buffer*> &input_cis,
    unsigned int* state,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    for (int hc = 0; hc < hidden_commits[hidden_column_index]; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        hidden_acts[hidden_cell_index] = 0.0f;
    }

    int max_index = -1;
    float max_activation = 0.0f;

    float total_importance = 0.0f;

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

        float sub_sum = 0.0f;
        int sub_count = (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1);

        float scale = vl.importance / sub_count;

        int hidden_stride = diam * diam * vld.size.z;

        total_importance += vl.importance;

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci = (*input_cis[vli])[visible_column_index];

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi_offset = in_ci + vld.size.z * (offset.y + diam * offset.x);

                for (int hc = 0; hc < hidden_commits[hidden_column_index]; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int wi = wi_offset + hidden_cell_index * hidden_stride;

                    hidden_acts[hidden_cell_index] += vl.weights[wi] * scale;
                }
            }
    }

    for (int hc = 0; hc < hidden_commits[hidden_column_index]; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        hidden_acts[hidden_cell_index] /= max(limit_small, total_importance);

        if (hidden_acts[hidden_cell_index] > max_activation) {
            max_activation = hidden_acts[hidden_cell_index];
            max_index = hc;
        }
    }

    if (max_index == -1 || max_activation < params.vigilance) {
        if (randf(state) < params.alloc_prob && hidden_commits[hidden_column_index] < hidden_size.z) {
            learn_cis[hidden_column_index] = hidden_commits[hidden_column_index];

            hidden_maxs[hidden_column_index] = 1.0f + randf(state) * limit_small;
        }
        else {
            learn_cis[hidden_column_index] = max_index;

            hidden_maxs[hidden_column_index] = max_activation + randf(state) * limit_small;
        }
    }
    else {
        learn_cis[hidden_column_index] = -1;

        hidden_maxs[hidden_column_index] = 0.0f;
    }

    hidden_cis[hidden_column_index] = max(0, max_index);
}

void Encoder::learn(
    const Int2 &column_pos,
    const Array<const Int_Buffer*> &input_cis,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    if (learn_cis[hidden_column_index] == -1)
        return;

    float max_activation = hidden_maxs[hidden_column_index];

    for (int dcx = -params.l_radius; dcx <= params.l_radius; dcx++)
        for (int dcy = -params.l_radius; dcy <= params.l_radius; dcy++) {
            if (dcx == 0 && dcy == 0)
                continue;

            Int2 other_column_pos(column_pos.x + dcx, column_pos.y + dcy);

            if (in_bounds0(other_column_pos, Int2(hidden_size.x, hidden_size.y))) {
                int other_hidden_column_index = address2(other_column_pos, Int2(hidden_size.x, hidden_size.y));

                if (hidden_maxs[other_hidden_column_index] >= max_activation)
                    return;
            }
        }

    bool commit = (learn_cis[hidden_column_index] == hidden_commits[hidden_column_index]);

    int hidden_cell_index_max = learn_cis[hidden_column_index] + hidden_cells_start;

    float rate = (commit ? 1.0f : params.lr);

    float total = 0.0f;
    float total_importance = 0.0f;

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

        int sub_count = (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1) * vld.size.z;

        float sub_total = 0.0f;

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci = (*input_cis[vli])[visible_column_index];

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi_start = vld.size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index_max));

                vl.weights[in_ci + wi_start] += rate * (1.0f - vl.weights[in_ci + wi_start]);

                for (int vc = 0; vc < vld.size.z; vc++) {
                    int wi = vc + wi_start;

                    sub_total += vl.weights[wi];
                }
            }

        total += sub_total * vl.importance;
        total_importance += vl.importance;
    }

    total /= max(limit_small, total_importance);

    hidden_totals[hidden_cell_index_max] = total;

    if (commit)
        hidden_commits[hidden_column_index]++;
}

void Encoder::init_random(
    const Int3 &hidden_size,
    const Array<Visible_Layer_Desc> &visible_layer_descs
) {
    this->visible_layer_descs = visible_layer_descs;

    this->hidden_size = hidden_size;

    base_state = rand();

    visible_layers.resize(visible_layer_descs.size());

    // pre-compute dimensions
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;

    // create layers
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;
        int num_visible_cells = num_visible_columns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights = Float_Buffer(num_hidden_cells * area * vld.size.z, 0.0f);
    }

    hidden_cis = Int_Buffer(num_hidden_columns, 0);

    learn_cis.resize(num_hidden_columns);

    hidden_commits = Int_Buffer(num_hidden_columns, 0);

    hidden_acts.resize(num_hidden_cells);

    hidden_totals = Float_Buffer(num_hidden_cells, 0.0f);

    hidden_maxs.resize(num_hidden_columns);
}

void Encoder::step(
    const Array<const Int_Buffer*> &input_cis,
    bool learn_enabled,
    const Params &params
) {
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    
    unsigned int base_state = rand();

    PARALLEL_FOR
    for (int i = 0; i < num_hidden_columns; i++) {
        unsigned int state = base_state + i * 12345;

        forward(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, &state, params);
    }

    if (learn_enabled) {
        PARALLEL_FOR
        for (int i = 0; i < num_hidden_columns; i++)
            learn(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, params);
    }
}

void Encoder::clear_state() {
    hidden_cis.fill(0);
}

int Encoder::size() const {
    int size = sizeof(Int3) + sizeof(unsigned int) + hidden_cis.size() * sizeof(int) + hidden_commits.size() * sizeof(int) + hidden_totals.size() * sizeof(float) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += sizeof(Visible_Layer_Desc) + vl.weights.size() * sizeof(float) + sizeof(float);
    }

    return size;
}

int Encoder::state_size() const {
    return hidden_cis.size() * sizeof(int);
}

void Encoder::write(
    Stream_Writer &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hidden_size), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&base_state), sizeof(unsigned int));

    writer.write(reinterpret_cast<const void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));

    writer.write(reinterpret_cast<const void*>(&hidden_commits[0]), hidden_commits.size() * sizeof(int));

    writer.write(reinterpret_cast<const void*>(&hidden_totals[0]), hidden_totals.size() * sizeof(float));

    int num_visible_layers = visible_layers.size();

    writer.write(reinterpret_cast<const void*>(&num_visible_layers), sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(Visible_Layer_Desc));

        writer.write(reinterpret_cast<const void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));

        writer.write(reinterpret_cast<const void*>(&vl.importance), sizeof(float));
    }
}

void Encoder::read(
    Stream_Reader &reader
) {
    reader.read(reinterpret_cast<void*>(&hidden_size), sizeof(Int3));

    reader.read(reinterpret_cast<void*>(&base_state), sizeof(unsigned int));

    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;

    hidden_cis.resize(num_hidden_columns);

    reader.read(reinterpret_cast<void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));

    learn_cis.resize(num_hidden_columns);

    hidden_commits.resize(num_hidden_columns);

    reader.read(reinterpret_cast<void*>(&hidden_commits[0]), hidden_commits.size() * sizeof(int));

    hidden_acts.resize(num_hidden_cells);

    hidden_totals.resize(num_hidden_cells);

    reader.read(reinterpret_cast<void*>(&hidden_totals[0]), hidden_totals.size() * sizeof(float));

    hidden_maxs.resize(num_hidden_columns);

    int num_visible_layers = visible_layers.size();

    reader.read(reinterpret_cast<void*>(&num_visible_layers), sizeof(int));

    visible_layers.resize(num_visible_layers);
    visible_layer_descs.resize(num_visible_layers);
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        Visible_Layer_Desc &vld = visible_layer_descs[vli];

        reader.read(reinterpret_cast<void*>(&vld), sizeof(Visible_Layer_Desc));

        int num_visible_columns = vld.size.x * vld.size.y;
        int num_visible_cells = num_visible_columns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(num_hidden_cells * area * vld.size.z);

        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(float));

        reader.read(reinterpret_cast<void*>(&vl.importance), sizeof(float));
    }
}

void Encoder::write_state(
    Stream_Writer &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));
}

void Encoder::read_state(
    Stream_Reader &reader
) {
    reader.read(reinterpret_cast<void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));
}
