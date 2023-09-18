// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "encoder.h"

using namespace aon;

void Encoder::forward(
    const Int2 &column_pos,
    const Array<const Int_Buffer*> &input_cis,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        hidden_matches[hidden_cell_index] = 0.0f;
    }

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

        int sub_count = (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1);

        float influence = vl.importance / (sub_count * 255);

        total_importance += vl.importance;

        const Int_Buffer &vl_input_cis = *input_cis[vli];

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci = vl_input_cis[visible_column_index];

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi_start = hidden_size.z * (offset.y + diam * (offset.x + diam * (in_ci + vld.size.z * hidden_column_index)));

                for (int hc = 0; hc < hidden_size.z; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int wi = hc + wi_start;

                    hidden_matches[hidden_cell_index] += vl.weights[wi] * influence;
                }
            }
    }

    int max_index = -1;
    float max_activation = 0.0f;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        hidden_matches[hidden_cell_index] /= max(limit_small, total_importance);

        float activation = hidden_matches[hidden_cell_index] / (params.choice + hidden_totals[hidden_cell_index]);

        hidden_acts[hidden_cell_index] = activation;

        if (activation > max_activation) {
            max_activation = activation;
            max_index = hc;
        }
    }

    hidden_cis[hidden_column_index] = max_index;
    hidden_max_acts[hidden_column_index] = max_activation;
}

void Encoder::update(
    const Int2 &column_pos,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    float max_activation = hidden_max_acts[hidden_column_index];

    for (int dcx = -params.l_radius; dcx <= params.l_radius; dcx++)
        for (int dcy = -params.l_radius; dcy <= params.l_radius; dcy++) {
            if (dcx == 0 && dcy == 0)
                continue;

            Int2 other_column_pos(column_pos.x + dcx, column_pos.y + dcy);

            if (in_bounds0(other_column_pos, Int2(hidden_size.x, hidden_size.y))) {
                int other_hidden_column_index = address2(other_column_pos, Int2(hidden_size.x, hidden_size.y));

                if (hidden_max_acts[other_hidden_column_index] >= max_activation)
                    return;
            }
        }

    // if made it here, this is a winner node
    int hidden_cell_index_max = hidden_cis[hidden_column_index] + hidden_cells_start;

    float max_match = hidden_matches[hidden_cell_index_max];

    if (max_match >= params.vigilance_lower) {
        // Check inner vigilance
        if (max_match >= params.vigilance_upper)
            // ready to learn
            learn_cis[hidden_column_index] = hidden_cis[hidden_column_index];
        else {
            // reset local
            hidden_acts[hidden_cell_index_max] = 0.0f;

            // Find new max
            int max_index = -1;
            float max_activation = 0.0f;

            for (int hc = 0; hc < hidden_size.z; hc++) {
                int hidden_cell_index = hc + hidden_cells_start;

                float activation = hidden_acts[hidden_cell_index];

                if (activation > max_activation) {
                    max_activation = activation;
                    max_index = hc;
                }
            }

            hidden_cis[hidden_column_index] = max_index;
            hidden_max_acts[hidden_column_index] = max_activation;
        }
    }
    else {
        // reset global
        for (int hc = 0; hc < hidden_size.z; hc++) {
            int hidden_cell_index = hc + hidden_cells_start;

            hidden_acts[hidden_cell_index] = 0.0f;
        }

        hidden_max_acts[hidden_column_index] = 0.0f;
    }
}

void Encoder::fallback(
    const Int2 &column_pos,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    // if couldn't resonate, just set to max match
    if (hidden_cis[hidden_column_index] == -1) {
        int max_index = 0;
        float max_match = 0.0f;

        for (int hc = 0; hc < hidden_size.z; hc++) {
            int hidden_cell_index = hc + hidden_cells_start;

            float match = hidden_matches[hidden_cell_index];

            if (match > max_match) {
                max_match = match;
                max_index = hc;
            }
        }

        hidden_cis[hidden_column_index] = max_index;
    }
}

void Encoder::learn(
    const Int2 &column_pos,
    const Array<const Int_Buffer*> &input_cis,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    int learn_ci = learn_cis[hidden_column_index];

    if (learn_ci == -1)
        return;

    int hidden_cell_index_max = learn_ci + hidden_cells_start;

    float rate = (hidden_totals[hidden_cell_index_max] == 1.0f ? 1.0f : params.lr);

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

        int sub_total = 0;
        int sub_count = (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1) * vld.size.z;

        const Int_Buffer &vl_input_cis = *input_cis[vli];

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci = vl_input_cis[visible_column_index];

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                for (int vc = 0; vc < vld.size.z; vc++) {
                    int wi = learn_ci + hidden_size.z * (offset.y + diam * (offset.x + diam * (vc + vld.size.z * hidden_column_index)));

                    if (vc != in_ci)
                        vl.weights[wi] = max(0, vl.weights[wi] - ceilf(rate * vl.weights[wi]));

                    sub_total += vl.weights[wi];
                }
            }

        total += static_cast<float>(sub_total) / (sub_count * 255) * vl.importance;
        total_importance += vl.importance;
    }

    total /= max(limit_small, total_importance);

    hidden_totals[hidden_cell_index_max] = total;
}

void Encoder::init_random(
    const Int3 &hidden_size,
    const Array<Visible_Layer_Desc> &visible_layer_descs
) {
    this->visible_layer_descs = visible_layer_descs;

    this->hidden_size = hidden_size;

    visible_layers.resize(visible_layer_descs.size());

    // pre-compute dimensions
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;

    // create layers
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(num_hidden_cells * area * vld.size.z);

        for (int i = 0; i < vl.weights.size(); i++)
            vl.weights[i] = 255 - (rand() % init_weight_noise);
    }

    hidden_cis = Int_Buffer(num_hidden_columns, 0);

    learn_cis.resize(num_hidden_columns);

    hidden_matches.resize(num_hidden_cells);
    hidden_acts.resize(num_hidden_cells);

    hidden_totals = Float_Buffer(num_hidden_cells, 1.0f);

    hidden_max_acts.resize(num_hidden_columns);
}

void Encoder::step(
    const Array<const Int_Buffer*> &input_cis,
    bool learn_enabled,
    const Params &params
) {
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    
    learn_cis.fill(-1);

    PARALLEL_FOR
    for (int i = 0; i < num_hidden_columns; i++)
        forward(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, params);

    int resets = 0;

    for (int it = 0; it < params.max_resets; it++) {
        PARALLEL_FOR
        for (int i = 0; i < num_hidden_columns; i++)
            update(Int2(i / hidden_size.y, i % hidden_size.y), params);

        // if all learns set, can early-out
        bool all_learn = true;

        for (int i = 0; i < learn_cis.size(); i++) {
            if (learn_cis[i] == -1) {
                all_learn = false;
                break;
            }
        }

        if (all_learn)
            break;

        resets++;
    }

    PARALLEL_FOR
    for (int i = 0; i < num_hidden_columns; i++)
        fallback(Int2(i / hidden_size.y, i % hidden_size.y), params);

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
    int size = sizeof(Int3) + hidden_cis.size() * sizeof(int) + hidden_totals.size() * sizeof(float) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += sizeof(Visible_Layer_Desc) + vl.weights.size() * sizeof(Byte) + sizeof(float);
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

    writer.write(reinterpret_cast<const void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));

    writer.write(reinterpret_cast<const void*>(&hidden_totals[0]), hidden_totals.size() * sizeof(float));

    int num_visible_layers = visible_layers.size();

    writer.write(reinterpret_cast<const void*>(&num_visible_layers), sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(Visible_Layer_Desc));

        writer.write(reinterpret_cast<const void*>(&vl.weights[0]), vl.weights.size() * sizeof(Byte));

        writer.write(reinterpret_cast<const void*>(&vl.importance), sizeof(float));
    }
}

void Encoder::read(
    Stream_Reader &reader
) {
    reader.read(reinterpret_cast<void*>(&hidden_size), sizeof(Int3));

    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;

    hidden_cis.resize(num_hidden_columns);

    reader.read(reinterpret_cast<void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));

    learn_cis.resize(num_hidden_columns);

    hidden_matches.resize(num_hidden_cells);
    hidden_acts.resize(num_hidden_cells);

    hidden_totals.resize(num_hidden_cells);

    reader.read(reinterpret_cast<void*>(&hidden_totals[0]), hidden_totals.size() * sizeof(float));

    hidden_max_acts.resize(num_hidden_columns);

    int num_visible_layers = visible_layers.size();

    reader.read(reinterpret_cast<void*>(&num_visible_layers), sizeof(int));

    visible_layers.resize(num_visible_layers);
    visible_layer_descs.resize(num_visible_layers);
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        Visible_Layer_Desc &vld = visible_layer_descs[vli];

        reader.read(reinterpret_cast<void*>(&vld), sizeof(Visible_Layer_Desc));

        int num_visible_columns = vld.size.x * vld.size.y;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(num_hidden_cells * area * vld.size.z);

        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(Byte));

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
