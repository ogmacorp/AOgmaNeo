// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2025 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "encoder.h"

using namespace aon;

void Encoder::forward_spatial(
    const Int2 &column_pos,
    const Array<Int_Buffer_View> &input_cis,
    unsigned long* state,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

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

        total_importance += vl.importance;

        Int_Buffer_View vl_input_cis = input_cis[vli];

        for (int hc = 0; hc < hidden_size.z; hc++) {
            int hidden_cell_index = hc + hidden_cells_start;

            vl.hidden_sums[hidden_cell_index] = 0.0f;
            vl.hidden_totals[hidden_cell_index] = 0.0f;
        }

        const float vld_size_z_inv = 1.0f / vld.size.z;

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci = vl_input_cis[visible_column_index];

                float in_value = (in_ci + 0.5f) * vld_size_z_inv;

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi_start = hidden_size.z * (offset.y + diam * (offset.x + diam * hidden_column_index));

                for (int hc = 0; hc < hidden_size.z; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int wi = hc + wi_start;

                    vl.hidden_sums[hidden_cell_index] += min(vl.weights0[wi], in_value) + min(vl.weights1[wi], 1.0f - in_value);
                    vl.hidden_totals[hidden_cell_index] += vl.weights0[wi] + vl.weights1[wi];
                }
            }
    }

    int max_index = -1;
    float max_activation = 0.0f;

    int max_complete_index = 0;
    float max_complete_activation = 0.0f;
    
    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        float sum = 0.0f;
        float total = 0.0f;

        bool all_match = true;

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];

            int sub_count = vl.hidden_counts[hidden_column_index];

            float sub_sum = vl.hidden_sums[hidden_cell_index];
            float sub_total = vl.hidden_totals[hidden_cell_index];

            float match = sub_sum / sub_count;

            float vigilance = 1.0f - params.spatial_mismatch / vld.size.z;

            if (vl.importance > 0.0f && match < vigilance)
                all_match = false;

            sum += sub_sum * vl.importance;
            total += sub_total * vl.importance;
        }

        sum /= max(limit_small, total_importance);
        total /= max(limit_small, total_importance);

        float activation = sum / (params.choice + total);

        hidden_learn_flags[hidden_cell_index] = all_match;

        if (all_match && activation > max_activation) {
            max_activation = activation;
            max_index = hc;
        }

        if (activation > max_complete_activation) {
            max_complete_activation = activation;
            max_complete_index = hc;
        }
    }

    hidden_comparisons[hidden_column_index] = max_activation * powf(randf(state), params.temperature);

    hidden_cis[hidden_column_index] = (max_index == -1 ? max_complete_index : max_index);
}

void Encoder::forward_recurrent(
    const Int2 &column_pos,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int full_column_size = hidden_size.z * temporal_size;

    int temporal_cells_start = hidden_column_index * temporal_size;

    int hidden_ci = hidden_cis[hidden_column_index];

    for (int tc = 0; tc < temporal_size; tc++) {
        int temporal_cell_index = tc + temporal_cells_start;

        recurrent_sums[temporal_cell_index] = 0.0f;
        recurrent_totals[temporal_cell_index] = 0.0f;
    }

    int diam = recurrent_radius * 2 + 1;

    // lower corner
    Int2 field_lower_bound(column_pos.x - recurrent_radius, column_pos.y - recurrent_radius);

    // bounds of receptive field, clamped to input size
    Int2 iter_lower_bound(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
    Int2 iter_upper_bound(min(hidden_size.x - 1, column_pos.x + recurrent_radius), min(hidden_size.y - 1, column_pos.y + recurrent_radius));

    int count = (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1);

    const float full_column_size_inv = 1.0f / full_column_size;

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            int other_hidden_column_index = address2(Int2(ix, iy), Int2(hidden_size.x, hidden_size.y));

            int in_ci = temporal_cis_prev[other_hidden_column_index];

            float in_value = (in_ci + 0.5f) * full_column_size_inv;

            Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

            int wi_offset = in_ci + full_column_size * (offset.y + diam * offset.x);

            for (int tc = 0; tc < temporal_size; tc++) {
                int temporal_cell_index = tc + temporal_cells_start;

                int full_ci = tc + hidden_ci * temporal_size;

                int wi = full_ci + full_column_size * (offset.y + diam * (offset.x + diam * hidden_column_index));

                recurrent_sums[temporal_cell_index] += min(recurrent_weights0[wi], in_value) + min(recurrent_weights1[wi], 1.0f - in_value);
                recurrent_totals[temporal_cell_index] += recurrent_weights0[wi] + recurrent_weights1[wi];
            }
        }

    int max_index = -1;
    float max_activation = 0.0f;

    int max_complete_index = 0;
    float max_complete_activation = 0.0f;
    
    for (int tc = 0; tc < temporal_size; tc++) {
        int temporal_cell_index = tc + temporal_cells_start;

        float sum = recurrent_sums[temporal_cell_index];
        float total = recurrent_totals[temporal_cell_index];

        float match = sum / count;

        float activation = sum / (params.choice + total);

        float vigilance = 1.0f - params.temporal_mismatch / full_column_size;

        temporal_learn_flags[temporal_cell_index] = (match >= vigilance);

        if (match >= vigilance && activation > max_activation) {
            max_activation = activation;
            max_index = tc;
        }

        if (activation > max_complete_activation) {
            max_complete_activation = activation;
            max_complete_index = tc;
        }
    }

    temporal_cis[hidden_column_index] = (max_index == -1 ? max_complete_index : max_index) + hidden_ci * temporal_size;
}

void Encoder::learn(
    const Int2 &column_pos,
    const Array<Int_Buffer_View> &input_cis,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int full_column_size = hidden_size.z * temporal_size;

    int hidden_cells_start = hidden_column_index * hidden_size.z;
    int temporal_cells_start = hidden_column_index * temporal_size;
    int full_cells_start = hidden_column_index * full_column_size;

    float hidden_max = hidden_comparisons[hidden_column_index];

    int num_higher = 0;
    int count = 1; // start at 1 since self is skipped

    for (int dcx = -params.l_radius; dcx <= params.l_radius; dcx++)
        for (int dcy = -params.l_radius; dcy <= params.l_radius; dcy++) {
            if (dcx == 0 && dcy == 0)
                continue;

            Int2 other_column_pos(column_pos.x + dcx, column_pos.y + dcy);

            if (in_bounds0(other_column_pos, Int2(hidden_size.x, hidden_size.y))) {
                int other_hidden_column_index = address2(other_column_pos, Int2(hidden_size.x, hidden_size.y));

                if (hidden_comparisons[other_hidden_column_index] >= hidden_max)
                    num_higher++;

                count++;
            }
        }

    float ratio = static_cast<float>(num_higher) / static_cast<float>(count);

    int hidden_ci = hidden_cis[hidden_column_index];

    if (ratio <= params.active_ratio) {
        for (int dhc = -params.n_radius; dhc <= params.n_radius; dhc++) {
            int hc = hidden_ci + dhc;

            if (hc < 0 || hc >= hidden_size.z)
                continue;

            // spatial
            int hidden_cell_index = hc + hidden_cells_start;

            if (!hidden_learn_flags[hidden_cell_index])
                continue;

            float rate = (hidden_commits[hidden_cell_index] ? params.lr : 1.0f) * powf(params.falloff, abs(dhc));

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

                Int_Buffer_View vl_input_cis = input_cis[vli];

                const float vld_size_z_inv = 1.0f / vld.size.z;

                for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                    for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                        int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                        int in_ci = vl_input_cis[visible_column_index];

                        float in_value = (in_ci + 0.5f) * vld_size_z_inv;

                        Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                        int wi = hc + hidden_size.z * (offset.y + diam * (offset.x + diam * hidden_column_index));

                        vl.weights0[wi] += rate * min(0.0f, in_value - vl.weights0[wi]);
                        vl.weights1[wi] += rate * min(0.0f, 1.0f - in_value - vl.weights1[wi]);
                    }
            }

            hidden_commits[hidden_cell_index] = true;
        }
    }

    const float full_column_size_inv = 1.0f / full_column_size;

    int temporal_ci = temporal_cis[hidden_column_index];

    // recurrent
    int diam = recurrent_radius * 2 + 1;

    // lower corner
    Int2 field_lower_bound(column_pos.x - recurrent_radius, column_pos.y - recurrent_radius);

    // bounds of receptive field, clamped to input size
    Int2 iter_lower_bound(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
    Int2 iter_upper_bound(min(hidden_size.x - 1, column_pos.x + recurrent_radius), min(hidden_size.y - 1, column_pos.y + recurrent_radius));

    int temporal_sub_ci = temporal_ci % temporal_size;

    for (int dtc = -params.n_radius; dtc <= params.n_radius; dtc++) {
        int tc = temporal_sub_ci + dtc;

        if (tc < 0 || tc >= temporal_size)
            continue;

        int temporal_cell_index = tc + temporal_cells_start;

        if (!temporal_learn_flags[temporal_cell_index])
            continue;

        int full_ci = tc + hidden_ci * temporal_size;

        int full_cell_index = full_ci + full_cells_start;

        float rate = (temporal_commits[full_cell_index] ? params.lr : 1.0f);

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int other_hidden_column_index = address2(Int2(ix, iy), Int2(hidden_size.x, hidden_size.y));

                int in_ci = temporal_cis_prev[other_hidden_column_index];

                float in_value = (in_ci + 0.5f) * full_column_size_inv;

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi = full_ci + full_column_size * (offset.y + diam * (offset.x + diam * hidden_column_index));

                recurrent_weights0[wi] += rate * min(0.0f, in_value - recurrent_weights0[wi]);
                recurrent_weights1[wi] += rate * min(0.0f, 1.0f - in_value - recurrent_weights1[wi]);
            }

        temporal_commits[full_cell_index] = true;
    }
}

void Encoder::init_random(
    const Int3 &hidden_size,
    int temporal_size,
    int recurrent_radius,
    const Array<Visible_Layer_Desc> &visible_layer_descs
) {
    this->visible_layer_descs = visible_layer_descs;

    this->hidden_size = hidden_size;
    this->temporal_size = temporal_size;
    this->recurrent_radius = recurrent_radius;

    visible_layers.resize(visible_layer_descs.size());

    int full_column_size = hidden_size.z * temporal_size;

    // pre-compute dimensions
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;
    int num_temporal_cells = num_hidden_columns * temporal_size;
    int num_full_cells = num_hidden_columns * full_column_size;

    // create layers
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights0.resize(num_hidden_cells * area);
        vl.weights1.resize(vl.weights0.size());

        for (int i = 0; i < vl.weights0.size(); i++) {
            vl.weights0[i] = 1.0f - randf() * init_weight_noisef;
            vl.weights1[i] = 1.0f - randf() * init_weight_noisef;
        }

        vl.hidden_sums.resize(num_hidden_cells);
        vl.hidden_totals.resize(num_hidden_cells);
        vl.hidden_counts.resize(num_hidden_columns);
    }

    hidden_cis = Int_Buffer(num_hidden_columns, 0);
    temporal_cis = Int_Buffer(num_hidden_columns, 0);
    temporal_cis_prev.resize(num_hidden_columns);

    hidden_learn_flags.resize(num_hidden_cells);
    temporal_learn_flags.resize(num_temporal_cells);

    hidden_commits = Byte_Buffer(num_hidden_cells, false);
    temporal_commits = Byte_Buffer(num_full_cells, false);

    hidden_comparisons.resize(num_hidden_columns);

    int diam = recurrent_radius * 2 + 1;
    int area = diam * diam;

    recurrent_weights0.resize(num_full_cells * area);
    recurrent_weights1.resize(recurrent_weights0.size());

    for (int i = 0; i < recurrent_weights0.size(); i++) {
        recurrent_weights0[i] = 1.0f - randf() * init_weight_noisef;
        recurrent_weights1[i] = 1.0f - randf() * init_weight_noisef;
    }

    recurrent_sums.resize(num_temporal_cells);
    recurrent_totals.resize(num_temporal_cells);

    // init totals and counts
    for (int i = 0; i < num_hidden_columns; i++) {
        Int2 column_pos(i / hidden_size.y, i % hidden_size.y);

        int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

        int hidden_cells_start = hidden_column_index * hidden_size.z;
        int full_cells_start = hidden_column_index * full_column_size;

        // spatial
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

            vl.hidden_counts[hidden_column_index] = sub_count;
        }
    }
}

void Encoder::step(
    const Array<Int_Buffer_View> &input_cis,
    bool learn_enabled,
    const Params &params
) {
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    
    temporal_cis_prev = temporal_cis;

    unsigned int base_state = rand();

    PARALLEL_FOR
    for (int i = 0; i < num_hidden_columns; i++) {
        unsigned long state = rand_get_state(base_state + i * rand_subseed_offset);

        forward_spatial(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, &state, params);
    }

    PARALLEL_FOR
    for (int i = 0; i < num_hidden_columns; i++)
        forward_recurrent(Int2(i / hidden_size.y, i % hidden_size.y), params);

    if (learn_enabled) {
        PARALLEL_FOR
        for (int i = 0; i < num_hidden_columns; i++)
            learn(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, params);
    }
}

void Encoder::clear_state() {
    hidden_cis.fill(0);
    temporal_cis.fill(0);
}

long Encoder::size() const {
    long size = sizeof(Int3) + 2 * sizeof(int) + 2 * hidden_cis.size() * sizeof(int) + hidden_commits.size() * sizeof(Byte) + temporal_commits.size() * sizeof(Byte) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += sizeof(Visible_Layer_Desc) + 2 * vl.weights0.size() * sizeof(float) + vl.hidden_counts.size() * sizeof(int) + sizeof(float);
    }

    size += 2 * recurrent_weights0.size() * sizeof(float);

    return size;
}

long Encoder::state_size() const {
    return 2 * hidden_cis.size() * sizeof(int);
}

long Encoder::weights_size() const {
    long size = 0;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += 2 * vl.weights0.size() * sizeof(float);
    }

    size += 2 * recurrent_weights0.size() * sizeof(float);

    return size;
}

void Encoder::write(
    Stream_Writer &writer
) const {
    writer.write(&hidden_size, sizeof(Int3));
    writer.write(&temporal_size, sizeof(int));
    writer.write(&recurrent_radius, sizeof(int));

    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    writer.write(&temporal_cis[0], temporal_cis.size() * sizeof(int));

    writer.write(&hidden_commits[0], hidden_commits.size() * sizeof(Byte));
    writer.write(&temporal_commits[0], temporal_commits.size() * sizeof(Byte));

    int num_visible_layers = visible_layers.size();

    writer.write(&num_visible_layers, sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(&vld, sizeof(Visible_Layer_Desc));

        writer.write(&vl.weights0[0], vl.weights0.size() * sizeof(float));
        writer.write(&vl.weights1[0], vl.weights1.size() * sizeof(float));

        writer.write(&vl.hidden_counts[0], vl.hidden_counts.size() * sizeof(int));

        writer.write(&vl.importance, sizeof(float));
    }

    writer.write(&recurrent_weights0[0], recurrent_weights0.size() * sizeof(float));
    writer.write(&recurrent_weights1[0], recurrent_weights1.size() * sizeof(float));
}

void Encoder::read(
    Stream_Reader &reader
) {
    reader.read(&hidden_size, sizeof(Int3));
    reader.read(&temporal_size, sizeof(int));
    reader.read(&recurrent_radius, sizeof(int));

    int full_column_size = hidden_size.z * temporal_size;

    // pre-compute dimensions
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;
    int num_temporal_cells = num_hidden_columns * temporal_size;
    int num_full_cells = num_hidden_columns * full_column_size;

    hidden_cis.resize(num_hidden_columns);
    temporal_cis.resize(num_hidden_columns);

    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    reader.read(&temporal_cis[0], temporal_cis.size() * sizeof(int));

    hidden_learn_flags.resize(num_hidden_cells);
    temporal_learn_flags.resize(num_temporal_cells);

    hidden_commits.resize(num_hidden_cells);
    temporal_commits.resize(num_full_cells);

    reader.read(&hidden_commits[0], hidden_commits.size() * sizeof(Byte));
    reader.read(&temporal_commits[0], temporal_commits.size() * sizeof(Byte));

    hidden_comparisons.resize(num_hidden_columns);

    int num_visible_layers = visible_layers.size();

    reader.read(&num_visible_layers, sizeof(int));

    visible_layers.resize(num_visible_layers);
    visible_layer_descs.resize(num_visible_layers);
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        Visible_Layer_Desc &vld = visible_layer_descs[vli];

        reader.read(&vld, sizeof(Visible_Layer_Desc));

        int num_visible_columns = vld.size.x * vld.size.y;
        int num_visible_cells = num_visible_columns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights0.resize(num_hidden_cells * area);
        vl.weights1.resize(vl.weights0.size());

        reader.read(&vl.weights0[0], vl.weights0.size() * sizeof(float));
        reader.read(&vl.weights1[0], vl.weights1.size() * sizeof(float));

        vl.hidden_sums.resize(num_hidden_cells);
        vl.hidden_totals.resize(num_hidden_cells);

        vl.hidden_counts.resize(num_hidden_columns);

        reader.read(&vl.hidden_counts[0], vl.hidden_counts.size() * sizeof(int));

        reader.read(&vl.importance, sizeof(float));
    }

    int diam = recurrent_radius * 2 + 1;
    int area = diam * diam;

    recurrent_weights0.resize(num_full_cells * area);
    recurrent_weights1.resize(recurrent_weights0.size());

    reader.read(&recurrent_weights0[0], recurrent_weights0.size() * sizeof(float));
    reader.read(&recurrent_weights1[0], recurrent_weights1.size() * sizeof(float));

    recurrent_sums.resize(num_temporal_cells);
    recurrent_totals.resize(num_full_cells);
}

void Encoder::write_state(
    Stream_Writer &writer
) const {
    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    writer.write(&temporal_cis[0], temporal_cis.size() * sizeof(int));
}

void Encoder::read_state(
    Stream_Reader &reader
) {
    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    reader.read(&temporal_cis[0], temporal_cis.size() * sizeof(int));
}

void Encoder::write_weights(
    Stream_Writer &writer
) const {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        writer.write(&vl.weights0[0], vl.weights0.size() * sizeof(float));
        writer.write(&vl.weights1[0], vl.weights1.size() * sizeof(float));
    }

    writer.write(&recurrent_weights0[0], recurrent_weights0.size() * sizeof(float));
    writer.write(&recurrent_weights1[0], recurrent_weights1.size() * sizeof(float));
}

void Encoder::read_weights(
    Stream_Reader &reader
) {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(&vl.weights0[0], vl.weights0.size() * sizeof(float));
        reader.read(&vl.weights1[0], vl.weights1.size() * sizeof(float));
    }

    reader.read(&recurrent_weights0[0], recurrent_weights0.size() * sizeof(float));
    reader.read(&recurrent_weights1[0], recurrent_weights1.size() * sizeof(float));
}

void Encoder::merge(
    const Array<Encoder*> &encoders,
    Merge_Mode mode
) {
    switch (mode) {
    case merge_random:
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];
        
            for (int i = 0; i < vl.weights0.size(); i++) {
                int e = rand() % encoders.size();                

                vl.weights0[i] = encoders[e]->visible_layers[vli].weights0[i];
                vl.weights1[i] = encoders[e]->visible_layers[vli].weights1[i];
            }
        }

        for (int i = 0; i < recurrent_weights0.size(); i++) {
            int e = rand() % encoders.size();                

            recurrent_weights0[i] = encoders[e]->recurrent_weights0[i];
            recurrent_weights1[i] = encoders[e]->recurrent_weights1[i];
        }

        break;
    case merge_average:
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];
        
            for (int i = 0; i < vl.weights0.size(); i++) {
                float total0 = 0.0f;
                float total1 = 0.0f;

                for (int e = 0; e < encoders.size(); e++) {
                    total0 += encoders[e]->visible_layers[vli].weights0[i];
                    total1 += encoders[e]->visible_layers[vli].weights1[i];
                }

                vl.weights0[i] = roundf(total0 / encoders.size());
                vl.weights1[i] = roundf(total1 / encoders.size());
            }
        }

        for (int i = 0; i < recurrent_weights0.size(); i++) {
            float total0 = 0.0f;
            float total1 = 0.0f;

            for (int e = 0; e < encoders.size(); e++) {
                total0 += encoders[e]->recurrent_weights0[i];
                total1 += encoders[e]->recurrent_weights1[i];
            }

            recurrent_weights0[i] = total0 / encoders.size();
            recurrent_weights1[i] = total1 / encoders.size();
        }

        break;
    }
}
