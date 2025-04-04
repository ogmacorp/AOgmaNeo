// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2025 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "encoder.h"

using namespace aon;

void Encoder::forward(
    const Int2 &column_pos,
    const Array<Int_Buffer_View> &input_cis,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    // spatial
    {
        float count = 0.0f;
        float count_except = 0.0f;
        float count_all = 0.0f;
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

            int sub_count = vl.hidden_counts[hidden_column_index];

            count += vl.importance * sub_count;
            count_except += vl.importance * sub_count * (vld.size.z - 1);
            count_all += vl.importance * sub_count * vld.size.z;

            total_importance += vl.importance;

            Int_Buffer_View vl_input_cis = input_cis[vli];

            for (int hc = 0; hc < hidden_size.z; hc++) {
                int hidden_cell_index = hc + hidden_cells_start;

                vl.hidden_sums[hidden_cell_index] = 0;
            }

            for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                    int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                    int in_ci = vl_input_cis[visible_column_index];

                    Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                    int wi_start = hidden_size.z * (offset.y + diam * (offset.x + diam * (in_ci + vld.size.z * hidden_column_index)));

                    for (int hc = 0; hc < hidden_size.z; hc++) {
                        int hidden_cell_index = hc + hidden_cells_start;

                        int wi = hc + wi_start;

                        vl.hidden_sums[hidden_cell_index] += vl.weights[wi];
                    }
                }
        }

        count /= max(limit_small, total_importance);
        count_except /= max(limit_small, total_importance);
        count_all /= max(limit_small, total_importance);

        int max_index = -1;
        float max_activation = 0.0f;

        int max_complete_index = 0;
        float max_complete_activation = 0.0f;

        float max_match = 0.0f;
        
        const float byte_inv = 1.0f / 255.0f;

        for (int hc = 0; hc < hidden_size.z; hc++) {
            int hidden_cell_index = hc + hidden_cells_start;

            float sum = 0.0f;
            float total = 0.0f;

            for (int vli = 0; vli < visible_layers.size(); vli++) {
                Visible_Layer &vl = visible_layers[vli];
                const Visible_Layer_Desc &vld = visible_layer_descs[vli];

                float influence = vl.importance * byte_inv;

                sum += vl.hidden_sums[hidden_cell_index] * influence;
                total += vl.hidden_totals[hidden_cell_index] * influence;
            }

            sum /= max(limit_small, total_importance);
            total /= max(limit_small, total_importance);

            float complemented = (count_all - total) - (count - sum);

            float match = complemented / count_except;

            float activation = complemented / (params.choice + count_all - total);

            if (match >= params.category_vigilance && activation > max_activation) {
                max_activation = activation;
                max_index = hc;
            }

            if (activation > max_complete_activation) {
                max_complete_activation = activation;
                max_complete_index = hc;
            }

            max_match = max(max_match, match);
        }

        hidden_comparisons[hidden_column_index] = (max_match >= params.compare_vigilance ? max_activation : 0.0f);

        hidden_cis[hidden_column_index] = (max_index == -1 ? max_complete_index : max_index);

        hidden_learn_flags[hidden_column_index] = (max_index != -1);
    }

    // temporal
    {
        int full_column_size = hidden_size.z * temporal_size;

        int temporal_cells_start = hidden_column_index * temporal_size;
        int full_cells_start = hidden_column_index * full_column_size;

        int hidden_ci = hidden_cis[hidden_column_index];

        for (int tc = 0; tc < temporal_size; tc++) {
            int temporal_cell_index = tc + temporal_cells_start;

            recurrent_sums[temporal_cell_index] = 0;
        }

        int diam = recurrent_radius * 2 + 1;

        // lower corner
        Int2 field_lower_bound(column_pos.x - recurrent_radius, column_pos.y - recurrent_radius);

        // bounds of receptive field, clamped to input size
        Int2 iter_lower_bound(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
        Int2 iter_upper_bound(min(hidden_size.x - 1, column_pos.x + recurrent_radius), min(hidden_size.y - 1, column_pos.y + recurrent_radius));

        int count = (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1);
        int count_except = count * (full_column_size - 1);
        int count_all = count * full_column_size;

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int other_hidden_column_index = address2(Int2(ix, iy), Int2(hidden_size.x, hidden_size.y));

                int in_ci = temporal_cis_prev[other_hidden_column_index];

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi_offset = in_ci + full_column_size * (offset.y + diam * offset.x);

                for (int tc = 0; tc < temporal_size; tc++) {
                    int temporal_cell_index = tc + temporal_cells_start;

                    int full_ci = tc + hidden_ci * temporal_size;

                    int wi = full_ci + full_column_size * (offset.y + diam * (offset.x + diam * (in_ci + full_column_size * hidden_column_index)));

                    recurrent_sums[temporal_cell_index] += recurrent_weights[wi];
                }
            }

        int max_index = -1;
        float max_activation = 0.0f;

        int max_complete_index = 0;
        float max_complete_activation = 0.0f;
        
        const float byte_inv = 1.0f / 255.0f;

        for (int tc = 0; tc < temporal_size; tc++) {
            int temporal_cell_index = tc + temporal_cells_start;

            int full_cell_index = tc + hidden_ci * temporal_size + full_cells_start;

            float complemented = (count_all - recurrent_totals[full_cell_index] * byte_inv) - (count - recurrent_sums[temporal_cell_index] * byte_inv);

            float match = complemented / count_except;

            float activation = complemented / (params.choice + count_all - recurrent_totals[full_cell_index] * byte_inv);

            if (match >= params.category_vigilance && activation > max_activation) {
                max_activation = activation;
                max_index = tc;
            }

            if (activation > max_complete_activation) {
                max_complete_activation = activation;
                max_complete_index = tc;
            }
        }

        temporal_cis[hidden_column_index] = (max_index == -1 ? max_complete_index : max_index) + hidden_ci * temporal_size;

        temporal_learn_flags[hidden_column_index] = (max_index != -1);
    }
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

    if (ratio <= params.active_ratio && hidden_learn_flags[hidden_column_index]) {
        int hidden_ci = hidden_cis[hidden_column_index];

        // spatial
        int hidden_cell_index_max = hidden_ci + hidden_cells_start;

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

            for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                    int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                    int in_ci = vl_input_cis[visible_column_index];

                    Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                    int wi = hidden_ci + hidden_size.z * (offset.y + diam * (offset.x + diam * (in_ci + vld.size.z * hidden_column_index)));

                    Byte w_old = vl.weights[wi];

                    vl.weights[wi] = min(255, vl.weights[wi] + ceilf(params.lr * (255.0f - vl.weights[wi])));

                    vl.hidden_totals[hidden_cell_index_max] += vl.weights[wi] - w_old;
                }
        }
    }

    if (temporal_learn_flags[hidden_column_index]) {
        int temporal_ci = temporal_cis[hidden_column_index];

        int full_cell_index_max = temporal_ci + full_cells_start;

        int diam = recurrent_radius * 2 + 1;

        // lower corner
        Int2 field_lower_bound(column_pos.x - recurrent_radius, column_pos.y - recurrent_radius);

        // bounds of receptive field, clamped to input size
        Int2 iter_lower_bound(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
        Int2 iter_upper_bound(min(hidden_size.x - 1, column_pos.x + recurrent_radius), min(hidden_size.y - 1, column_pos.y + recurrent_radius));

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int other_hidden_column_index = address2(Int2(ix, iy), Int2(hidden_size.x, hidden_size.y));

                int in_ci = temporal_cis_prev[other_hidden_column_index];

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi = temporal_ci + full_column_size * (offset.y + diam * (offset.x + diam * (in_ci + full_column_size * hidden_column_index)));

                Byte w_old = recurrent_weights[wi];

                recurrent_weights[wi] = min(255, recurrent_weights[wi] + ceilf(params.lr * (255.0f - recurrent_weights[wi])));

                recurrent_totals[full_cell_index_max] += recurrent_weights[wi] - w_old;
            }
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

        vl.weights.resize(num_hidden_cells * area * vld.size.z);

        for (int i = 0; i < vl.weights.size(); i++)
            vl.weights[i] = (rand() % init_weight_noisei);

        vl.hidden_sums.resize(num_hidden_cells);
        vl.hidden_totals.resize(num_hidden_cells);
        vl.hidden_counts.resize(num_hidden_columns);
    }

    hidden_cis = Int_Buffer(num_hidden_columns, 0);
    temporal_cis = Int_Buffer(num_hidden_columns, 0);
    temporal_cis_prev.resize(num_hidden_columns);

    hidden_learn_flags.resize(num_hidden_columns);
    temporal_learn_flags.resize(num_hidden_columns);

    hidden_comparisons.resize(num_hidden_columns);

    int diam = recurrent_radius * 2 + 1;
    int area = diam * diam;

    recurrent_weights.resize(num_full_cells * area * full_column_size);

    for (int i = 0; i < recurrent_weights.size(); i++)
        recurrent_weights[i] = (rand() % init_weight_noisei);

    recurrent_totals.resize(num_full_cells);

    recurrent_sums.resize(num_temporal_cells);

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

            vl.hidden_counts[hidden_column_index] = (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1);

            for (int hc = 0; hc < hidden_size.z; hc++) {
                int hidden_cell_index = hc + hidden_cells_start;

                int total = 0;

                for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                    for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                        int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                        Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                        for (int vc = 0; vc < vld.size.z; vc++) {
                            int wi = hc + hidden_size.z * (offset.y + diam * (offset.x + diam * (vc + vld.size.z * hidden_column_index)));

                            total += vl.weights[wi];
                        }
                    }

                vl.hidden_totals[hidden_cell_index] = total;
            }
        }

        // recurrent
        int diam = recurrent_radius * 2 + 1;

        // lower corner
        Int2 field_lower_bound(column_pos.x - recurrent_radius, column_pos.y - recurrent_radius);

        // bounds of receptive field, clamped to input size
        Int2 iter_lower_bound(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
        Int2 iter_upper_bound(min(hidden_size.x - 1, column_pos.x + recurrent_radius), min(hidden_size.y - 1, column_pos.y + recurrent_radius));

        for (int fc = 0; fc < full_column_size; fc++) {
            int full_cell_index = fc + full_cells_start;

            int total = 0;

            for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                    Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                    for (int ofc = 0; ofc < full_column_size; ofc++) {
                        int wi = fc + full_column_size * (offset.y + diam * (offset.x + diam * (ofc + full_column_size * hidden_column_index)));

                        total += recurrent_weights[wi];
                    }
                }

            recurrent_totals[full_cell_index] = total;
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

    PARALLEL_FOR
    for (int i = 0; i < num_hidden_columns; i++)
        forward(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, params);

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
    long size = sizeof(Int3) + 2 * sizeof(int) + 2 * hidden_cis.size() * sizeof(int) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += sizeof(Visible_Layer_Desc) + vl.weights.size() * sizeof(Byte) + vl.hidden_totals.size() * sizeof(int) + vl.hidden_counts.size() * sizeof(int) + sizeof(float);
    }

    size += recurrent_weights.size() * sizeof(Byte) + recurrent_totals.size() * sizeof(int);

    return size;
}

long Encoder::state_size() const {
    return 2 * hidden_cis.size() * sizeof(int);
}

long Encoder::weights_size() const {
    long size = 0;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += vl.weights.size() * sizeof(Byte);
    }

    size += recurrent_weights.size() * sizeof(Byte);

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

    int num_visible_layers = visible_layers.size();

    writer.write(&num_visible_layers, sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(&vld, sizeof(Visible_Layer_Desc));

        writer.write(&vl.weights[0], vl.weights.size() * sizeof(Byte));

        writer.write(&vl.hidden_totals[0], vl.hidden_totals.size() * sizeof(int));
        writer.write(&vl.hidden_counts[0], vl.hidden_counts.size() * sizeof(int));

        writer.write(&vl.importance, sizeof(float));
    }

    writer.write(&recurrent_weights[0], recurrent_weights.size() * sizeof(Byte));

    writer.write(&recurrent_totals[0], recurrent_totals.size() * sizeof(int));
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

    hidden_learn_flags.resize(num_hidden_columns);
    temporal_learn_flags.resize(num_hidden_columns);

    hidden_comparisons.resize(num_hidden_columns);

    recurrent_sums.resize(num_temporal_cells);

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

        vl.weights.resize(num_hidden_cells * area * vld.size.z);

        reader.read(&vl.weights[0], vl.weights.size() * sizeof(Byte));

        vl.hidden_sums.resize(num_hidden_cells);

        vl.hidden_totals.resize(num_hidden_cells);

        reader.read(&vl.hidden_totals[0], vl.hidden_totals.size() * sizeof(int));

        vl.hidden_counts.resize(num_hidden_columns);

        reader.read(&vl.hidden_counts[0], vl.hidden_counts.size() * sizeof(int));

        reader.read(&vl.importance, sizeof(float));
    }

    int diam = recurrent_radius * 2 + 1;
    int area = diam * diam;

    recurrent_weights.resize(num_full_cells * area * full_column_size);

    reader.read(&recurrent_weights[0], recurrent_weights.size() * sizeof(Byte));

    recurrent_totals.resize(num_full_cells);

    reader.read(&recurrent_totals[0], recurrent_totals.size() * sizeof(int));
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

        writer.write(&vl.weights[0], vl.weights.size() * sizeof(Byte));
    }

    writer.write(&recurrent_weights[0], recurrent_weights.size() * sizeof(Byte));
}

void Encoder::read_weights(
    Stream_Reader &reader
) {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(&vl.weights[0], vl.weights.size() * sizeof(Byte));
    }

    reader.read(&recurrent_weights[0], recurrent_weights.size() * sizeof(Byte));
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
        
            for (int i = 0; i < vl.weights.size(); i++) {
                int e = rand() % encoders.size();                

                vl.weights[i] = encoders[e]->visible_layers[vli].weights[i];
            }
        }

        for (int i = 0; i < recurrent_weights.size(); i++) {
            int e = rand() % encoders.size();                

            recurrent_weights[i] = encoders[e]->recurrent_weights[i];
        }

        break;
    case merge_average:
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];
        
            for (int i = 0; i < vl.weights.size(); i++) {
                float total = 0.0f;

                for (int e = 0; e < encoders.size(); e++)
                    total += encoders[e]->visible_layers[vli].weights[i];

                vl.weights[i] = roundf(total / encoders.size());
            }
        }

        for (int i = 0; i < recurrent_weights.size(); i++) {
            float total = 0.0f;

            for (int e = 0; e < encoders.size(); e++)
                total += encoders[e]->recurrent_weights[i];

            recurrent_weights[i] = roundf(total / encoders.size());
        }

        break;
    }

    int full_column_size = hidden_size.z * temporal_size;

    int num_hidden_columns = hidden_size.x * hidden_size.y;

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

            for (int hc = 0; hc < hidden_size.z; hc++) {
                int hidden_cell_index = hc + hidden_cells_start;

                int total = 0;

                for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                    for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                        int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                        Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                        for (int vc = 0; vc < vld.size.z; vc++) {
                            int wi = hc + hidden_size.z * (offset.y + diam * (offset.x + diam * (vc + vld.size.z * hidden_column_index)));

                            total += vl.weights[wi];
                        }
                    }

                vl.hidden_totals[hidden_cell_index] = total;
            }
        }

        // recurrent
        int diam = recurrent_radius * 2 + 1;

        // lower corner
        Int2 field_lower_bound(column_pos.x - recurrent_radius, column_pos.y - recurrent_radius);

        // bounds of receptive field, clamped to input size
        Int2 iter_lower_bound(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
        Int2 iter_upper_bound(min(hidden_size.x - 1, column_pos.x + recurrent_radius), min(hidden_size.y - 1, column_pos.y + recurrent_radius));

        for (int fc = 0; fc < full_column_size; fc++) {
            int full_cell_index = fc + full_cells_start;

            int total = 0;

            for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                    Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                    for (int ofc = 0; ofc < full_column_size; ofc++) {
                        int wi = fc + full_column_size * (offset.y + diam * (offset.x + diam * (ofc + full_column_size * hidden_column_index)));

                        total += recurrent_weights[wi];
                    }
                }

            recurrent_totals[full_cell_index] = total;
        }
    }
}
