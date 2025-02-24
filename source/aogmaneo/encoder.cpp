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
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        hidden_acts[hidden_cell_index] = 0.0f;
    }

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        if (vl.importance == 0.0f)
            continue;

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

        int hidden_stride = vld.size.z * diam * diam;

        const float influence = vl.importance * sqrtf(1.0f / sub_count) / 255.0f;

        Int_Buffer_View vl_input_cis = input_cis[vli];

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci = vl_input_cis[visible_column_index];

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi_offset = in_ci + vld.size.z * (offset.y + diam * offset.x);

                for (int hc = 0; hc < hidden_size.z; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int wi = wi_offset + hidden_cell_index * hidden_stride;

                    hidden_acts[hidden_cell_index] += vl.weights[wi] * influence;
                }
            }
    }

    int max_index = 0;
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
}

void Encoder::forward_recurrent(
    const Int2 &column_pos,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int full_column_size = hidden_size.z * temporal_size;

    int temporal_cells_start = hidden_column_index * temporal_size;
    int full_cells_start = hidden_column_index * full_column_size;

    int hidden_ci = hidden_cis[hidden_column_index];

    for (int tc = 0; tc < temporal_size; tc++) {
        int temporal_cell_index = tc + temporal_cells_start;

        temporal_acts[temporal_cell_index] = 0;
    }

    int diam = recurrent_radius * 2 + 1;

    // lower corner
    Int2 field_lower_bound(column_pos.x - recurrent_radius, column_pos.y - recurrent_radius);

    // bounds of receptive field, clamped to input size
    Int2 iter_lower_bound(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
    Int2 iter_upper_bound(min(hidden_size.x - 1, column_pos.x + recurrent_radius), min(hidden_size.y - 1, column_pos.y + recurrent_radius));

    int hidden_stride = full_column_size * diam * diam;

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            int other_hidden_column_index = address2(Int2(ix, iy), Int2(hidden_size.x, hidden_size.y));

            int in_ci = temporal_cis_prev[other_hidden_column_index];

            Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

            int wi_offset = in_ci + full_column_size * (offset.y + diam * offset.x);

            for (int tc = 0; tc < temporal_size; tc++) {
                int temporal_cell_index = tc + temporal_cells_start;

                int full_cell_index = tc + hidden_ci * temporal_size + full_cells_start;

                int wi = wi_offset + full_cell_index * hidden_stride;

                temporal_acts[temporal_cell_index] += recurrent_weights[wi];
            }
        }

    int max_index = 0;
    int max_activation = 0;

    for (int tc = 0; tc < temporal_size; tc++) {
        int temporal_cell_index = tc + temporal_cells_start;

        int activation = temporal_acts[temporal_cell_index];

        if (activation > max_activation) {
            max_activation = activation;
            max_index = tc;
        }
    }

    temporal_cis[hidden_column_index] = max_index + hidden_ci * temporal_size;
}

void Encoder::learn_spatial(
    const Int2 &column_pos,
    Int_Buffer_View input_cis,
    int vli,
    unsigned long* state,
    const Params &params
) {
    Visible_Layer &vl = visible_layers[vli];
    const Visible_Layer_Desc &vld = visible_layer_descs[vli];

    if (vl.importance == 0.0f)
        return;

    int diam = vld.radius * 2 + 1;

    int visible_column_index = address2(column_pos, Int2(vld.size.x, vld.size.y));

    int visible_cells_start = visible_column_index * vld.size.z;

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

    int in_ci = input_cis[visible_column_index];

    // clear
    for (int vc = 0; vc < vld.size.z; vc++) {
        int visible_cell_index = vc + visible_cells_start;

        vl.recon_sums[visible_cell_index] = 0;
    }

    int count = 0;

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            Int2 hidden_pos = Int2(ix, iy);

            int hidden_column_index = address2(hidden_pos, Int2(hidden_size.x, hidden_size.y));

            Int2 visible_center = project(hidden_pos, h_to_v);

            if (in_bounds(column_pos, Int2(visible_center.x - vld.radius, visible_center.y - vld.radius), Int2(visible_center.x + vld.radius + 1, visible_center.y + vld.radius + 1))) {
                Int2 offset(column_pos.x - visible_center.x + vld.radius, column_pos.y - visible_center.y + vld.radius);

                int hidden_cell_index = hidden_cis[hidden_column_index] + hidden_column_index * hidden_size.z;

                int wi_start = vld.size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index));

                for (int vc = 0; vc < vld.size.z; vc++) {
                    int visible_cell_index = vc + visible_cells_start;

                    int wi = vc + wi_start;

                    vl.recon_sums[visible_cell_index] += vl.weights[wi];
                }

                count++;
            }
        }

    const float recon_scale = sqrtf(1.0f / max(1, count)) / 255.0f * params.scale;

    int num_higher = 0;

    int target_recon_sum = vl.recon_sums[in_ci + visible_cells_start];

    for (int vc = 0; vc < vld.size.z; vc++) {
        int visible_cell_index = vc + visible_cells_start;

        int recon_sum = vl.recon_sums[visible_cell_index];

        if (recon_sum >= target_recon_sum && vc != in_ci)
            num_higher++;

        // re-use sums as deltas
        vl.recon_sums[visible_cell_index] = rand_roundf(params.lr * 255.0f * (vc == in_ci ? 1.0f - params.bias : 1.0f) * ((vc == in_ci) - expf((recon_sum - count * 255) * recon_scale)), state);
    }

    if (num_higher < params.spatial_recon_tolerance)
        return;

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            Int2 hidden_pos = Int2(ix, iy);

            int hidden_column_index = address2(hidden_pos, Int2(hidden_size.x, hidden_size.y));

            Int2 visible_center = project(hidden_pos, h_to_v);

            if (in_bounds(column_pos, Int2(visible_center.x - vld.radius, visible_center.y - vld.radius), Int2(visible_center.x + vld.radius + 1, visible_center.y + vld.radius + 1))) {
                Int2 offset(column_pos.x - visible_center.x + vld.radius, column_pos.y - visible_center.y + vld.radius);

                int hidden_cell_index = hidden_cis[hidden_column_index] + hidden_column_index * hidden_size.z;

                int wi_start = vld.size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index));

                for (int vc = 0; vc < vld.size.z; vc++) {
                    int visible_cell_index = vc + visible_cells_start;

                    int wi = vc + wi_start;

                    vl.weights[wi] = min(255, max(0, vl.weights[wi] + vl.recon_sums[visible_cell_index]));
                }
            }
        }
}

void Encoder::learn_recurrent(
    const Int2 &column_pos,
    unsigned long* state,
    const Params &params
) {
    int other_hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int full_column_size = hidden_size.z * temporal_size;

    int other_full_cells_start = other_hidden_column_index * full_column_size;

    int temporal_ci_prev = temporal_cis_prev[other_hidden_column_index];

    int diam = recurrent_radius * 2 + 1;

    // lower corner
    Int2 field_lower_bound(column_pos.x - recurrent_radius, column_pos.y - recurrent_radius);

    // bounds of receptive field, clamped to input size
    Int2 iter_lower_bound(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
    Int2 iter_upper_bound(min(hidden_size.x - 1, column_pos.x + recurrent_radius), min(hidden_size.y - 1, column_pos.y + recurrent_radius));

    // clear
    for (int ofc = 0; ofc < full_column_size; ofc++) {
        int other_full_cell_index = ofc + other_full_cells_start;

        recurrent_recon_sums[other_full_cell_index] = 0;
    }

    int count = (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1);

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            Int2 hidden_pos = Int2(ix, iy);

            int hidden_column_index = address2(hidden_pos, Int2(hidden_size.x, hidden_size.y));

            int full_cell_index = temporal_cis[hidden_column_index] + hidden_column_index * full_column_size;

            Int2 offset(column_pos.x - hidden_pos.x + recurrent_radius, column_pos.y - hidden_pos.y + recurrent_radius);

            int wi_start = full_column_size * (offset.y + diam * (offset.x + diam * full_cell_index));

            for (int ofc = 0; ofc < full_column_size; ofc++) {
                int other_full_cell_index = ofc + other_full_cells_start;

                int wi = ofc + wi_start;

                recurrent_recon_sums[other_full_cell_index] += recurrent_weights[wi];
            }
        }

    const float recon_scale = sqrtf(1.0f / max(1, count)) / 255.0f * params.scale;

    int num_higher = 0;

    int target_recon_sum = recurrent_recon_sums[temporal_ci_prev + other_full_cells_start];

    for (int ofc = 0; ofc < full_column_size; ofc++) {
        int other_full_cell_index = ofc + other_full_cells_start;

        int recon_sum = recurrent_recon_sums[other_full_cell_index];

        if (recon_sum >= target_recon_sum && ofc != temporal_ci_prev)
            num_higher++;

        // re-use sums as deltas
        recurrent_recon_sums[other_full_cell_index] = rand_roundf(params.lr * 255.0f * ((ofc == temporal_ci_prev) - expf((recon_sum - count * 255) * recon_scale)), state);
    }

    if (num_higher < params.recurrent_recon_tolerance)
        return;

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            Int2 hidden_pos = Int2(ix, iy);

            int hidden_column_index = address2(hidden_pos, Int2(hidden_size.x, hidden_size.y));

            int full_cell_index = temporal_cis[hidden_column_index] + hidden_column_index * full_column_size;

            Int2 offset(column_pos.x - hidden_pos.x + recurrent_radius, column_pos.y - hidden_pos.y + recurrent_radius);

            int wi_start = full_column_size * (offset.y + diam * (offset.x + diam * full_cell_index));

            for (int ofc = 0; ofc < full_column_size; ofc++) {
                int other_full_cell_index = ofc + other_full_cells_start;

                int wi = ofc + wi_start;

                recurrent_weights[wi] = min(255, max(0, recurrent_weights[wi] + recurrent_recon_sums[other_full_cell_index]));
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

    int total_num_visible_columns = 0;

    // create layers
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;
        int num_visible_cells = num_visible_columns * vld.size.z;

        total_num_visible_columns += num_visible_columns;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(num_hidden_cells * area * vld.size.z);

        for (int i = 0; i < vl.weights.size(); i++)
            vl.weights[i] = 255 - (rand() % init_weight_noisei);

        vl.recon_sums.resize(num_visible_cells);
    }

    hidden_cis = Int_Buffer(num_hidden_columns, 0);
    temporal_cis = Int_Buffer(num_hidden_columns, 0);
    temporal_cis_prev.resize(num_hidden_columns);

    hidden_acts.resize(num_hidden_cells);
    temporal_acts.resize(num_temporal_cells);

    int diam = recurrent_radius * 2 + 1;
    int area = diam * diam;

    recurrent_weights.resize(num_full_cells * area * full_column_size);

    for (int i = 0; i < recurrent_weights.size(); i++)
        recurrent_weights[i] = 255 - (rand() % init_weight_noisei);

    recurrent_recon_sums.resize(num_full_cells);

    // generate helper buffers for parallelization
    visible_pos_vlis.resize(total_num_visible_columns);

    int index = 0;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;

        for (int i = 0; i < num_visible_columns; i++) {
            visible_pos_vlis[index] = Int3(i / vld.size.y, i % vld.size.y, vli);
            index++;
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
        forward_spatial(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, params);

    PARALLEL_FOR
    for (int i = 0; i < num_hidden_columns; i++)
        forward_recurrent(Int2(i / hidden_size.y, i % hidden_size.y), params);

    if (learn_enabled) {
        unsigned int base_state = rand();

        PARALLEL_FOR
        for (int i = 0; i < visible_pos_vlis.size(); i++) {
            Int2 pos = Int2(visible_pos_vlis[i].x, visible_pos_vlis[i].y);
            int vli = visible_pos_vlis[i].z;

            unsigned long state = rand_get_state(base_state + i * rand_subseed_offset);

            learn_spatial(pos, input_cis[vli], vli, &state, params);
        }

        base_state = rand();

        PARALLEL_FOR
        for (int i = 0; i < num_hidden_columns; i++) {
            unsigned long state = rand_get_state(base_state + i * rand_subseed_offset);

            learn_recurrent(Int2(i / hidden_size.y, i % hidden_size.y), &state, params);
        }
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

        size += sizeof(Visible_Layer_Desc) + vl.weights.size() * sizeof(Byte) + sizeof(float);
    }

    size += recurrent_weights.size() * sizeof(Byte);

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

        writer.write(&vl.importance, sizeof(float));
    }

    writer.write(&recurrent_weights[0], recurrent_weights.size() * sizeof(Byte));
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

    temporal_cis_prev.resize(num_hidden_columns);

    hidden_acts.resize(num_hidden_cells);
    temporal_acts.resize(num_temporal_cells);

    int num_visible_layers = visible_layers.size();

    reader.read(&num_visible_layers, sizeof(int));

    visible_layers.resize(num_visible_layers);
    visible_layer_descs.resize(num_visible_layers);
    
    int total_num_visible_columns = 0;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        Visible_Layer_Desc &vld = visible_layer_descs[vli];

        reader.read(&vld, sizeof(Visible_Layer_Desc));

        int num_visible_columns = vld.size.x * vld.size.y;
        int num_visible_cells = num_visible_columns * vld.size.z;

        total_num_visible_columns += num_visible_columns;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(num_hidden_cells * area * vld.size.z);

        reader.read(&vl.weights[0], vl.weights.size() * sizeof(Byte));

        vl.recon_sums.resize(num_visible_cells);

        reader.read(&vl.importance, sizeof(float));
    }

    int diam = recurrent_radius * 2 + 1;
    int area = diam * diam;

    recurrent_weights.resize(num_full_cells * area * full_column_size);

    reader.read(&recurrent_weights[0], recurrent_weights.size() * sizeof(Byte));

    recurrent_recon_sums.resize(num_full_cells);

    // generate helper buffers for parallelization
    visible_pos_vlis.resize(total_num_visible_columns);

    int index = 0;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;

        for (int i = 0; i < num_visible_columns; i++) {
            visible_pos_vlis[index] = Int3(i / vld.size.y, i % vld.size.y, vli);
            index++;
        }
    }
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

        break;
    }
}
