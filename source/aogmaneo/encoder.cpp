// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
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

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        // update traces from last time (delayed by 1 step)
        hidden_traces[hidden_cell_index] *= params.trace_decay;
    }

    hidden_traces[hidden_cis[hidden_column_index] + hidden_cells_start] = 1.0f;

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

                    hidden_acts[hidden_cell_index] += vl.weights_forward[wi] * influence;
                }
            }
    }

    int max_index = 0;
    float max_activation = 0.0f;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        if (hidden_acts[hidden_cell_index] > max_activation) {
            max_activation = hidden_acts[hidden_cell_index];
            max_index = hc;
        }
    }

    hidden_cis[hidden_column_index] = max_index;
    hidden_acts[max_index + hidden_cells_start] = 0.0f; // reset (fired)
}

void Encoder::backward(
    const Int2 &column_pos,
    int vli,
    Int_Buffer_View input_cis,
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
                int hidden_cell_index_max = hidden_cis[hidden_column_index] + hidden_column_index * hidden_size.z;

                Int2 offset(column_pos.x - visible_center.x + vld.radius, column_pos.y - visible_center.y + vld.radius);

                int wi_start = vld.size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index_max));

                for (int vc = 0; vc < vld.size.z; vc++) {
                    int visible_cell_index = vc + visible_cells_start;

                    int wi = vc + wi_start;

                    vl.recon_sums[visible_cell_index] += vl.weights_backward[wi];
                }

                count++;
            }
        }

    const float recon_scale = sqrtf(1.0f / max(1, count)) / 127.0f * params.scale;

    int in_ci = input_cis[visible_column_index];

    int max_recon_sum = limit_min;

    for (int vc = 0; vc < vld.size.z; vc++) {
        int visible_cell_index = vc + visible_cells_start;

        max_recon_sum = max(max_recon_sum, vl.recon_sums[visible_cell_index]);
    }

    float total = 0.0f;

    for (int vc = 0; vc < vld.size.z; vc++) {
        int visible_cell_index = vc + visible_cells_start;

        vl.recon_acts[visible_cell_index] = expf((vl.recon_sums[visible_cell_index] - max_recon_sum) * recon_scale);

        total += vl.recon_acts[visible_cell_index];
    }

    float total_inv = 1.0f / max(limit_small, total);

    for (int vc = 0; vc < vld.size.z; vc++) {
        int visible_cell_index = vc + visible_cells_start;

        vl.recon_acts[visible_cell_index] *= total_inv;

        // re-use buffers as deltas
        vl.recon_acts[visible_cell_index] = ((vc == in_ci) - vl.recon_acts[visible_cell_index]);
        vl.recon_sums[visible_cell_index] = rand_roundf(params.blr * 127.0f * vl.recon_acts[visible_cell_index], state);
    }

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            Int2 hidden_pos = Int2(ix, iy);

            int hidden_column_index = address2(hidden_pos, Int2(hidden_size.x, hidden_size.y));

            Int2 visible_center = project(hidden_pos, h_to_v);

            if (in_bounds(column_pos, Int2(visible_center.x - vld.radius, visible_center.y - vld.radius), Int2(visible_center.x + vld.radius + 1, visible_center.y + vld.radius + 1))) {
                int hidden_cell_index_max = hidden_cis[hidden_column_index] + hidden_column_index * hidden_size.z;

                Int2 offset(column_pos.x - visible_center.x + vld.radius, column_pos.y - visible_center.y + vld.radius);

                int wi_start = vld.size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index_max));

                for (int vc = 0; vc < vld.size.z; vc++) {
                    int visible_cell_index = vc + visible_cells_start;

                    int wi = vc + wi_start;

                    vl.weights_backward[wi] = min(127, max(-127, vl.weights_backward[wi] + vl.recon_sums[visible_cell_index]));
                }
            }
        }
}

void Encoder::learn(
    const Int2 &column_pos,
    const Array<Int_Buffer_View> &input_cis,
    unsigned long* state,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    int hidden_cell_index_max = hidden_cis[hidden_column_index] + hidden_cells_start;

    float error = 0.0f;

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

        const float influence = vl.importance * sqrtf(1.0f / sub_count) / 127.0f;

        Int_Buffer_View vl_input_cis = input_cis[vli];

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int visible_cells_start = visible_column_index * vld.size.z;

                int in_ci = vl_input_cis[visible_column_index];

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi_start = vld.size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index_max));

                for (int vc = 0; vc < vld.size.z; vc++) {
                    int visible_cell_index = vc + visible_cells_start;

                    int wi = vc + wi_start;

                    error += influence * vl.weights_backward[wi] * vl.recon_acts[visible_cell_index];
                }
            }
    }

    // hidden-triggered STDP (positive)
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

        Int_Buffer_View vl_input_cis = input_cis[vli];

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int visible_cells_start = visible_column_index * vld.size.z;

                int in_ci = vl_input_cis[visible_column_index];

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);
                
                int wi_start = vld.size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index_max));

                for (int vc = 0; vc < vld.size.z; vc++) {
                    int visible_cell_index = vc + visible_cells_start;

                    int wi = vc + wi_start;

                    vl.weights_forward[wi] = min(255, max(0, vl.weights_forward[wi] + roundf(params.flr * 255.0f * error * vl.traces[visible_cell_index])));
                }
            }
    }

    // input-triggered STDP (negative)
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

        int hidden_stride = vld.size.z * diam * diam;

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

                    vl.weights_forward[wi] = min(255, max(0, vl.weights_forward[wi] - roundf(params.flr * 255.0f * error * hidden_traces[hidden_cell_index])));
                }
            }
    }
}

void Encoder::input_trace_update(
    const Int2 &column_pos,
    int vli,
    Int_Buffer_View input_cis,
    const Params &params
) {
    Visible_Layer &vl = visible_layers[vli];
    const Visible_Layer_Desc &vld = visible_layer_descs[vli];

    int visible_column_index = address2(column_pos, Int2(vld.size.x, vld.size.y));

    int visible_cells_start = visible_column_index * vld.size.z;

    for (int vc = 0; vc < vld.size.z; vc++)
        vl.traces[vc + visible_cells_start] *= params.trace_decay;

    vl.traces[input_cis[visible_column_index] + visible_cells_start] = 1.0f;
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

        vl.weights_forward.resize(num_hidden_cells * area * vld.size.z);
        vl.weights_backward.resize(vl.weights_forward.size());

        for (int i = 0; i < vl.weights_forward.size(); i++) {
            vl.weights_forward[i] = 255 - (rand() % init_weight_noisei);
            vl.weights_backward[i] = (rand() % (init_weight_noisei + 1)) - init_weight_noisei / 2;
        }

        vl.traces = Float_Buffer(num_visible_cells, 0.0f);

        vl.recon_sums.resize(num_visible_cells);
        vl.recon_acts.resize(num_visible_cells);
    }

    hidden_cis = Int_Buffer(num_hidden_columns, 0);

    hidden_acts = Float_Buffer(num_hidden_cells, 0.0f);

    hidden_traces = Float_Buffer(num_hidden_cells, 0.0f);

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

    // update visible traces
    PARALLEL_FOR
    for (int i = 0; i < visible_pos_vlis.size(); i++) {
        Int2 pos = Int2(visible_pos_vlis[i].x, visible_pos_vlis[i].y);
        int vli = visible_pos_vlis[i].z;

        input_trace_update(pos, vli, input_cis[vli], params);
    }
    
    PARALLEL_FOR
    for (int i = 0; i < num_hidden_columns; i++)
        forward(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, params);

    if (learn_enabled) {
        unsigned int base_state = rand();

        PARALLEL_FOR
        for (int i = 0; i < visible_pos_vlis.size(); i++) {
            Int2 pos = Int2(visible_pos_vlis[i].x, visible_pos_vlis[i].y);
            int vli = visible_pos_vlis[i].z;

            unsigned long state = rand_get_state(base_state + i * rand_subseed_offset);

            backward(pos, vli, input_cis[vli], &state, params);
        }

        base_state = rand();

        PARALLEL_FOR
        for (int i = 0; i < num_hidden_columns; i++) {
            unsigned long state = rand_get_state(base_state + i * rand_subseed_offset);

            learn(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, &state, params);
        }
    }
}

void Encoder::clear_state() {
    hidden_cis.fill(0);
}

long Encoder::size() const {
    long size = sizeof(Int3) + hidden_cis.size() * sizeof(int) + hidden_acts.size() * sizeof(float) + hidden_traces.size() * sizeof(float) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += sizeof(Visible_Layer_Desc) + vl.weights_forward.size() * sizeof(Byte) + vl.weights_backward.size() * sizeof(S_Byte) + vl.traces.size() * sizeof(float) + sizeof(float);
    }

    return size;
}

long Encoder::state_size() const {
    long size = hidden_cis.size() * sizeof(int) + hidden_acts.size() * sizeof(float) + hidden_traces.size() * sizeof(float);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += vl.traces.size() * sizeof(float);
    }

    return size;
}

long Encoder::weights_size() const {
    long size = 0;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += vl.weights_forward.size() * sizeof(Byte) + vl.weights_backward.size() * sizeof(S_Byte);
    }

    return size;
}

void Encoder::write(
    Stream_Writer &writer
) const {
    writer.write(&hidden_size, sizeof(Int3));

    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    writer.write(&hidden_acts[0], hidden_acts.size() * sizeof(float));
    writer.write(&hidden_traces[0], hidden_traces.size() * sizeof(float));

    int num_visible_layers = visible_layers.size();

    writer.write(&num_visible_layers, sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(&vld, sizeof(Visible_Layer_Desc));

        writer.write(&vl.weights_forward[0], vl.weights_forward.size() * sizeof(Byte));
        writer.write(&vl.weights_backward[0], vl.weights_backward.size() * sizeof(S_Byte));

        writer.write(&vl.traces[0], vl.traces.size() * sizeof(float));

        writer.write(&vl.importance, sizeof(float));
    }
}

void Encoder::read(
    Stream_Reader &reader
) {
    reader.read(&hidden_size, sizeof(Int3));

    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;

    hidden_cis.resize(num_hidden_columns);
    hidden_acts.resize(num_hidden_cells);
    hidden_traces.resize(num_hidden_cells);

    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    reader.read(&hidden_acts[0], hidden_cis.size() * sizeof(float));
    reader.read(&hidden_traces[0], hidden_traces.size() * sizeof(float));

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

        vl.weights_forward.resize(num_hidden_cells * area * vld.size.z);
        vl.weights_backward.resize(vl.weights_forward.size());

        reader.read(&vl.weights_forward[0], vl.weights_forward.size() * sizeof(Byte));
        reader.read(&vl.weights_backward[0], vl.weights_backward.size() * sizeof(S_Byte));

        vl.traces.resize(num_visible_cells);

        reader.read(&vl.traces[0], vl.traces.size() * sizeof(float));

        vl.recon_sums.resize(num_visible_cells);
        vl.recon_acts.resize(num_visible_cells);

        reader.read(&vl.importance, sizeof(float));
    }

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
    writer.write(&hidden_acts[0], hidden_acts.size() * sizeof(float));
    writer.write(&hidden_traces[0], hidden_traces.size() * sizeof(float));

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        writer.write(&vl.traces[0], vl.traces.size() * sizeof(float));
    }
}

void Encoder::read_state(
    Stream_Reader &reader
) {
    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    reader.read(&hidden_acts[0], hidden_acts.size() * sizeof(float));
    reader.read(&hidden_traces[0], hidden_traces.size() * sizeof(float));

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(&vl.traces[0], vl.traces.size() * sizeof(float));
    }
}

void Encoder::write_weights(
    Stream_Writer &writer
) const {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        writer.write(&vl.weights_forward[0], vl.weights_forward.size() * sizeof(Byte));
        writer.write(&vl.weights_backward[0], vl.weights_backward.size() * sizeof(S_Byte));
    }
}

void Encoder::read_weights(
    Stream_Reader &reader
) {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(&vl.weights_forward[0], vl.weights_forward.size() * sizeof(Byte));
        reader.read(&vl.weights_backward[0], vl.weights_backward.size() * sizeof(S_Byte));
    }
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
        
            for (int i = 0; i < vl.weights_forward.size(); i++) {
                int e = rand() % encoders.size();                

                vl.weights_forward[i] = encoders[e]->visible_layers[vli].weights_forward[i];
                vl.weights_backward[i] = encoders[e]->visible_layers[vli].weights_backward[i];
            }
        }

        break;
    case merge_average:
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];
        
            for (int i = 0; i < vl.weights_forward.size(); i++) {
                float total_forward = 0.0f;
                float total_backward = 0.0f;

                for (int e = 0; e < encoders.size(); e++) {
                    total_forward += encoders[e]->visible_layers[vli].weights_forward[i];
                    total_backward += encoders[e]->visible_layers[vli].weights_backward[i];
                }

                vl.weights_forward[i] = roundf(total_forward / encoders.size());
                vl.weights_backward[i] = roundf(total_backward / encoders.size());
            }
        }

        break;
    }
}
