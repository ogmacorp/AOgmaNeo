// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "actor.h"

using namespace aon;

void Actor::forward(
    const Int2 &column_pos,
    const Array<Int_Buffer_View> &input_cis,
    Int_Buffer_View hidden_target_cis,
    float reward,
    float mimic,
    bool learn_enabled,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    int target_ci = hidden_target_cis[hidden_column_index];

    int hidden_cell_index_target = target_ci + hidden_cells_start;

    float value_prev = hidden_values[hidden_column_index];

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        hidden_acts[hidden_cell_index] = 0.0f;
    }

    float value = 0.0f;
    int count = 0;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        int diam = vld.radius * 2 + 1;

        // Projection
        Float2 h_to_v = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hidden_size.x),
            static_cast<float>(vld.size.y) / static_cast<float>(hidden_size.y));

        Int2 visible_center = project(column_pos, h_to_v);

        // Lower corner
        Int2 field_lower_bound(visible_center.x - vld.radius, visible_center.y - vld.radius);

        // Bounds of receptive field, clamped to input size
        Int2 iter_lower_bound(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
        Int2 iter_upper_bound(min(vld.size.x - 1, visible_center.x + vld.radius), min(vld.size.y - 1, visible_center.y + vld.radius));

        count += (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1);

        Int_Buffer_View vl_input_cis = input_cis[vli];

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci = vl_input_cis[visible_column_index];

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);
                
                int value_wi = offset.y + diam * (offset.x + diam * (in_ci + vld.size.z * hidden_column_index));
                int action_wi_start = hidden_size.z * value_wi;

                for (int hc = 0; hc < hidden_size.z; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int action_wi = hc + action_wi_start;

                    hidden_acts[hidden_cell_index] += vl.action_weights[action_wi];
                }

                value += vl.value_weights[value_wi];
            }
    }

    value /= count;

    hidden_values[hidden_column_index] = value;

    int max_index = 0;
    float max_activation = limit_min;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        hidden_acts[hidden_cell_index] /= count;

        float activation = hidden_acts[hidden_cell_index];

        if (activation > max_activation) {
            max_activation = activation;
            max_index = hc;
        }
    }

    float total = 0.0f;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;
    
        hidden_acts[hidden_cell_index] = expf(hidden_acts[hidden_cell_index] - max_activation);

        total += hidden_acts[hidden_cell_index];
    }

    float total_inv = 1.0f / max(limit_small, total);

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        hidden_acts[hidden_cell_index] *= total_inv;
    }

    hidden_cis[hidden_column_index] = max_index;

    if (learn_enabled) {
        float td_error = reward + params.discount * value - value_prev;

        float value_delta = params.vlr * tanhf(td_error);

        float action_delta = params.alr * ((1.0f - mimic) * tanhf(td_error) + mimic);

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];

            int diam = vld.radius * 2 + 1;

            // Projection
            Float2 h_to_v = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hidden_size.x),
                static_cast<float>(vld.size.y) / static_cast<float>(hidden_size.y));

            Int2 visible_center = project(column_pos, h_to_v);

            // Lower corner
            Int2 field_lower_bound(visible_center.x - vld.radius, visible_center.y - vld.radius);

            // Bounds of receptive field, clamped to input size
            Int2 iter_lower_bound(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
            Int2 iter_upper_bound(min(vld.size.x - 1, visible_center.x + vld.radius), min(vld.size.y - 1, visible_center.y + vld.radius));

            for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                    int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                    int in_ci_prev = vl.input_cis_prev[visible_column_index];

                    Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                    for (int vc = 0; vc < vld.size.z; vc++) {
                        int value_wi = offset.y + diam * (offset.x + diam * (vc + vld.size.z * hidden_column_index));
                        int action_wi_start = hidden_size.z * value_wi;

                        if (vc == in_ci_prev)
                            vl.value_traces[value_wi] += 1.0f;

                        vl.value_weights[value_wi] += value_delta * vl.value_traces[value_wi];

                        vl.value_traces[value_wi] *= params.trace_decay;

                        for (int hc = 0; hc < hidden_size.z; hc++) {
                            int hidden_cell_index = hc + hidden_cells_start;

                            int action_wi = hc + action_wi_start;

                            vl.action_traces[action_wi] += ((hc == target_ci) - hidden_acts_prev[hidden_cell_index]) * (vc == in_ci_prev);

                            vl.action_weights[action_wi] += action_delta * vl.action_traces[action_wi];

                            vl.action_traces[action_wi] *= params.trace_decay;
                        }
                    }
                }
        }
    }
}

void Actor::init_random(
    const Int3 &hidden_size,
    const Array<Visible_Layer_Desc> &visible_layer_descs
) {
    this->visible_layer_descs = visible_layer_descs;

    this->hidden_size = hidden_size;

    visible_layers.resize(visible_layer_descs.size());

    // Pre-compute dimensions
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;

    // Create layers
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;
        int num_visible_cells = num_visible_columns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        // Create weight matrix for this visible layer
        vl.action_weights.resize(num_hidden_cells * area * vld.size.z);
        vl.value_weights.resize(num_hidden_columns * area * vld.size.z);

        for (int i = 0; i < vl.action_weights.size(); i++)
            vl.action_weights[i] = randf(-0.01f, 0.01f);

        for (int i = 0; i < vl.value_weights.size(); i++)
            vl.value_weights[i] = randf(-0.01f, 0.01f);

        vl.action_traces = Float_Buffer(vl.action_weights.size(), 0.0f);
        vl.value_traces = Float_Buffer(vl.value_weights.size(), 0.0f);

        vl.input_cis_prev = Int_Buffer(num_visible_columns, 0);
    }

    hidden_cis = Int_Buffer(num_hidden_columns, 0);

    hidden_acts = Float_Buffer(num_hidden_cells, 0.0f);
    hidden_acts_prev = Float_Buffer(num_hidden_cells, 0.0f);
    hidden_values = Float_Buffer(num_hidden_columns, 0.0f);
}

void Actor::step(
    const Array<Int_Buffer_View> &input_cis,
    Int_Buffer_View hidden_target_cis,
    float reward,
    float mimic,
    bool learn_enabled,
    const Params &params
) {
    int num_hidden_columns = hidden_size.x * hidden_size.y;

    PARALLEL_FOR
    for (int i = 0; i < num_hidden_columns; i++)
        forward(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, hidden_target_cis, reward, mimic, learn_enabled, params);

    // copy to prevs
    hidden_acts_prev = hidden_acts;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        vl.input_cis_prev = input_cis[vli];
    }
}

void Actor::clear_state() {
    hidden_cis.fill(0);
    hidden_acts.fill(0.0f);
    hidden_acts_prev.fill(0.0f);
    hidden_values.fill(0.0f);

    for (int vli = 0; vli < visible_layers.size(); vli++)
        visible_layers[vli].input_cis_prev.fill(0);
}

int Actor::size() const {
    int size = sizeof(Int3) + hidden_cis.size() * sizeof(int) + 2 * hidden_acts.size() * sizeof(float) + hidden_values.size() * sizeof(float) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        size += sizeof(Visible_Layer_Desc) + 2 * vl.action_weights.size() * sizeof(float) + 2 * vl.value_weights.size() * sizeof(float) + vl.input_cis_prev.size() * sizeof(int);
    }

    return size;
}

int Actor::state_size() const {
    int size = hidden_cis.size() * sizeof(int) + 2 * hidden_acts.size() * sizeof(float) + hidden_values.size() * sizeof(float);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += vl.input_cis_prev.size() * sizeof(int);
    }

    return size;
}

void Actor::write(
    Stream_Writer &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hidden_size), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hidden_acts[0]), hidden_acts.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&hidden_acts_prev[0]), hidden_acts_prev.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&hidden_values[0]), hidden_values.size() * sizeof(float));

    int num_visible_cells_layers = visible_layers.size();

    writer.write(reinterpret_cast<const void*>(&num_visible_cells_layers), sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(Visible_Layer_Desc));

        writer.write(reinterpret_cast<const void*>(&vl.action_weights[0]), vl.action_weights.size() * sizeof(float));
        writer.write(reinterpret_cast<const void*>(&vl.value_weights[0]), vl.value_weights.size() * sizeof(float));
        writer.write(reinterpret_cast<const void*>(&vl.action_traces[0]), vl.action_traces.size() * sizeof(float));
        writer.write(reinterpret_cast<const void*>(&vl.value_traces[0]), vl.value_traces.size() * sizeof(float));

        writer.write(reinterpret_cast<const void*>(&vl.input_cis_prev[0]), vl.input_cis_prev.size() * sizeof(int));
    }
}

void Actor::read(
    Stream_Reader &reader
) {
    reader.read(reinterpret_cast<void*>(&hidden_size), sizeof(Int3));

    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;
    
    hidden_cis.resize(num_hidden_columns);
    hidden_acts.resize(num_hidden_cells);
    hidden_acts_prev.resize(num_hidden_cells);
    hidden_values.resize(num_hidden_columns);

    reader.read(reinterpret_cast<void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hidden_acts[0]), hidden_acts.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&hidden_acts_prev[0]), hidden_acts_prev.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&hidden_values[0]), hidden_values.size() * sizeof(float));

    int num_visible_cells_layers = visible_layers.size();

    reader.read(reinterpret_cast<void*>(&num_visible_cells_layers), sizeof(int));

    visible_layers.resize(num_visible_cells_layers);
    visible_layer_descs.resize(num_visible_cells_layers);
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        Visible_Layer_Desc &vld = visible_layer_descs[vli];

        reader.read(reinterpret_cast<void*>(&vld), sizeof(Visible_Layer_Desc));

        int num_visible_columns = vld.size.x * vld.size.y;
        int num_visible_cells = num_visible_columns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.action_weights.resize(num_hidden_cells * area * vld.size.z);
        vl.value_weights.resize(num_hidden_columns * area * vld.size.z);
        vl.action_traces.resize(vl.action_weights.size());
        vl.value_traces.resize(vl.value_weights.size());

        reader.read(reinterpret_cast<void*>(&vl.action_weights[0]), vl.action_weights.size() * sizeof(float));
        reader.read(reinterpret_cast<void*>(&vl.value_weights[0]), vl.value_weights.size() * sizeof(float));
        reader.read(reinterpret_cast<void*>(&vl.action_traces[0]), vl.action_traces.size() * sizeof(float));
        reader.read(reinterpret_cast<void*>(&vl.value_traces[0]), vl.value_traces.size() * sizeof(float));

        vl.input_cis_prev.resize(num_visible_columns);

        reader.read(reinterpret_cast<void*>(&vl.input_cis_prev[0]), vl.input_cis_prev.size() * sizeof(int));
    }
}

void Actor::write_state(
    Stream_Writer &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hidden_acts[0]), hidden_acts.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&hidden_acts_prev[0]), hidden_acts_prev.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&hidden_values[0]), hidden_values.size() * sizeof(float));

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        writer.write(reinterpret_cast<const void*>(&vl.input_cis_prev[0]), vl.input_cis_prev.size() * sizeof(int));
    }
}

void Actor::read_state(
    Stream_Reader &reader
) {
    reader.read(reinterpret_cast<void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hidden_acts[0]), hidden_acts.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&hidden_acts_prev[0]), hidden_acts_prev.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&hidden_values[0]), hidden_values.size() * sizeof(float));

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(reinterpret_cast<void*>(&vl.input_cis_prev[0]), vl.input_cis_prev.size() * sizeof(int));
    }
}
