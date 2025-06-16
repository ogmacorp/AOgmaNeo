// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2025 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "actor.h"

using namespace aon;

void Actor::forward(
    const Int2 &column_pos,
    const Array<Int_Buffer_View> &input_cis,
    Int_Buffer_View hidden_target_cis_prev,
    bool learn_enabled,
    float reward,
    float mimic,
    unsigned long* state,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    int target_ci = hidden_target_cis_prev[hidden_column_index];

    float value_prev = hidden_values[hidden_column_index];

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + dendrites_start;

            dendrite_acts[dendrite_index] = 0.0f;
        }
    }

    float value = 0.0f;
    int count = 0;

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

        count += (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1);

        Int_Buffer_View vl_input_cis = input_cis[vli];

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci = vl_input_cis[visible_column_index];

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi_base = offset.y + diam * (offset.x + diam * (in_ci + vld.size.z * hidden_column_index));

                value += vl.value_weights[wi_base];

                int wi_start_partial = hidden_size.z * wi_base;

                for (int hc = 0; hc < hidden_size.z; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

                    int wi_start = num_dendrites_per_cell * (hc + wi_start_partial);

                    for (int di = 0; di < num_dendrites_per_cell; di++) {
                        int dendrite_index = di + dendrites_start;

                        int wi = di + wi_start;

                        dendrite_acts[dendrite_index] += vl.policy_weights[wi];
                    }
                }
            }
    }

    const int half_num_dendrites_per_cell = num_dendrites_per_cell / 2;
    const float dendrite_scale = sqrtf(1.0f / count);
    const float activation_scale = sqrtf(1.0f / num_dendrites_per_cell);

    value *= dendrite_scale;

    hidden_values[hidden_column_index] = value;

    int max_index = 0;
    float max_activation = limit_min;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

        float sum = 0.0f;

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + dendrites_start;

            float act = dendrite_acts[dendrite_index] * dendrite_scale;

            dendrite_acts[dendrite_index] = sigmoidf(act); // store derivative

            sum += softplusf(act) * ((di >= half_num_dendrites_per_cell) * 2.0f - 1.0f);
        }

        sum *= activation_scale;

        hidden_acts[hidden_cell_index] = sum;

        if (sum > max_activation) {
            max_activation = sum;
            max_index = hc;
        }
    }

    hidden_cis[hidden_column_index] = max_index;

    if (learn_enabled) {
        float td_error = reward + params.discount * value - value_prev;

        float value_rate = min(1.0f, max(-1.0f, params.vlr * td_error));
        float policy_rate = min(1.0f, max(-1.0f, params.plr * td_error));

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

                    int in_ci_prev = vl.input_cis_prev[visible_column_index];

                    Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                    // regular weights update
                    for (int vc = 0; vc < vld.size.z; vc++) {
                        int wi_base = offset.y + diam * (offset.x + diam * (vc + vld.size.z * hidden_column_index));

                        if (vc == in_ci_prev)
                            vl.value_traces[wi_base] = 1.0f;

                        vl.value_weights[wi_base] += value_rate * vl.value_traces[wi_base];

                        vl.value_traces[wi_base] *= params.trace_decay;

                        int wi_start_partial = hidden_size.z * wi_base;

                        for (int hc = 0; hc < hidden_size.z; hc++) {
                            int hidden_cell_index = hc + hidden_cells_start;

                            int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

                            int wi_start = num_dendrites_per_cell * (hc + wi_start_partial);

                            for (int di = 0; di < num_dendrites_per_cell; di++) {
                                int dendrite_index = di + dendrites_start;

                                int wi = di + wi_start;

                                if (vc == in_ci_prev && hc == target_ci)
                                    vl.policy_traces[wi] = max(vl.policy_traces[wi], dendrite_acts_prev[dendrite_index]);

                                vl.policy_weights[wi] += ((di >= half_num_dendrites_per_cell) * 2.0f - 1.0f) * (policy_rate * vl.policy_traces[wi] + mimic * ((hc == target_ci) - hidden_acts_prev[hidden_cell_index]) * dendrite_acts_prev[dendrite_index] * (vc == in_ci_prev));

                                vl.policy_traces[wi] *= params.trace_decay;
                            }
                        }
                    }
                }
        }
    }
}

void Actor::init_random(
    const Int3 &hidden_size,
    int num_dendrites_per_cell,
    const Array<Visible_Layer_Desc> &visible_layer_descs
) {
    this->visible_layer_descs = visible_layer_descs;

    this->hidden_size = hidden_size;
    this->num_dendrites_per_cell = num_dendrites_per_cell;

    visible_layers.resize(visible_layer_descs.size());

    // pre-compute dimensions
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;
    int num_dendrites = num_hidden_cells * num_dendrites_per_cell;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;
        int num_visible_cells = num_visible_columns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.value_weights.resize(num_hidden_columns * area * vld.size.z);

        for (int i = 0; i < vl.value_weights.size(); i++)
            vl.value_weights[i] = randf(-init_weight_noisef, init_weight_noisef);

        vl.value_traces = Float_Buffer(vl.value_weights.size(), 0.0f);

        vl.policy_weights.resize(num_dendrites * area * vld.size.z);

        for (int i = 0; i < vl.policy_weights.size(); i++)
            vl.policy_weights[i] = randf(-init_weight_noisef, init_weight_noisef);

        vl.policy_traces = Float_Buffer(vl.policy_weights.size(), 0.0f);
        
        vl.input_cis_prev = Int_Buffer(num_visible_columns, 0);
    }

    hidden_cis = Int_Buffer(num_hidden_columns, 0);

    dendrite_acts.resize(num_dendrites);
    dendrite_acts_prev = Float_Buffer(num_dendrites, 0.0f);

    hidden_acts.resize(num_hidden_cells);
    hidden_acts_prev = Float_Buffer(num_hidden_cells, 0.0f);

    hidden_values = Float_Buffer(num_hidden_columns, 0.0f);
}

void Actor::step(
    const Array<Int_Buffer_View> &input_cis,
    Int_Buffer_View hidden_target_cis_prev,
    bool learn_enabled,
    float reward,
    float mimic,
    const Params &params
) {
    int num_hidden_columns = hidden_size.x * hidden_size.y;

    unsigned int base_state = rand();

    // forward kernel
    PARALLEL_FOR
    for (int i = 0; i < num_hidden_columns; i++) {
        unsigned long state = rand_get_state(base_state + i * rand_subseed_offset);

        forward(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, hidden_target_cis_prev, learn_enabled, reward, mimic, &state, params);
    }

    // update prevs
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        vl.input_cis_prev = input_cis[vli];
    }

    hidden_acts_prev = hidden_acts;
    dendrite_acts_prev = dendrite_acts;
}

void Actor::clear_state() {
    hidden_cis.fill(0);
    hidden_acts_prev.fill(0.0f);
    dendrite_acts_prev.fill(0.0f);

    hidden_values.fill(0.0f);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        vl.value_traces.fill(0.0f);
        vl.policy_traces.fill(0.0f);

        vl.input_cis_prev.fill(0);
    }
}

long Actor::size() const {
    long size = sizeof(Int3) + 2 * sizeof(int) + hidden_cis.size() * sizeof(int) + hidden_acts_prev.size() * sizeof(float) + dendrite_acts_prev.size() * sizeof(float) + hidden_values.size() * sizeof(float) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        size += sizeof(Visible_Layer_Desc) + 2 * vl.value_weights.size() * sizeof(float) + 2 * vl.policy_weights.size() * sizeof(float) + vl.input_cis_prev.size() * sizeof(int);
    }

    return size;
}

long Actor::state_size() const {
    long size = hidden_cis.size() * sizeof(int) + hidden_acts_prev.size() * sizeof(float) + dendrite_acts_prev.size() * sizeof(float) + hidden_values.size() * sizeof(float);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += vl.value_traces.size() * sizeof(float) + vl.policy_traces.size() * sizeof(float) + vl.input_cis_prev.size() * sizeof(int);
    }

    return size;
}

long Actor::weights_size() const {
    long size = 0;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += vl.value_weights.size() * sizeof(float) + vl.policy_weights.size() * sizeof(float);
    }

    return size;
}

void Actor::write(
    Stream_Writer &writer
) const {
    writer.write(&hidden_size, sizeof(Int3));
    writer.write(&num_dendrites_per_cell, sizeof(int));

    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    writer.write(&hidden_acts_prev[0], hidden_acts_prev.size() * sizeof(float));
    writer.write(&dendrite_acts_prev[0], dendrite_acts_prev.size() * sizeof(float));
    writer.write(&hidden_values[0], hidden_values.size() * sizeof(float));

    int num_visible_layers = visible_layers.size();

    writer.write(&num_visible_layers, sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(&vld, sizeof(Visible_Layer_Desc));

        writer.write(&vl.value_weights[0], vl.value_weights.size() * sizeof(float));
        writer.write(&vl.value_traces[0], vl.value_traces.size() * sizeof(float));

        writer.write(&vl.policy_weights[0], vl.policy_weights.size() * sizeof(float));
        writer.write(&vl.policy_traces[0], vl.policy_traces.size() * sizeof(float));

        writer.write(&vl.input_cis_prev[0], vl.input_cis_prev.size() * sizeof(int));
    }
}

void Actor::read(
    Stream_Reader &reader
) {
    reader.read(&hidden_size, sizeof(Int3));
    reader.read(&num_dendrites_per_cell, sizeof(int));

    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;
    int num_dendrites = num_hidden_cells * num_dendrites_per_cell;
    
    hidden_cis.resize(num_hidden_columns);
    hidden_acts_prev.resize(num_hidden_cells);
    dendrite_acts_prev.resize(num_dendrites);
    hidden_values.resize(num_hidden_columns);

    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    reader.read(&hidden_acts_prev[0], hidden_acts_prev.size() * sizeof(float));
    reader.read(&dendrite_acts_prev[0], dendrite_acts_prev.size() * sizeof(float));
    reader.read(&hidden_values[0], hidden_values.size() * sizeof(float));

    hidden_acts.resize(num_hidden_cells);
    dendrite_acts.resize(num_dendrites);

    int num_visible_layers;

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

        vl.value_weights.resize(num_hidden_columns * area * vld.size.z);
        vl.value_traces.resize(vl.value_weights.size());
        vl.policy_weights.resize(num_dendrites * area * vld.size.z);
        vl.policy_traces.resize(vl.policy_weights.size());

        reader.read(&vl.value_weights[0], vl.value_weights.size() * sizeof(float));
        reader.read(&vl.value_traces[0], vl.value_traces.size() * sizeof(float));
        reader.read(&vl.policy_weights[0], vl.policy_weights.size() * sizeof(float));
        reader.read(&vl.policy_traces[0], vl.policy_traces.size() * sizeof(float));

        vl.input_cis_prev.resize(num_visible_columns);

        reader.read(&vl.input_cis_prev[0], vl.input_cis_prev.size() * sizeof(int));
    }
}

void Actor::write_state(
    Stream_Writer &writer
) const {
    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    writer.write(&hidden_acts_prev[0], hidden_acts_prev.size() * sizeof(float));
    writer.write(&dendrite_acts_prev[0], dendrite_acts_prev.size() * sizeof(float));
    writer.write(&hidden_values[0], hidden_values.size() * sizeof(float));

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        writer.write(&vl.value_traces[0], vl.value_traces.size() * sizeof(float));
        writer.write(&vl.policy_traces[0], vl.policy_traces.size() * sizeof(float));

        writer.write(&vl.input_cis_prev[0], vl.input_cis_prev.size() * sizeof(int));
    }
}

void Actor::read_state(
    Stream_Reader &reader
) {
    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    reader.read(&hidden_acts_prev[0], hidden_acts_prev.size() * sizeof(float));
    reader.read(&dendrite_acts_prev[0], dendrite_acts_prev.size() * sizeof(float));
    reader.read(&hidden_values[0], hidden_values.size() * sizeof(float));

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(&vl.value_traces[0], vl.value_traces.size() * sizeof(float));
        reader.read(&vl.policy_traces[0], vl.policy_traces.size() * sizeof(float));

        reader.read(&vl.input_cis_prev[0], vl.input_cis_prev.size() * sizeof(int));
    }
}

void Actor::write_weights(
    Stream_Writer &writer
) const {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        writer.write(&vl.value_weights[0], vl.value_weights.size() * sizeof(float));
        writer.write(&vl.policy_weights[0], vl.policy_weights.size() * sizeof(float));
    }
}

void Actor::read_weights(
    Stream_Reader &reader
) {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(&vl.value_weights[0], vl.value_weights.size() * sizeof(float));
        reader.read(&vl.policy_weights[0], vl.policy_weights.size() * sizeof(float));
    }
}

void Actor::merge(
    const Array<Actor*> &actors,
    Merge_Mode mode
) {
    switch (mode) {
    case merge_random:
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];

            for (int i = 0; i < vl.value_weights.size(); i++) {
                int d = rand() % actors.size();                

                vl.value_weights[i] = actors[d]->visible_layers[vli].value_weights[i];
            }

            for (int i = 0; i < vl.policy_weights.size(); i++) {
                int d = rand() % actors.size();                

                vl.policy_weights[i] = actors[d]->visible_layers[vli].policy_weights[i];
            }
        }

        break;
    case merge_average:
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];

            for (int i = 0; i < vl.value_weights.size(); i++) {
                float total = 0.0f;

                for (int d = 0; d < actors.size(); d++)
                    total += actors[d]->visible_layers[vli].value_weights[i];

                vl.value_weights[i] = total / actors.size();
            }

            for (int i = 0; i < vl.policy_weights.size(); i++) {
                float total = 0.0f;

                for (int d = 0; d < actors.size(); d++)
                    total += actors[d]->visible_layers[vli].policy_weights[i];

                vl.policy_weights[i] = total / actors.size();
            }
        }

        break;
    }
}
