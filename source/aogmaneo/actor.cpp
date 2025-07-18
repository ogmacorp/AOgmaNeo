// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2025 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "actor.h"
#include <iostream>

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
    int value_cells_start = hidden_column_index * value_size;

    float value_prev = hidden_values[hidden_column_index];

    for (int vac = 0; vac < value_size; vac++) {
        int value_cell_index = vac + value_cells_start;

        int value_dendrites_start = value_num_dendrites_per_cell * value_cell_index;

        for (int di = 0; di < value_num_dendrites_per_cell; di++) {
            int dendrite_index = di + value_dendrites_start;

            value_dendrite_acts[dendrite_index] = 0.0f;
        }
    }

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int policy_dendrites_start = policy_num_dendrites_per_cell * hidden_cell_index;

        for (int di = 0; di < policy_num_dendrites_per_cell; di++) {
            int dendrite_index = di + policy_dendrites_start;

            policy_dendrite_acts[dendrite_index] = 0.0f;
        }
    }

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

                int wi_start_partial = offset.y + diam * (offset.x + diam * (in_ci + vld.size.z * hidden_column_index));
                int wi_value_partial = value_size * wi_start_partial;
                int wi_policy_partial = hidden_size.z * wi_start_partial;

                for (int vac = 0; vac < value_size; vac++) {
                    int value_cell_index = vac + value_cells_start;

                    int value_dendrites_start = value_num_dendrites_per_cell * value_cell_index;

                    int wi_start = value_num_dendrites_per_cell * (vac + wi_value_partial);

                    for (int di = 0; di < value_num_dendrites_per_cell; di++) {
                        int dendrite_index = di + value_dendrites_start;

                        int wi = di + wi_start;

                        value_dendrite_acts[dendrite_index] += vl.value_weights[wi];
                    }
                }

                for (int hc = 0; hc < hidden_size.z; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int policy_dendrites_start = policy_num_dendrites_per_cell * hidden_cell_index;

                    int wi_start = policy_num_dendrites_per_cell * (hc + wi_policy_partial);

                    for (int di = 0; di < policy_num_dendrites_per_cell; di++) {
                        int dendrite_index = di + policy_dendrites_start;

                        int wi = di + wi_start;

                        policy_dendrite_acts[dendrite_index] += vl.policy_weights[wi];
                    }
                }
            }
    }

    const int half_value_num_dendrites_per_cell = value_num_dendrites_per_cell / 2;
    const int half_policy_num_dendrites_per_cell = policy_num_dendrites_per_cell / 2;
    const float dendrite_scale = sqrtf(1.0f / count);
    const float value_activation_scale = sqrtf(1.0f / value_num_dendrites_per_cell);
    const float policy_activation_scale = sqrtf(1.0f / policy_num_dendrites_per_cell);

    int max_value_index = 0;
    float max_value_activation = limit_min;

    // value
    for (int vac = 0; vac < value_size; vac++) {
        int value_cell_index = vac + value_cells_start;

        int value_dendrites_start = value_num_dendrites_per_cell * value_cell_index;

        float activation = 0.0f;

        for (int di = 0; di < value_num_dendrites_per_cell; di++) {
            int dendrite_index = di + value_dendrites_start;

            float act = value_dendrite_acts[dendrite_index] * dendrite_scale;

            value_dendrite_acts[dendrite_index] = sigmoidf(act); // store derivative

            activation += softplusf(act) * ((di >= half_value_num_dendrites_per_cell) * 2.0f - 1.0f);
        }

        activation *= value_activation_scale;

        hidden_value_acts[value_cell_index] = activation;

        if (activation > max_value_activation) {
            max_value_activation = activation;
            max_value_index = vac;
        }
    }

    float smooth_max_value_index;

    // softmax
    {
        float total = 0.0f;

        for (int vac = 0; vac < value_size; vac++) {
            int value_cell_index = vac + value_cells_start;
        
            hidden_value_acts[value_cell_index] = expf(hidden_value_acts[value_cell_index] - max_value_activation);

            total += hidden_value_acts[value_cell_index];
        }

        float total_inv = 1.0f / max(limit_small, total);

        smooth_max_value_index = 0.0f;

        for (int vac = 0; vac < value_size; vac++) {
            int value_cell_index = vac + value_cells_start;

            hidden_value_acts[value_cell_index] *= total_inv;

            smooth_max_value_index += hidden_value_acts[value_cell_index] * vac;
        }
    }

    float value = symexpf((smooth_max_value_index / static_cast<float>(value_size - 1) * 2.0f - 1.0f) * params.value_range);

    hidden_values[hidden_column_index] = value;

    float max_activation = limit_min;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int dendrites_start = policy_num_dendrites_per_cell * hidden_cell_index;

        float activation = 0.0f;

        for (int di = 0; di < policy_num_dendrites_per_cell; di++) {
            int dendrite_index = di + dendrites_start;

            float act = policy_dendrite_acts[dendrite_index] * dendrite_scale;

            policy_dendrite_acts[dendrite_index] = sigmoidf(act); // store derivative

            activation += softplusf(act) * ((di >= half_policy_num_dendrites_per_cell) * 2.0f - 1.0f);
        }

        activation *= policy_activation_scale;

        hidden_policy_acts[hidden_cell_index] = activation;

        max_activation = max(max_activation, activation);
    }

    // softmax
    {
        float total = 0.0f;

        for (int hc = 0; hc < hidden_size.z; hc++) {
            int hidden_cell_index = hc + hidden_cells_start;
        
            hidden_policy_acts[hidden_cell_index] = expf(hidden_policy_acts[hidden_cell_index] - max_activation);

            total += hidden_policy_acts[hidden_cell_index];
        }

        float total_inv = 1.0f / max(limit_small, total);

        for (int hc = 0; hc < hidden_size.z; hc++) {
            int hidden_cell_index = hc + hidden_cells_start;

            hidden_policy_acts[hidden_cell_index] *= total_inv;
        }
    }

    float cusp = randf(state);

    int select_index = 0;
    float sum_so_far = 0.0f;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        sum_so_far += hidden_policy_acts[hidden_cell_index];

        if (sum_so_far >= cusp) {
            select_index = hc;

            break;
        }
    }
    
    hidden_cis[hidden_column_index] = select_index;

    if (learn_enabled) {
        int target_ci = hidden_target_cis_prev[hidden_column_index];

        float new_value = reward + params.discount * value;

        std::cout << value << std::endl;
        float td_error = new_value - value_prev;

        hidden_td_scales[hidden_column_index] = max(hidden_td_scales[hidden_column_index] * params.td_scale_decay, abs(td_error));

        float scaled_td_error = td_error / max(limit_small, hidden_td_scales[hidden_column_index]);

        float value_rate = params.vlr * td_error;
        float policy_rate = params.plr * scaled_td_error;

        float smooth_new_value_index_plus = min(value_size - 1.0f, (min(1.0f, max(0.0f, symlogf(new_value) / params.value_range * 0.5f + 0.5f)) * (value_size - 1) + 1.0f));

        for (int vac = 0; vac < value_size; vac++) {
            int value_cell_index = vac + value_cells_start;

            int value_dendrites_start = value_num_dendrites_per_cell * value_cell_index;

            float target = max(0.0f, 1.0f - abs(vac - smooth_new_value_index_plus));

            float error = (target - hidden_value_acts[value_cell_index]);

            for (int di = 0; di < value_num_dendrites_per_cell; di++) {
                int dendrite_index = di + value_dendrites_start;

                // re-use as deltas
                value_dendrite_acts_prev[dendrite_index] = error * ((di >= half_value_num_dendrites_per_cell) * 2.0f - 1.0f) * value_dendrite_acts_prev[dendrite_index];
            }
        }

        for (int hc = 0; hc < hidden_size.z; hc++) {
            int hidden_cell_index = hc + hidden_cells_start;

            int dendrites_start = policy_num_dendrites_per_cell * hidden_cell_index;

            float error = ((hc == target_ci) - hidden_policy_acts_prev[hidden_cell_index]);

            for (int di = 0; di < policy_num_dendrites_per_cell; di++) {
                int dendrite_index = di + dendrites_start;

                // re-use acts as partial derivatives
                policy_dendrite_acts_prev[dendrite_index] = error * ((di >= half_policy_num_dendrites_per_cell) * 2.0f - 1.0f) * policy_dendrite_acts_prev[dendrite_index];
            }
        }

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
                        int wi_start_partial = offset.y + diam * (offset.x + diam * (vc + vld.size.z * hidden_column_index));
                        int wi_value_partial = value_size * wi_start_partial;
                        int wi_policy_partial = hidden_size.z * wi_start_partial;

                        for (int vac = 0; vac < value_size; vac++) {
                            int value_cell_index = vac + value_cells_start;

                            int value_dendrites_start = value_num_dendrites_per_cell * value_cell_index;

                            int wi_start = value_num_dendrites_per_cell * (vac + wi_value_partial);

                            for (int di = 0; di < value_num_dendrites_per_cell; di++) {
                                int dendrite_index = di + value_dendrites_start;

                                int wi = di + wi_start;

                                vl.value_traces[wi] += params.trace_rate * ((vc == in_ci_prev) * value_dendrite_acts_prev[dendrite_index] - vl.value_traces[wi]);

                                vl.value_weights[wi] += value_rate * vl.value_traces[wi];
                            }
                        }

                        for (int hc = 0; hc < hidden_size.z; hc++) {
                            int hidden_cell_index = hc + hidden_cells_start;

                            int policy_dendrites_start = policy_num_dendrites_per_cell * hidden_cell_index;

                            int wi_start = policy_num_dendrites_per_cell * (hc + wi_policy_partial);

                            for (int di = 0; di < policy_num_dendrites_per_cell; di++) {
                                int dendrite_index = di + policy_dendrites_start;

                                int wi = di + wi_start;

                                vl.policy_traces[wi] += ((vc == in_ci_prev) * policy_dendrite_acts_prev[dendrite_index] - vl.policy_traces[wi]);

                                vl.policy_weights[wi] += policy_rate * vl.policy_traces[wi];
                            }
                        }
                    }
                }
        }
    }
}

void Actor::init_random(
    const Int3 &hidden_size,
    int value_size,
    int value_num_dendrites_per_cell,
    int policy_num_dendrites_per_cell,
    const Array<Visible_Layer_Desc> &visible_layer_descs
) {
    this->visible_layer_descs = visible_layer_descs;

    // pre-compute dimensions
    this->hidden_size = hidden_size;
    this->value_size = value_size;
    this->value_num_dendrites_per_cell = value_num_dendrites_per_cell;
    this->policy_num_dendrites_per_cell = policy_num_dendrites_per_cell;

    visible_layers.resize(visible_layer_descs.size());

    // pre-compute dimensions
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;
    int num_value_cells = num_hidden_columns * value_size;
    int value_num_dendrites = num_value_cells * value_num_dendrites_per_cell;
    int policy_num_dendrites = num_hidden_cells * policy_num_dendrites_per_cell;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;
        int num_visible_cells = num_visible_columns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.value_weights.resize(value_num_dendrites * area * vld.size.z);

        for (int i = 0; i < vl.value_weights.size(); i++)
            vl.value_weights[i] = randf(-init_weight_noisef, init_weight_noisef);

        vl.policy_weights.resize(policy_num_dendrites * area * vld.size.z);

        for (int i = 0; i < vl.policy_weights.size(); i++)
            vl.policy_weights[i] = randf(-init_weight_noisef, init_weight_noisef);

        vl.value_traces = Float_Buffer(vl.value_weights.size(), 0.0f);
        vl.policy_traces = Float_Buffer(vl.policy_weights.size(), 0.0f);

        vl.input_cis_prev = Int_Buffer(num_visible_columns, 0);
    }

    hidden_cis = Int_Buffer(num_hidden_columns, 0);

    hidden_values = Float_Buffer(num_hidden_columns, 0.0f);

    hidden_td_scales = Float_Buffer(num_hidden_cells, 0.0f);

    value_dendrite_acts.resize(value_num_dendrites);
    policy_dendrite_acts.resize(policy_num_dendrites);

    value_dendrite_acts_prev = Float_Buffer(value_num_dendrites, 0.0f);
    policy_dendrite_acts_prev = Float_Buffer(policy_num_dendrites, 0.0f);

    hidden_value_acts.resize(num_value_cells);
    hidden_policy_acts.resize(num_hidden_cells);

    hidden_value_acts_prev = Float_Buffer(num_value_cells, 0.0f);
    hidden_policy_acts_prev = Float_Buffer(num_hidden_cells, 0.0f);
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

    // forward kernel
    unsigned int base_state = rand();

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

    hidden_value_acts_prev = hidden_value_acts;
    hidden_policy_acts_prev = hidden_policy_acts;
    value_dendrite_acts_prev = value_dendrite_acts;
    policy_dendrite_acts_prev = policy_dendrite_acts;
}

void Actor::clear_state() {
    hidden_cis.fill(0);
    hidden_value_acts_prev.fill(0.0f);
    hidden_policy_acts_prev.fill(0.0f);
    value_dendrite_acts_prev.fill(0.0f);
    policy_dendrite_acts_prev.fill(0.0f);

    hidden_values.fill(0.0f);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        vl.value_traces.fill(0.0f);
        vl.policy_traces.fill(0.0f);

        vl.input_cis_prev.fill(0);
    }
}

long Actor::size() const {
    long size = sizeof(Int3) + 3 * sizeof(int) + hidden_cis.size() * sizeof(int) +
        hidden_value_acts_prev.size() * sizeof(float) + hidden_policy_acts_prev.size() * sizeof(float) +
        value_dendrite_acts_prev.size() * sizeof(float) + policy_dendrite_acts_prev.size() * sizeof(float) +
        hidden_values.size() * sizeof(float) + hidden_td_scales.size() * sizeof(float) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        size += sizeof(Visible_Layer_Desc) + 2 * vl.value_weights.size() * sizeof(float) + 2 * vl.policy_weights.size() * sizeof(float) + vl.input_cis_prev.size() * sizeof(int);
    }

    return size;
}

long Actor::state_size() const {
    long size = hidden_cis.size() * sizeof(int) +
        hidden_value_acts_prev.size() * sizeof(float) + hidden_policy_acts_prev.size() * sizeof(float) +
        value_dendrite_acts_prev.size() * sizeof(float) + policy_dendrite_acts_prev.size() * sizeof(float) +
        hidden_values.size() * sizeof(float);

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
    writer.write(&value_size, sizeof(int));
    writer.write(&value_num_dendrites_per_cell, sizeof(int));
    writer.write(&policy_num_dendrites_per_cell, sizeof(int));

    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    writer.write(&hidden_value_acts_prev[0], hidden_value_acts_prev.size() * sizeof(float));
    writer.write(&hidden_policy_acts_prev[0], hidden_policy_acts_prev.size() * sizeof(float));
    writer.write(&value_dendrite_acts_prev[0], value_dendrite_acts_prev.size() * sizeof(float));
    writer.write(&policy_dendrite_acts_prev[0], policy_dendrite_acts_prev.size() * sizeof(float));
    writer.write(&hidden_values[0], hidden_values.size() * sizeof(float));
    writer.write(&hidden_td_scales[0], hidden_td_scales.size() * sizeof(float));

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
    reader.read(&value_size, sizeof(int));
    reader.read(&value_num_dendrites_per_cell, sizeof(int));
    reader.read(&policy_num_dendrites_per_cell, sizeof(int));

    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;
    int num_value_cells = num_hidden_columns * value_size;
    int value_num_dendrites = num_value_cells * value_num_dendrites_per_cell;
    int policy_num_dendrites = num_hidden_cells * policy_num_dendrites_per_cell;
    
    hidden_cis.resize(num_hidden_columns);
    hidden_value_acts_prev.resize(num_value_cells);
    hidden_policy_acts_prev.resize(num_hidden_cells);
    value_dendrite_acts_prev.resize(value_num_dendrites);
    policy_dendrite_acts_prev.resize(policy_num_dendrites);
    hidden_values.resize(num_hidden_columns);
    hidden_td_scales.resize(num_hidden_columns);

    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    reader.read(&hidden_value_acts_prev[0], hidden_value_acts_prev.size() * sizeof(float));
    reader.read(&hidden_policy_acts_prev[0], hidden_policy_acts_prev.size() * sizeof(float));
    reader.read(&value_dendrite_acts[0], value_dendrite_acts_prev.size() * sizeof(float));
    reader.read(&policy_dendrite_acts_prev[0], policy_dendrite_acts_prev.size() * sizeof(float));
    reader.read(&hidden_values[0], hidden_values.size() * sizeof(float));
    reader.read(&hidden_td_scales[0], hidden_td_scales.size() * sizeof(float));

    hidden_value_acts.resize(num_value_cells);
    hidden_policy_acts.resize(num_hidden_cells);
    value_dendrite_acts.resize(value_num_dendrites);
    policy_dendrite_acts.resize(policy_num_dendrites);

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

        vl.value_weights.resize(value_num_dendrites * area * vld.size.z);
        vl.value_traces.resize(vl.value_weights.size());
        vl.policy_weights.resize(policy_num_dendrites * area * vld.size.z);
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
    writer.write(&hidden_value_acts_prev[0], hidden_value_acts_prev.size() * sizeof(float));
    writer.write(&hidden_policy_acts_prev[0], hidden_policy_acts_prev.size() * sizeof(float));
    writer.write(&value_dendrite_acts_prev[0], value_dendrite_acts_prev.size() * sizeof(float));
    writer.write(&policy_dendrite_acts_prev[0], policy_dendrite_acts_prev.size() * sizeof(float));
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
    reader.read(&hidden_value_acts_prev[0], hidden_value_acts_prev.size() * sizeof(float));
    reader.read(&hidden_policy_acts_prev[0], hidden_policy_acts_prev.size() * sizeof(float));
    reader.read(&value_dendrite_acts[0], value_dendrite_acts_prev.size() * sizeof(float));
    reader.read(&policy_dendrite_acts_prev[0], policy_dendrite_acts_prev.size() * sizeof(float));
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
