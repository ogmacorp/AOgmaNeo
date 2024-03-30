// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
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
    float reward,
    float mimic,
    bool learn_enabled,
    unsigned long* state,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    float value_prev = hidden_values[hidden_column_index];

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int policy_dendrites_start = num_dendrites_per_cell * hidden_cell_index;

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + policy_dendrites_start;

            policy_dendrite_acts[dendrite_index] = 0.0f;
            policy_dendrite_acts_delayed[dendrite_index] = 0.0f;
        }
    }

    int value_dendrites_start = hidden_column_index * num_dendrites_per_cell;

    for (int di = 0; di < num_dendrites_per_cell; di++) {
        int dendrite_index = di + value_dendrites_start;

        value_dendrite_acts[dendrite_index] = 0.0f;
        value_dendrite_acts_delayed[dendrite_index] = 0.0f;
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

                int wi_value_partial = offset.y + diam * (offset.x + diam * (in_ci + vld.size.z * hidden_column_index));
                int wi_start_partial = hidden_size.z * wi_value_partial;

                for (int hc = 0; hc < hidden_size.z; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int policy_dendrites_start = num_dendrites_per_cell * hidden_cell_index;

                    int wi_start = num_dendrites_per_cell * (hc + wi_start_partial);

                    for (int di = 0; di < num_dendrites_per_cell; di++) {
                        int dendrite_index = di + policy_dendrites_start;

                        int wi = di + wi_start;

                        policy_dendrite_acts[dendrite_index] += vl.policy_weights[wi];
                        policy_dendrite_acts_delayed[dendrite_index] += vl.policy_weights_delayed[wi];
                    }
                }

                int wi_value_start = num_dendrites_per_cell * wi_value_partial;

                for (int di = 0; di < num_dendrites_per_cell; di++) {
                    int dendrite_index = di + value_dendrites_start;

                    int wi = di + wi_value_start;

                    value_dendrite_acts[dendrite_index] += vl.value_weights[wi];
                    value_dendrite_acts_delayed[dendrite_index] += vl.value_weights_delayed[wi];
                }
            }
    }

    const float dendrite_scale = sqrtf(1.0f / count);

    float value;

    // value
    {
        float max_act = limit_min;
        float max_act_delayed = limit_min;

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + value_dendrites_start;

            float act = value_dendrite_acts[dendrite_index] * dendrite_scale;
            float act_delayed = value_dendrite_acts_delayed[dendrite_index] * dendrite_scale;

            value_dendrite_acts[dendrite_index] = act;
            value_dendrite_acts_delayed[dendrite_index] = act_delayed;

            max_act = max(max_act, act);
            max_act_delayed = max(max_act_delayed, act);
        }

        float total = 0.0f;
        float total_delayed = 0.0f;

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + value_dendrites_start;

            value_dendrite_acts[dendrite_index] = expf(value_dendrite_acts[dendrite_index] - max_act);
            value_dendrite_acts_delayed[dendrite_index] = expf(value_dendrite_acts_delayed[dendrite_index] - max_act_delayed);

            total += value_dendrite_acts[dendrite_index];
            total_delayed += value_dendrite_acts_delayed[dendrite_index];
        }

        float total_inv = 1.0f / max(limit_small, total);
        //float total_inv_delayed = 1.0f / max(limit_small, total_delayed);

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + value_dendrites_start;

            value_dendrite_acts[dendrite_index] *= total_inv;
            //value_dendrite_acts_delayed[dendrite_index] *= total_inv_delayed;
        }

        value = max_act_delayed + logf(total_delayed); // log sum exp

        hidden_values[hidden_column_index] = max_act + logf(total);
    }

    float max_activation = limit_min;
    float max_activation_delayed = limit_min;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

        float max_act = limit_min;
        float max_act_delayed = limit_min;

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + dendrites_start;

            float act = policy_dendrite_acts[dendrite_index] * dendrite_scale;
            float act_delayed = policy_dendrite_acts_delayed[dendrite_index] * dendrite_scale;

            policy_dendrite_acts[dendrite_index] = act;
            policy_dendrite_acts_delayed[dendrite_index] = act_delayed;

            max_act = max(max_act, act);
            max_act_delayed = max(max_act_delayed, act_delayed);
        }

        float total = 0.0f;
        float total_delayed = 0.0f;

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + dendrites_start;

            policy_dendrite_acts[dendrite_index] = expf(policy_dendrite_acts[dendrite_index] - max_act);
            policy_dendrite_acts_delayed[dendrite_index] = expf(policy_dendrite_acts_delayed[dendrite_index] - max_act_delayed);

            total += policy_dendrite_acts[dendrite_index];
            total_delayed += policy_dendrite_acts_delayed[dendrite_index];
        }

        float total_inv = 1.0f / max(limit_small, total);

        // delayed not needed, since it's inference only here
        //float total_inv_delayed = 1.0f / max(limit_small, total_delayed);

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + dendrites_start;

            policy_dendrite_acts[dendrite_index] *= total_inv;
            //policy_dendrite_acts_delayed[dendrite_index] *= total_inv_delayed;
        }

        float activation = max_act + logf(total); // log sum exp
        float activation_delayed = max_act_delayed + logf(total_delayed); // log sum exp

        hidden_acts[hidden_cell_index] = activation;
        hidden_acts_delayed[hidden_cell_index] = activation_delayed;

        max_activation = max(max_activation, activation);
        max_activation_delayed = max(max_activation_delayed, activation_delayed);
    }

    // softmax
    float total = 0.0f;
    float total_delayed = 0.0f;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;
    
        hidden_acts[hidden_cell_index] = expf(hidden_acts[hidden_cell_index] - max_activation);
        hidden_acts_delayed[hidden_cell_index] = expf(hidden_acts_delayed[hidden_cell_index] - max_activation_delayed);

        total += hidden_acts[hidden_cell_index];
        total_delayed += hidden_acts_delayed[hidden_cell_index];
    }

    float total_inv = 1.0f / max(limit_small, total);
    float total_inv_delayed = 1.0f / max(limit_small, total_delayed);

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        hidden_acts[hidden_cell_index] *= total_inv;
        hidden_acts_delayed[hidden_cell_index] *= total_inv_delayed;
    }

    float cusp = randf(state);

    int select_index = 0;
    float sum_so_far = 0.0f;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        sum_so_far += hidden_acts[hidden_cell_index];

        if (sum_so_far >= cusp) {
            select_index = hc;

            break;
        }
    }
    
    hidden_cis[hidden_column_index] = select_index;

    if (learn_enabled) {
        float td_error_value = tanhf(reward + params.discount * value - value_prev);
        
        float value_delta = params.vlr * td_error_value;

        int target_ci = hidden_target_cis_prev[hidden_column_index];

        // probability ratio
        float ratio = hidden_acts_prev[target_ci + hidden_cells_start] / max(limit_small, hidden_acts_delayed[target_ci + hidden_cells_start]);

        // https://huggingface.co/blog/deep-rl-ppo
        bool clip = (ratio < (1.0f - params.clip_coef) && td_error_value < 0.0f) || (ratio > (1.0f + params.clip_coef) && td_error_value > 0.0f);
        
        float policy_error_partial = params.plr * (mimic + (1.0f - mimic) * td_error_value * (!clip));

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

                    for (int vc = 0; vc < vld.size.z; vc++) {
                        int wi_value_partial = offset.y + diam * (offset.x + diam * (vc + vld.size.z * hidden_column_index));
                        int wi_start_partial = hidden_size.z * wi_value_partial;

                        for (int hc = 0; hc < hidden_size.z; hc++) {
                            int hidden_cell_index = hc + hidden_cells_start;

                            int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

                            float error = (hc == target_ci) - hidden_acts_prev[hidden_cell_index];

                            int wi_start = num_dendrites_per_cell * (hc + wi_start_partial);

                            for (int di = 0; di < num_dendrites_per_cell; di++) {
                                int dendrite_index = di + dendrites_start;

                                int wi = di + wi_start;

                                if (vc == in_ci_prev) {
                                    float trace = vl.policy_traces[wi];

                                    vl.policy_traces[wi] += error * policy_dendrite_acts_prev[dendrite_index] * expf(-trace * trace * params.trace_curve);
                                }

                                vl.policy_weights[wi] += policy_error_partial * vl.policy_traces[wi];
                            }
                        }

                        int wi_value_start = num_dendrites_per_cell * wi_value_partial;

                        for (int di = 0; di < num_dendrites_per_cell; di++) {
                            int dendrite_index = di + value_dendrites_start;

                            int wi = di + wi_value_start;

                            if (vc == in_ci_prev) {
                                float trace = vl.value_traces[wi];

                                vl.value_traces[wi] += value_dendrite_acts_prev[dendrite_index] * expf(-trace * trace * params.trace_curve);
                            }

                            vl.value_weights[wi] += value_delta * vl.value_traces[wi];
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
    int policy_num_dendrites = num_hidden_cells * num_dendrites_per_cell;
    int value_num_dendrites = num_hidden_columns * num_dendrites_per_cell;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;
        int num_visible_cells = num_visible_columns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.policy_weights.resize(policy_num_dendrites * area * vld.size.z);

        for (int i = 0; i < vl.policy_weights.size(); i++)
            vl.policy_weights[i] = randf(-init_weight_noisef, init_weight_noisef);

        vl.policy_weights_delayed = vl.policy_weights;

        vl.policy_traces = Float_Buffer(vl.policy_weights.size(), 0.0f);

        vl.value_weights.resize(value_num_dendrites * area * vld.size.z);

        for (int i = 0; i < vl.value_weights.size(); i++)
            vl.value_weights[i] = randf(-init_weight_noisef, init_weight_noisef);
        
        vl.value_weights_delayed = vl.value_weights;

        vl.value_traces = Float_Buffer(vl.value_weights.size(), 0.0f);

        vl.input_cis_prev = Int_Buffer(num_visible_columns, 0);
    }

    hidden_cis = Int_Buffer(num_hidden_columns, 0);

    hidden_values = Float_Buffer(num_hidden_columns, 0.0f);

    policy_dendrite_acts.resize(policy_num_dendrites);
    policy_dendrite_acts_delayed.resize(policy_num_dendrites);

    policy_dendrite_acts_prev = Float_Buffer(policy_num_dendrites, 0.0f);

    value_dendrite_acts.resize(value_num_dendrites);
    value_dendrite_acts_delayed.resize(value_num_dendrites);

    value_dendrite_acts_prev = Float_Buffer(value_num_dendrites, 0.0f);

    hidden_acts.resize(num_hidden_cells);
    hidden_acts_delayed.resize(num_hidden_cells);
    hidden_acts_prev = Float_Buffer(num_hidden_cells, 0.0f);
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

        forward(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, hidden_target_cis_prev, reward, mimic, learn_enabled, &state, params);
    }

    // update delayed weights and traces
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        PARALLEL_FOR
        for (int i = 0; i < vl.policy_weights.size(); i++) {
            vl.policy_traces[i] *= params.trace_decay;
            vl.policy_weights_delayed[i] += params.policy_rate * (vl.policy_weights[i] - vl.policy_weights_delayed[i]);
        }

        PARALLEL_FOR
        for (int i = 0; i < vl.value_weights.size(); i++) {
            vl.value_traces[i] *= params.trace_decay;
            vl.value_weights_delayed[i] += params.value_rate * (vl.value_weights[i] - vl.value_weights_delayed[i]);
        }

        vl.input_cis_prev = input_cis[vli];
    }

    hidden_acts_prev = hidden_acts;
    policy_dendrite_acts_prev = policy_dendrite_acts;
    value_dendrite_acts_prev = value_dendrite_acts;
}

void Actor::clear_state() {
    hidden_cis.fill(0);
    hidden_values.fill(0.0f);
    hidden_acts_prev.fill(0.0f);

    policy_dendrite_acts_prev.fill(0.0f);
    value_dendrite_acts_prev.fill(0.0f);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        vl.policy_traces.fill(0.0f);
        vl.value_traces.fill(0.0f);

        vl.input_cis_prev.fill(0);
    }
}

long Actor::size() const {
    long size = sizeof(Int3) + sizeof(int) + hidden_cis.size() * sizeof(int) + hidden_values.size() * sizeof(float) + hidden_acts_prev.size() * sizeof(float) + policy_dendrite_acts_prev.size() * sizeof(float) + value_dendrite_acts_prev.size() * sizeof(float) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        size += sizeof(Visible_Layer_Desc) + 2 * vl.policy_weights.size() * sizeof(float) + 2 * vl.value_weights.size() * sizeof(float) + vl.input_cis_prev.size() * sizeof(int);
    }

    return size;
}

long Actor::state_size() const {
    long size = hidden_cis.size() * sizeof(int) + hidden_values.size() * sizeof(float) + hidden_acts_prev.size() * sizeof(float) + policy_dendrite_acts_prev.size() * sizeof(float) + value_dendrite_acts_prev.size() * sizeof(float);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += vl.policy_traces.size() * sizeof(float) + vl.value_traces.size() * sizeof(float) + vl.input_cis_prev.size() * sizeof(int);
    }

    return size;
}

long Actor::weights_size() const {
    long size = 0;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += vl.policy_weights.size() * sizeof(float) + vl.value_weights.size() * sizeof(float);
    }

    return size;
}

void Actor::write(
    Stream_Writer &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hidden_size), sizeof(Int3));
    writer.write(reinterpret_cast<const void*>(&num_dendrites_per_cell), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hidden_values[0]), hidden_values.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&hidden_acts_prev[0]), hidden_acts_prev.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&policy_dendrite_acts_prev[0]), policy_dendrite_acts_prev.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&value_dendrite_acts_prev[0]), value_dendrite_acts_prev.size() * sizeof(float));

    int num_visible_layers = visible_layers.size();

    writer.write(reinterpret_cast<const void*>(&num_visible_layers), sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(Visible_Layer_Desc));

        writer.write(reinterpret_cast<const void*>(&vl.policy_weights[0]), vl.policy_weights.size() * sizeof(float));
        writer.write(reinterpret_cast<const void*>(&vl.value_weights[0]), vl.value_weights.size() * sizeof(float));

        writer.write(reinterpret_cast<const void*>(&vl.input_cis_prev[0]), vl.input_cis_prev.size() * sizeof(int));
    }
}

void Actor::read(
    Stream_Reader &reader
) {
    reader.read(reinterpret_cast<void*>(&hidden_size), sizeof(Int3));
    reader.read(reinterpret_cast<void*>(&num_dendrites_per_cell), sizeof(int));

    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;
    int policy_num_dendrites = num_hidden_cells * num_dendrites_per_cell;
    int value_num_dendrites = num_hidden_columns * num_dendrites_per_cell;
    
    hidden_cis.resize(num_hidden_columns);
    hidden_values.resize(num_hidden_columns);
    hidden_acts_prev.resize(num_hidden_cells);
    policy_dendrite_acts_prev.resize(policy_num_dendrites);
    value_dendrite_acts_prev.resize(value_num_dendrites);

    reader.read(reinterpret_cast<void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hidden_values[0]), hidden_values.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&hidden_acts_prev[0]), hidden_acts_prev.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&policy_dendrite_acts_prev[0]), policy_dendrite_acts_prev.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&value_dendrite_acts_prev[0]), value_dendrite_acts_prev.size() * sizeof(float));

    policy_dendrite_acts.resize(policy_num_dendrites);
    policy_dendrite_acts_delayed.resize(policy_num_dendrites);
    value_dendrite_acts.resize(value_num_dendrites);
    value_dendrite_acts_delayed.resize(value_num_dendrites);

    hidden_acts.resize(num_hidden_cells);
    hidden_acts_delayed.resize(num_hidden_cells);

    int num_visible_layers;

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

        vl.policy_weights.resize(policy_num_dendrites * area * vld.size.z);

        reader.read(reinterpret_cast<void*>(&vl.policy_weights[0]), vl.policy_weights.size() * sizeof(float));

        vl.policy_weights_delayed = vl.policy_weights;

        vl.value_weights.resize(value_num_dendrites * area * vld.size.z);

        reader.read(reinterpret_cast<void*>(&vl.value_weights[0]), vl.value_weights.size() * sizeof(float));

        vl.value_weights_delayed = vl.value_weights;

        vl.input_cis_prev.resize(num_visible_columns);

        reader.read(reinterpret_cast<void*>(&vl.input_cis_prev[0]), vl.input_cis_prev.size() * sizeof(int));
    }
}

void Actor::write_state(
    Stream_Writer &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hidden_values[0]), hidden_values.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&hidden_acts_prev[0]), hidden_acts_prev.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&policy_dendrite_acts_prev[0]), policy_dendrite_acts_prev.size() * sizeof(float));
    writer.write(reinterpret_cast<const void*>(&value_dendrite_acts_prev[0]), value_dendrite_acts_prev.size() * sizeof(float));

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        writer.write(reinterpret_cast<const void*>(&vl.policy_traces[0]), vl.policy_traces.size() * sizeof(float));
        writer.write(reinterpret_cast<const void*>(&vl.value_traces[0]), vl.value_traces.size() * sizeof(float));

        writer.write(reinterpret_cast<const void*>(&vl.input_cis_prev[0]), vl.input_cis_prev.size() * sizeof(int));
    }
}

void Actor::read_state(
    Stream_Reader &reader
) {
    reader.read(reinterpret_cast<void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hidden_values[0]), hidden_values.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&hidden_acts_prev[0]), hidden_acts_prev.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&policy_dendrite_acts_prev[0]), policy_dendrite_acts_prev.size() * sizeof(float));
    reader.read(reinterpret_cast<void*>(&value_dendrite_acts_prev[0]), value_dendrite_acts_prev.size() * sizeof(float));

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(reinterpret_cast<void*>(&vl.policy_traces[0]), vl.policy_traces.size() * sizeof(float));
        reader.read(reinterpret_cast<void*>(&vl.value_traces[0]), vl.value_traces.size() * sizeof(float));

        reader.read(reinterpret_cast<void*>(&vl.input_cis_prev[0]), vl.input_cis_prev.size() * sizeof(int));
    }
}

void Actor::write_weights(
    Stream_Writer &writer
) const {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        writer.write(reinterpret_cast<const void*>(&vl.value_weights[0]), vl.value_weights.size() * sizeof(float));
        writer.write(reinterpret_cast<const void*>(&vl.policy_weights[0]), vl.policy_weights.size() * sizeof(float));
    }
}

void Actor::read_weights(
    Stream_Reader &reader
) {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(reinterpret_cast<void*>(&vl.value_weights[0]), vl.value_weights.size() * sizeof(float));
        reader.read(reinterpret_cast<void*>(&vl.policy_weights[0]), vl.policy_weights.size() * sizeof(float));
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
        
            for (int i = 0; i < vl.policy_weights.size(); i++) {
                int d = rand() % actors.size();                

                vl.policy_weights[i] = actors[d]->visible_layers[vli].policy_weights[i];
            }

            for (int i = 0; i < vl.value_weights.size(); i++) {
                int d = rand() % actors.size();                

                vl.value_weights[i] = actors[d]->visible_layers[vli].value_weights[i];
            }
        }

        break;
    case merge_average:
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];
        
            for (int i = 0; i < vl.policy_weights.size(); i++) {
                float total = 0.0f;

                for (int d = 0; d < actors.size(); d++)
                    total += actors[d]->visible_layers[vli].policy_weights[i];

                vl.policy_weights[i] = total / actors.size();
            }

            for (int i = 0; i < vl.value_weights.size(); i++) {
                float total = 0.0f;

                for (int d = 0; d < actors.size(); d++)
                    total += actors[d]->visible_layers[vli].value_weights[i];

                vl.value_weights[i] = total / actors.size();
            }
        }

        break;
    }
}
