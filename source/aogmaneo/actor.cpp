// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
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
    unsigned long* state,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

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

                int wi_value = offset.y + diam * (offset.x + diam * (in_ci + vld.size.z * hidden_column_index));
                int wi_start_partial = hidden_size.z * wi_value;

                for (int hc = 0; hc < hidden_size.z; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

                    int wi_start = num_dendrites_per_cell * (hc + wi_start_partial);

                    for (int di = 0; di < num_dendrites_per_cell; di++) {
                        int dendrite_index = di + dendrites_start;

                        int wi = di + wi_start;

                        dendrite_acts[dendrite_index] += vl.action_weights[wi];
                    }
                }

                value += vl.value_weights[wi_value];
            }
    }

    value /= count;

    hidden_values[hidden_column_index] = value;

    float max_activation = limit_min;

    const int half_num_dendrites_per_cell = num_dendrites_per_cell / 2;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

        float activation = 0.0f;

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + dendrites_start;

            float act = dendrite_acts[dendrite_index] / count;

            dendrite_acts[dendrite_index] = max(act * params.leak, act);

            activation += dendrite_acts[dendrite_index] * ((di >= half_num_dendrites_per_cell) * 2.0f - 1.0f);
        }

        activation /= num_dendrites_per_cell;

        hidden_acts[hidden_cell_index] = activation;

        max_activation = max(max_activation, activation);
    }

    // softmax
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
}

void Actor::learn(
    const Int2 &column_pos,
    int t,
    float r,
    float d,
    float mimic,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    int target_ci = history_samples[t - 1].hidden_target_cis_prev[hidden_column_index];

    float new_value = r + d * hidden_values[hidden_column_index];

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + dendrites_start;

            dendrite_acts[dendrite_index] = 0.0f;
            dendrite_acts_delayed[dendrite_index] = 0.0f;
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

        Int_Buffer_View vl_input_cis = history_samples[t].input_cis[vli];

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci = vl_input_cis[visible_column_index];

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi_value = offset.y + diam * (offset.x + diam * (in_ci + vld.size.z * hidden_column_index));
                int wi_start_partial = hidden_size.z * wi_value;

                for (int hc = 0; hc < hidden_size.z; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

                    int wi_start = num_dendrites_per_cell * (hc + wi_start_partial);

                    for (int di = 0; di < num_dendrites_per_cell; di++) {
                        int dendrite_index = di + dendrites_start;

                        int wi = di + wi_start;

                        dendrite_acts[dendrite_index] += vl.action_weights[wi];
                        dendrite_acts_delayed[dendrite_index] += vl.action_weights_delayed[wi];
                    }
                }

                value += vl.value_weights[wi_value];
            }
    }

    value /= count;

    float max_activation = limit_min;
    float max_activation_delayed = limit_min;

    const int half_num_dendrites_per_cell = num_dendrites_per_cell / 2;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

        float activation = 0.0f;
        float activation_delayed = 0.0f;

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + dendrites_start;

            float act = dendrite_acts[dendrite_index] / count;
            float act_delayed = dendrite_acts_delayed[dendrite_index] / count;

            dendrite_acts[dendrite_index] = max(act * params.leak, act);
            dendrite_acts_delayed[dendrite_index] = max(act_delayed * params.leak, act_delayed);

            activation += dendrite_acts[dendrite_index] * ((di >= half_num_dendrites_per_cell) * 2.0f - 1.0f);
            activation_delayed += dendrite_acts_delayed[dendrite_index] * ((di >= half_num_dendrites_per_cell) * 2.0f - 1.0f);
        }

        activation /= num_dendrites_per_cell;
        activation_delayed /= num_dendrites_per_cell;

        hidden_acts[hidden_cell_index] = activation;
        hidden_acts_delayed[hidden_cell_index] = activation_delayed;

        max_activation = max(max_activation, activation);
        max_activation_delayed = max(max_activation, activation_delayed);
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

    float td_error_value = new_value - value;
    
    float value_delta = params.vlr * td_error_value;

    float advantage = tanhf(td_error_value);

    float ratio = hidden_acts[target_ci + hidden_cells_start] / max(limit_small, hidden_acts_delayed[target_ci + hidden_cells_start]);

    float action_error_partial = params.alr * (mimic + (1.0f - mimic) * advantage * (ratio < 1.0f + params.clip_coef && ratio > 1.0f - params.clip_coef));

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

        Int_Buffer_View vl_input_cis = history_samples[t].input_cis[vli];

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci = vl_input_cis[visible_column_index];

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi_value = offset.y + diam * (offset.x + diam * (in_ci + vld.size.z * hidden_column_index));
                int wi_start_partial = hidden_size.z * wi_value;

                for (int hc = 0; hc < hidden_size.z; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

                    float error = action_error_partial * ((hc == target_ci) - hidden_acts[hidden_cell_index]);

                    int wi_start = num_dendrites_per_cell * (hc + wi_start_partial);

                    for (int di = 0; di < num_dendrites_per_cell; di++) {
                        int dendrite_index = di + dendrites_start;

                        int wi = di + wi_start;

                        float delta = error * ((di >= half_num_dendrites_per_cell) * 2.0f - 1.0f) * ((dendrite_acts[dendrite_index] > 0.0f) * (1.0f - params.leak) + params.leak);

                        vl.action_weights[wi] += delta;
                    }
                }

                vl.value_weights[wi_value] += value_delta;
            }
    }
}

void Actor::init_random(
    const Int3 &hidden_size,
    int num_dendrites_per_cell,
    int history_capacity,
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

        vl.action_weights.resize(num_dendrites * area * vld.size.z);

        for (int i = 0; i < vl.action_weights.size(); i++)
            vl.action_weights[i] = randf(-init_weight_noisef, init_weight_noisef);

        vl.action_weights_delayed = vl.action_weights;

        vl.value_weights.resize(num_hidden_columns * area * vld.size.z);

        for (int i = 0; i < vl.value_weights.size(); i++)
            vl.value_weights[i] = randf(-init_weight_noisef, init_weight_noisef);
    }

    hidden_cis = Int_Buffer(num_hidden_columns, 0);

    hidden_values = Float_Buffer(num_hidden_columns, 0.0f);

    dendrite_acts.resize(num_dendrites);
    dendrite_acts_delayed.resize(num_dendrites);

    hidden_acts.resize(num_hidden_cells);
    hidden_acts_delayed.resize(num_hidden_cells);

    // create (pre-allocated) history samples
    history_size = 0;
    history_samples.resize(history_capacity);

    for (int i = 0; i < history_samples.size(); i++) {
        history_samples[i].input_cis.resize(visible_layers.size());

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

            int num_visible_columns = vld.size.x * vld.size.y;

            history_samples[i].input_cis[vli].resize(num_visible_columns);
        }

        history_samples[i].hidden_target_cis_prev.resize(num_hidden_columns);
    }
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

        forward(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, &state, params);
    }

    history_samples.push_front();

    // if not at cap, increment
    if (history_size < history_samples.size())
        history_size++;
    
    // add new sample
    {
        History_Sample &s = history_samples[0];

        for (int vli = 0; vli < visible_layers.size(); vli++)
            s.input_cis[vli] = input_cis[vli];

        // copy
        s.hidden_target_cis_prev = hidden_target_cis_prev;

        s.reward = reward;
    }

    // learn (if have sufficient samples)
    if (learn_enabled && history_size > params.min_steps) {
        for (int it = 0; it < params.history_iters; it++) {
            int t = rand() % (history_size - params.min_steps) + params.min_steps;

            // compute (partial) values, rest is completed in the kernel
            float r = 0.0f;
            float d = 1.0f;

            for (int t2 = t - 1; t2 >= 0; t2--) {
                r += history_samples[t2].reward * d;

                d *= params.discount;
            }

            PARALLEL_FOR
            for (int i = 0; i < num_hidden_columns; i++)
                learn(Int2(i / hidden_size.y, i % hidden_size.y), t, r, d, mimic, params);
        }

        // update delayed weights
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];

            PARALLEL_FOR
            for (int i = 0; i < vl.action_weights.size(); i++)
                vl.action_weights_delayed[i] += params.rate * (vl.action_weights[i] - vl.action_weights_delayed[i]);
        }
    }
}

void Actor::clear_state() {
    hidden_cis.fill(0);
    hidden_values.fill(0.0f);

    history_size = 0;
}

long Actor::size() const {
    long size = sizeof(Int3) + sizeof(int) + hidden_cis.size() * sizeof(int) + hidden_values.size() * sizeof(float) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        size += sizeof(Visible_Layer_Desc) + 2 * vl.action_weights.size() * sizeof(float) + vl.value_weights.size() * sizeof(float);
    }

    size += 3 * sizeof(int);

    int sample_size = 0;

    const History_Sample &s = history_samples[0];

    for (int vli = 0; vli < visible_layers.size(); vli++)
        sample_size += s.input_cis[vli].size() * sizeof(int);

    sample_size += s.hidden_target_cis_prev.size() * sizeof(int) + sizeof(float);

    size += history_samples.size() * sample_size;

    return size;
}

long Actor::state_size() const {
    long size = hidden_cis.size() * sizeof(int) + hidden_values.size() * sizeof(float) + 2 * sizeof(int);

    int sample_size = 0;

    const History_Sample &s = history_samples[0];

    for (int vli = 0; vli < visible_layers.size(); vli++)
        sample_size += s.input_cis[vli].size() * sizeof(int);

    sample_size += s.hidden_target_cis_prev.size() * sizeof(int) + sizeof(float);

    size += history_samples.size() * sample_size;

    return size;
}

long Actor::weights_size() const {
    long size = 0;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += 2 * vl.action_weights.size() * sizeof(float) + vl.value_weights.size() * sizeof(float);
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

    int num_visible_layers = visible_layers.size();

    writer.write(reinterpret_cast<const void*>(&num_visible_layers), sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(Visible_Layer_Desc));

        writer.write(reinterpret_cast<const void*>(&vl.action_weights[0]), vl.action_weights.size() * sizeof(float));
        writer.write(reinterpret_cast<const void*>(&vl.action_weights_delayed[0]), vl.action_weights_delayed.size() * sizeof(float));
        writer.write(reinterpret_cast<const void*>(&vl.value_weights[0]), vl.value_weights.size() * sizeof(float));
    }

    writer.write(reinterpret_cast<const void*>(&history_size), sizeof(int));

    int num_history_samples = history_samples.size();

    writer.write(reinterpret_cast<const void*>(&num_history_samples), sizeof(int));

    int history_start = history_samples.start;

    writer.write(reinterpret_cast<const void*>(&history_start), sizeof(int));

    for (int t = 0; t < history_samples.size(); t++) {
        const History_Sample &s = history_samples[t];

        for (int vli = 0; vli < visible_layers.size(); vli++)
            writer.write(reinterpret_cast<const void*>(&s.input_cis[vli][0]), s.input_cis[vli].size() * sizeof(int));

        writer.write(reinterpret_cast<const void*>(&s.hidden_target_cis_prev[0]), s.hidden_target_cis_prev.size() * sizeof(int));

        writer.write(reinterpret_cast<const void*>(&s.reward), sizeof(float));
    }
}

void Actor::read(
    Stream_Reader &reader
) {
    reader.read(reinterpret_cast<void*>(&hidden_size), sizeof(Int3));
    reader.read(reinterpret_cast<void*>(&num_dendrites_per_cell), sizeof(int));

    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;
    int num_dendrites = num_hidden_cells * num_dendrites_per_cell;
    
    hidden_cis.resize(num_hidden_columns);
    hidden_values.resize(num_hidden_columns);

    reader.read(reinterpret_cast<void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hidden_values[0]), hidden_values.size() * sizeof(float));

    dendrite_acts.resize(num_dendrites);
    dendrite_acts_delayed.resize(num_dendrites);

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

        vl.action_weights.resize(num_dendrites * area * vld.size.z);
        vl.action_weights_delayed.resize(vl.action_weights.size());

        reader.read(reinterpret_cast<void*>(&vl.action_weights[0]), vl.action_weights.size() * sizeof(float));
        reader.read(reinterpret_cast<void*>(&vl.action_weights_delayed[0]), vl.action_weights_delayed.size() * sizeof(float));

        vl.value_weights.resize(num_hidden_columns * area * vld.size.z);

        reader.read(reinterpret_cast<void*>(&vl.value_weights[0]), vl.value_weights.size() * sizeof(float));
    }

    reader.read(reinterpret_cast<void*>(&history_size), sizeof(int));

    int num_history_samples;

    reader.read(reinterpret_cast<void*>(&num_history_samples), sizeof(int));

    int history_start;

    reader.read(reinterpret_cast<void*>(&history_start), sizeof(int));

    history_samples.resize(num_history_samples);
    history_samples.start = history_start;

    for (int t = 0; t < history_samples.size(); t++) {
        History_Sample &s = history_samples[t];

        s.input_cis.resize(num_visible_layers);

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];

            int num_visible_columns = vld.size.x * vld.size.y;

            s.input_cis[vli].resize(num_visible_columns);

            reader.read(reinterpret_cast<void*>(&s.input_cis[vli][0]), s.input_cis[vli].size() * sizeof(int));
        }

        s.hidden_target_cis_prev.resize(num_hidden_columns);

        reader.read(reinterpret_cast<void*>(&s.hidden_target_cis_prev[0]), s.hidden_target_cis_prev.size() * sizeof(int));

        reader.read(reinterpret_cast<void*>(&s.reward), sizeof(float));
    }
}

void Actor::write_state(
    Stream_Writer &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hidden_values[0]), hidden_values.size() * sizeof(float));

    writer.write(reinterpret_cast<const void*>(&history_size), sizeof(int));

    int history_start = history_samples.start;

    writer.write(reinterpret_cast<const void*>(&history_start), sizeof(int));

    for (int t = 0; t < history_samples.size(); t++) {
        const History_Sample &s = history_samples[t];

        for (int vli = 0; vli < visible_layers.size(); vli++)
            writer.write(reinterpret_cast<const void*>(&s.input_cis[vli][0]), s.input_cis[vli].size() * sizeof(int));

        writer.write(reinterpret_cast<const void*>(&s.hidden_target_cis_prev[0]), s.hidden_target_cis_prev.size() * sizeof(int));

        writer.write(reinterpret_cast<const void*>(&s.reward), sizeof(float));
    }
}

void Actor::read_state(
    Stream_Reader &reader
) {
    reader.read(reinterpret_cast<void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hidden_values[0]), hidden_values.size() * sizeof(float));

    reader.read(reinterpret_cast<void*>(&history_size), sizeof(int));

    int history_start;

    reader.read(reinterpret_cast<void*>(&history_start), sizeof(int));

    history_samples.start = history_start;

    for (int t = 0; t < history_samples.size(); t++) {
        History_Sample &s = history_samples[t];

        for (int vli = 0; vli < visible_layers.size(); vli++)
            reader.read(reinterpret_cast<void*>(&s.input_cis[vli][0]), s.input_cis[vli].size() * sizeof(int));

        reader.read(reinterpret_cast<void*>(&s.hidden_target_cis_prev[0]), s.hidden_target_cis_prev.size() * sizeof(int));

        reader.read(reinterpret_cast<void*>(&s.reward), sizeof(float));
    }
}

void Actor::write_weights(
    Stream_Writer &writer
) const {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        writer.write(reinterpret_cast<const void*>(&vl.value_weights[0]), vl.value_weights.size() * sizeof(float));
        writer.write(reinterpret_cast<const void*>(&vl.action_weights[0]), vl.action_weights.size() * sizeof(float));
        writer.write(reinterpret_cast<const void*>(&vl.action_weights_delayed[0]), vl.action_weights_delayed.size() * sizeof(float));
    }
}

void Actor::read_weights(
    Stream_Reader &reader
) {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(reinterpret_cast<void*>(&vl.value_weights[0]), vl.value_weights.size() * sizeof(float));
        reader.read(reinterpret_cast<void*>(&vl.action_weights[0]), vl.action_weights.size() * sizeof(float));
        reader.read(reinterpret_cast<void*>(&vl.action_weights_delayed[0]), vl.action_weights_delayed.size() * sizeof(float));
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
        
            for (int i = 0; i < vl.action_weights.size(); i++) {
                int d = rand() % actors.size();                

                vl.action_weights[i] = actors[d]->visible_layers[vli].action_weights[i];
                vl.action_weights_delayed[i] = actors[d]->visible_layers[vli].action_weights_delayed[i];
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
        
            for (int i = 0; i < vl.action_weights.size(); i++) {
                float total = 0.0f;
                float total_delayed = 0.0f;

                for (int d = 0; d < actors.size(); d++) {
                    total += actors[d]->visible_layers[vli].action_weights[i];
                    total_delayed += actors[d]->visible_layers[vli].action_weights_delayed[i];
                }

                vl.action_weights[i] = total / actors.size();
                vl.action_weights_delayed[i] = total_delayed / actors.size();
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
