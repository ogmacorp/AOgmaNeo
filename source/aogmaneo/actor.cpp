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
    unsigned long* state,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;
    int dendrites_start = hidden_column_index * num_dendrites_per_column;

    for (int di = 0; di < num_dendrites_per_column; di++) {
        int dendrite_index = di + dendrites_start;

        dendrite_acts[dendrite_index] = 0.0f;
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

                int wi_start = num_dendrites_per_column * (offset.y + diam * (offset.x + diam * (in_ci + vld.size.z * hidden_column_index)));

                for (int di = 0; di < num_dendrites_per_column; di++) {
                    int dendrite_index = di + dendrites_start;

                    int wi = di + wi_start;

                    dendrite_acts[dendrite_index] += vl.weights[wi];
                }
            }
    }

    for (int di = 0; di < num_dendrites_per_column; di++) {
        int dendrite_index = di + dendrites_start;

        dendrite_acts[dendrite_index] = max(0.0f, (dendrite_acts[dendrite_index] / (count * 255) - 0.5f) * 2.0f * params.scale);
    }

    int max_index = 0;
    float max_activation = limit_min;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int wi_start = num_dendrites_per_column * hidden_cell_index;

        float activation = 0.0f;

        for (int di = 0; di < num_dendrites_per_column; di++) {
            int dendrite_index = di + dendrites_start;

            int wi = di + wi_start;

            activation += dendrite_weights[wi] * dendrite_acts[dendrite_index];
        }

        if (activation > max_activation) {
            max_activation = activation;
            max_index = hc;
        }
    }

    hidden_cis[hidden_column_index] = max_index;
}

void Actor::learn(
    const Int2 &column_pos,
    int t,
    unsigned long* state,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;
    int dendrites_start = hidden_column_index * num_dendrites_per_column;

    int target_ci = history_samples[t - 1].hidden_target_cis_prev[hidden_column_index];

    for (int di = 0; di < num_dendrites_per_column; di++) {
        int dendrite_index = di + dendrites_start;

        dendrite_acts[dendrite_index] = 0.0f;
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

        Int_Buffer_View vl_input_cis = history_samples[t - params.n_steps].input_cis[vli];

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci = vl_input_cis[visible_column_index];

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi_start = num_dendrites_per_column * (offset.y + diam * (offset.x + diam * (in_ci + vld.size.z * hidden_column_index)));

                for (int di = 0; di < num_dendrites_per_column; di++) {
                    int dendrite_index = di + dendrites_start;

                    int wi = di + wi_start;

                    dendrite_acts[dendrite_index] += vl.weights[wi];
                }
            }
    }

    for (int di = 0; di < num_dendrites_per_column; di++) {
        int dendrite_index = di + dendrites_start;

        dendrite_acts[dendrite_index] = max(0.0f, (dendrite_acts[dendrite_index] / (count * 255) - 0.5f) * 2.0f * params.scale);
    }

    float max_activation_next = limit_min;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int wi_start = num_dendrites_per_column * hidden_cell_index;

        float activation = 0.0f;

        for (int di = 0; di < num_dendrites_per_column; di++) {
            int dendrite_index = di + dendrites_start;

            int wi = di + wi_start;

            activation += dendrite_weights[wi] * dendrite_acts[dendrite_index];
        }

        max_activation_next = max(max_activation_next, activation);
    }

    for (int di = 0; di < num_dendrites_per_column; di++) {
        int dendrite_index = di + dendrites_start;

        dendrite_acts[dendrite_index] = 0.0f;
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

        Int_Buffer_View vl_input_cis = history_samples[t].input_cis[vli];

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci = vl_input_cis[visible_column_index];

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi_start = num_dendrites_per_column * (offset.y + diam * (offset.x + diam * (in_ci + vld.size.z * hidden_column_index)));

                for (int di = 0; di < num_dendrites_per_column; di++) {
                    int dendrite_index = di + dendrites_start;

                    int wi = di + wi_start;

                    dendrite_acts[dendrite_index] += vl.weights[wi];
                }
            }
    }

    for (int di = 0; di < num_dendrites_per_column; di++) {
        int dendrite_index = di + dendrites_start;

        dendrite_acts[dendrite_index] = max(0.0f, (dendrite_acts[dendrite_index] / (count * 255) - 0.5f) * 2.0f * params.scale);
    }

    float max_activation = limit_min;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int wi_start = num_dendrites_per_column * hidden_cell_index;

        float activation = 0.0f;

        for (int di = 0; di < num_dendrites_per_column; di++) {
            int dendrite_index = di + dendrites_start;

            int wi = di + wi_start;

            activation += dendrite_weights[wi] * dendrite_acts[dendrite_index];
        }

        hidden_acts[hidden_cell_index] = activation;

        max_activation = max(max_activation, activation);
    }

    float value = max_activation_next;

    for (int n = params.n_steps; n >= 1; n--)
        value = history_samples[t - n].reward + params.discount * value;

    float value_prev = hidden_acts[target_ci + hidden_cells_start];

    float td_error = value - value_prev;

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

    for (int di = 0; di < num_dendrites_per_column; di++) {
        int dendrite_index = di + dendrites_start;

        // use as error accum temporarily
        dendrite_deltas[dendrite_index] = 0.0f;
    }

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        float hidden_error = (params.cons * ((hc == target_ci) - hidden_acts[hidden_cell_index]) + (hc == target_ci) * td_error);

        float hidden_delta = params.dlr * hidden_error;

        int wi_start = num_dendrites_per_column * hidden_cell_index;

        for (int di = 0; di < num_dendrites_per_column; di++) {
            int dendrite_index = di + dendrites_start;

            int wi = di + wi_start;

            dendrite_deltas[dendrite_index] += hidden_error * dendrite_weights[wi];

            dendrite_weights[wi] += hidden_delta * dendrite_acts[dendrite_index];
        }
    }

    // scale by lr and for bytes
    for (int di = 0; di < num_dendrites_per_column; di++) {
        int dendrite_index = di + dendrites_start;

        dendrite_deltas[dendrite_index] = params.wlr * 255.0f * dendrite_deltas[dendrite_index] * (dendrite_acts[dendrite_index] > 0.0f);
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

        Int_Buffer_View vl_input_cis = history_samples[t].input_cis[vli];

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci_prev = vl_input_cis[visible_column_index];

                int visible_cell_index = in_ci_prev + visible_column_index * vld.size.z;

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi_start = num_dendrites_per_column * (offset.y + diam * (offset.x + diam * (in_ci_prev + vld.size.z * hidden_column_index)));

                for (int di = 0; di < num_dendrites_per_column; di++) {
                    int dendrite_index = di + dendrites_start;

                    int wi = di + wi_start;

                    vl.weights[wi] = min(255, max(0, vl.weights[wi] + rand_roundf(dendrite_deltas[dendrite_index], state)));
                }
            }
    }
}

void Actor::init_random(
    const Int3 &hidden_size,
    int num_dendrites_per_column,
    int history_capacity,
    const Array<Visible_Layer_Desc> &visible_layer_descs
) {
    this->visible_layer_descs = visible_layer_descs;

    this->hidden_size = hidden_size;
    this->num_dendrites_per_column = num_dendrites_per_column;

    visible_layers.resize(visible_layer_descs.size());

    // pre-compute dimensions
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;
    int num_dendrites = num_hidden_columns * num_dendrites_per_column;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;
        int num_visible_cells = num_visible_columns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(num_dendrites * area * vld.size.z);

        for (int i = 0; i < vl.weights.size(); i++)
            vl.weights[i] = rand() % 128 + 64;
    }

    // hidden cis
    hidden_cis = Int_Buffer(num_hidden_columns, 0);

    dendrite_acts = Float_Buffer(num_dendrites, 0.0f);
    hidden_acts = Float_Buffer(num_hidden_cells, 0.0f);

    dendrite_deltas.resize(num_dendrites);

    dendrite_weights.resize(num_hidden_cells * num_dendrites_per_column);

    for (int i = 0; i < dendrite_weights.size(); i++)
        dendrite_weights[i] = randf(-init_weight_noisef, init_weight_noisef);

    // create (pre-allocated) history samples
    history_size = 0;
    history_samples.resize(history_capacity);

    for (int i = 0; i < history_samples.size(); i++) {
        history_samples[i].input_cis.resize(visible_layers.size());

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

            int num_visible_columns = vld.size.x * vld.size.y;

            history_samples[i].input_cis[vli] = Int_Buffer(num_visible_columns);
        }

        history_samples[i].hidden_target_cis_prev = Int_Buffer(num_hidden_columns);
    }
}

void Actor::step(
    const Array<Int_Buffer_View> &input_cis,
    Int_Buffer_View hidden_target_cis_prev,
    float reward,
    bool learn_enabled,
    const Params &params
) {
    int num_hidden_columns = hidden_size.x * hidden_size.y;

    unsigned int base_state = rand();

    // forward kernel
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
    if (learn_enabled && history_size > params.n_steps) {
        for (int it = 0; it < params.history_iters; it++) {
            int t = rand() % (history_size - params.n_steps) + params.n_steps;

            base_state = rand();

            PARALLEL_FOR
            for (int i = 0; i < num_hidden_columns; i++) {
                unsigned long state = rand_get_state(base_state + i * rand_subseed_offset);

                learn(Int2(i / hidden_size.y, i % hidden_size.y), t, &state, params);
            }
        }
    }
}

void Actor::clear_state() {
    hidden_cis.fill(0);

    history_size = 0;
}

int Actor::size() const {
    int size = sizeof(Int3) + sizeof(int) + hidden_cis.size() * sizeof(int) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        size += sizeof(Visible_Layer_Desc) + vl.weights.size() * sizeof(Byte);
    }

    size += dendrite_weights.size() * sizeof(float);

    size += 3 * sizeof(int);

    int sample_size = 0;

    const History_Sample &s = history_samples[0];

    for (int vli = 0; vli < visible_layers.size(); vli++)
        sample_size += s.input_cis[vli].size() * sizeof(int);

    sample_size += s.hidden_target_cis_prev.size() * sizeof(int) + sizeof(float);

    size += history_samples.size() * sample_size;

    return size;
}

int Actor::state_size() const {
    int size = hidden_cis.size() * sizeof(int) + 2 * sizeof(int);

    int sample_size = 0;

    const History_Sample &s = history_samples[0];

    for (int vli = 0; vli < visible_layers.size(); vli++)
        sample_size += s.input_cis[vli].size() * sizeof(int);

    sample_size += s.hidden_target_cis_prev.size() * sizeof(int) + sizeof(float);

    size += history_samples.size() * sample_size;

    return size;
}

void Actor::write(
    Stream_Writer &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hidden_size), sizeof(Int3));
    writer.write(reinterpret_cast<const void*>(&num_dendrites_per_column), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));
    
    int num_visible_layers = visible_layers.size();

    writer.write(reinterpret_cast<const void*>(&num_visible_layers), sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(Visible_Layer_Desc));

        writer.write(reinterpret_cast<const void*>(&vl.weights[0]), vl.weights.size() * sizeof(Byte));
    }

    writer.write(reinterpret_cast<const void*>(&dendrite_weights[0]), dendrite_weights.size() * sizeof(float));

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
    reader.read(reinterpret_cast<void*>(&num_dendrites_per_column), sizeof(int));

    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;
    int num_dendrites = num_hidden_columns * num_dendrites_per_column;

    hidden_cis.resize(num_hidden_columns);

    reader.read(reinterpret_cast<void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));

    dendrite_acts.resize(num_dendrites);
    hidden_acts.resize(num_hidden_cells);

    dendrite_deltas.resize(num_dendrites);

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

        vl.weights.resize(num_dendrites * area * vld.size.z);

        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(Byte));
    }

    dendrite_weights.resize(num_hidden_cells * num_dendrites_per_column);

    reader.read(reinterpret_cast<void*>(&dendrite_weights[0]), dendrite_weights.size() * sizeof(float));

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
