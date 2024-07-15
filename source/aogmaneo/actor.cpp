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

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int adv_dendrites_start = adv_num_dendrites_per_cell * hidden_cell_index;

        for (int di = 0; di < adv_num_dendrites_per_cell; di++) {
            int adv_dendrite_index = di + adv_dendrites_start;

            adv_dendrite_acts[adv_dendrite_index] = 0.0f;
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

                for (int hc = 0; hc < hidden_size.z; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int adv_dendrites_start = adv_num_dendrites_per_cell * hidden_cell_index;

                    int adv_wi_start = adv_num_dendrites_per_cell * (hc + hidden_size.z * wi_start_partial);

                    for (int di = 0; di < adv_num_dendrites_per_cell; di++) {
                        int adv_dendrite_index = di + adv_dendrites_start;

                        int wi = di + adv_wi_start;

                        adv_dendrite_acts[adv_dendrite_index] += vl.adv_weights[wi];
                    }
                }
            }
    }

    const int half_adv_num_dendrites_per_cell = adv_num_dendrites_per_cell / 2;
    const float dendrite_scale = sqrtf(1.0f / count);
    const float adv_scale = sqrtf(1.0f / adv_num_dendrites_per_cell);

    int max_index = 0;
    float max_adv = limit_min;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int dendrites_start = adv_num_dendrites_per_cell * hidden_cell_index;

        float adv = 0.0f;

        for (int di = 0; di < adv_num_dendrites_per_cell; di++) {
            int adv_dendrite_index = di + dendrites_start;

            float act = adv_dendrite_acts[adv_dendrite_index] * dendrite_scale;

            adv_dendrite_acts[adv_dendrite_index] = max(act * params.leak, act); // relu

            adv += adv_dendrite_acts[adv_dendrite_index] * ((di >= half_adv_num_dendrites_per_cell) * 2.0f - 1.0f);
        }

        adv *= adv_scale;

        //hidden_advs[hidden_cell_index] = adv;

        if (adv > max_adv) {
            max_adv = adv;
            max_index = hc;
        }
    }

    hidden_cis[hidden_column_index] = max_index;
}

void Actor::learn(
    const Int2 &column_pos,
    int t,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    int target_ci = history_samples[t - 1].hidden_target_cis_prev[hidden_column_index];

    int value_dendrites_start = value_num_dendrites_per_cell * hidden_column_index;

    for (int di = 0; di < value_num_dendrites_per_cell; di++) {
        int value_dendrite_index = di + value_dendrites_start;

        value_dendrite_acts[value_dendrite_index] = 0.0f;
    }

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int adv_dendrites_start = adv_num_dendrites_per_cell * hidden_cell_index;

        for (int di = 0; di < adv_num_dendrites_per_cell; di++) {
            int adv_dendrite_index = di + adv_dendrites_start;

            adv_dendrite_acts[adv_dendrite_index] = 0.0f;
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

        Int_Buffer_View vl_input_cis = history_samples[t - params.n_steps].input_cis[vli];

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci = vl_input_cis[visible_column_index];

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi_start_partial = offset.y + diam * (offset.x + diam * (in_ci + vld.size.z * hidden_column_index));

                int wi_value_start = value_num_dendrites_per_cell * wi_start_partial;

                for (int di = 0; di < value_num_dendrites_per_cell; di++) {
                    int value_dendrite_index = di + value_dendrites_start;

                    int wi = di + wi_value_start;

                    value_dendrite_acts[value_dendrite_index] += vl.value_weights_delayed[wi];
                }

                for (int hc = 0; hc < hidden_size.z; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int adv_dendrites_start = adv_num_dendrites_per_cell * hidden_cell_index;

                    int wi_adv_start = adv_num_dendrites_per_cell * (hc + hidden_size.z * wi_start_partial);

                    for (int di = 0; di < adv_num_dendrites_per_cell; di++) {
                        int adv_dendrite_index = di + adv_dendrites_start;

                        int wi = di + wi_adv_start;

                        adv_dendrite_acts[adv_dendrite_index] += vl.adv_weights_delayed[wi];
                    }
                }
            }
    }

    const int half_value_num_dendrites_per_cell = value_num_dendrites_per_cell / 2;
    const int half_adv_num_dendrites_per_cell = adv_num_dendrites_per_cell / 2;
    const float dendrite_scale = sqrtf(1.0f / count);
    const float value_scale = sqrtf(1.0f / value_num_dendrites_per_cell);
    const float adv_scale = sqrtf(1.0f / adv_num_dendrites_per_cell);

    float value_next = 0.0f;

    for (int di = 0; di < value_num_dendrites_per_cell; di++) {
        int value_dendrite_index = di + value_dendrites_start;

        float act = value_dendrite_acts[value_dendrite_index] * dendrite_scale;

        value_dendrite_acts[value_dendrite_index] = max(act * params.leak, act); // relu

        value_next += value_dendrite_acts[value_dendrite_index] * ((di >= half_value_num_dendrites_per_cell) * 2.0f - 1.0f);
    }

    value_next *= value_scale;

    float max_adv_next = limit_min;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int adv_dendrites_start = adv_num_dendrites_per_cell * hidden_cell_index;

        float adv = 0.0f;

        for (int di = 0; di < adv_num_dendrites_per_cell; di++) {
            int adv_dendrite_index = di + adv_dendrites_start;

            float act = adv_dendrite_acts[adv_dendrite_index] * dendrite_scale;

            adv_dendrite_acts[adv_dendrite_index] = max(act * params.leak, act); // relu

            adv += adv_dendrite_acts[adv_dendrite_index] * ((di >= half_adv_num_dendrites_per_cell) * 2.0f - 1.0f);
        }

        adv *= adv_scale;

        //hidden_advs[hidden_cell_index] = adv;

        max_adv_next = max(max_adv_next, adv);
    }

    for (int di = 0; di < value_num_dendrites_per_cell; di++) {
        int value_dendrite_index = di + value_dendrites_start;

        value_dendrite_acts[value_dendrite_index] = 0.0f;
    }

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int adv_dendrites_start = adv_num_dendrites_per_cell * hidden_cell_index;

        for (int di = 0; di < adv_num_dendrites_per_cell; di++) {
            int adv_dendrite_index = di + adv_dendrites_start;

            adv_dendrite_acts[adv_dendrite_index] = 0.0f;
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

        Int_Buffer_View vl_input_cis = history_samples[t].input_cis[vli];

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci_prev = vl_input_cis[visible_column_index];

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi_start_partial = offset.y + diam * (offset.x + diam * (in_ci_prev + vld.size.z * hidden_column_index));

                int wi_value_start = value_num_dendrites_per_cell * wi_start_partial;

                for (int di = 0; di < value_num_dendrites_per_cell; di++) {
                    int value_dendrite_index = di + value_dendrites_start;

                    int wi = di + wi_value_start;

                    value_dendrite_acts[value_dendrite_index] += vl.value_weights[wi];
                }

                for (int hc = 0; hc < hidden_size.z; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int adv_dendrites_start = adv_num_dendrites_per_cell * hidden_cell_index;

                    int wi_adv_start = adv_num_dendrites_per_cell * (hc + hidden_size.z * wi_start_partial);

                    for (int di = 0; di < adv_num_dendrites_per_cell; di++) {
                        int adv_dendrite_index = di + adv_dendrites_start;

                        int wi = di + wi_adv_start;

                        adv_dendrite_acts[adv_dendrite_index] += vl.adv_weights[wi];
                    }
                }
            }
    }

    float value_prev = 0.0f;

    for (int di = 0; di < value_num_dendrites_per_cell; di++) {
        int value_dendrite_index = di + value_dendrites_start;

        float act = value_dendrite_acts[value_dendrite_index] * dendrite_scale;

        value_dendrite_acts[value_dendrite_index] = max(act * params.leak, act); // relu

        value_prev += value_dendrite_acts[value_dendrite_index] * ((di >= half_value_num_dendrites_per_cell) * 2.0f - 1.0f);
    }

    value_prev *= value_scale;

    float max_adv_prev = limit_min;
    float average_adv_prev = 0.0f;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int adv_dendrites_start = adv_num_dendrites_per_cell * hidden_cell_index;

        float adv = 0.0f;

        for (int di = 0; di < adv_num_dendrites_per_cell; di++) {
            int adv_dendrite_index = di + adv_dendrites_start;

            float act = adv_dendrite_acts[adv_dendrite_index] * dendrite_scale;

            adv_dendrite_acts[adv_dendrite_index] = max(act * params.leak, act); // relu

            adv += adv_dendrite_acts[adv_dendrite_index] * ((di >= half_adv_num_dendrites_per_cell) * 2.0f - 1.0f);
        }

        adv *= adv_scale;

        hidden_advs[hidden_cell_index] = adv;

        max_adv_prev = max(max_adv_prev, adv);
        average_adv_prev += adv;
    }

    average_adv_prev /= hidden_size.z;

    float target_q = value_next;

    for (int n = params.n_steps; n >= 1; n--)
        target_q = history_samples[t - n].reward + params.discount * target_q;

    float adv_prev = hidden_advs[target_ci + hidden_cells_start];

    float q_prev = value_prev + adv_prev - average_adv_prev;

    float td_error = target_q - q_prev;

    float value_delta = params.lr * td_error;

    for (int di = 0; di < value_num_dendrites_per_cell; di++) {
        int value_dendrite_index = di + value_dendrites_start;

        value_dendrite_acts[value_dendrite_index] = value_delta * ((di >= half_value_num_dendrites_per_cell) * 2.0f - 1.0f) * ((value_dendrite_acts[value_dendrite_index] > 0.0f) * (1.0f - params.leak) + params.leak);
    }

    // softmax, no longer actual advantage but softmax version of it stored in hidden_advs
    float total = 0.0f;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;
    
        hidden_probs[hidden_cell_index] = expf(hidden_advs[hidden_cell_index] - max_adv_prev);

        total += hidden_probs[hidden_cell_index];
    }

    float total_inv = 1.0f / max(limit_small, total);

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        hidden_probs[hidden_cell_index] *= total_inv;

        int adv_dendrites_start = adv_num_dendrites_per_cell * hidden_cell_index;

        float error = params.lr * (params.cons * (hidden_probs[hidden_cell_index] - 1.0f) + td_error * ((hc == target_ci) - 1.0f / hidden_size.z));

        for (int di = 0; di < adv_num_dendrites_per_cell; di++) {
            int adv_dendrite_index = di + adv_dendrites_start;

            // re-use as deltas
            adv_dendrite_acts[adv_dendrite_index] = error * ((di >= half_adv_num_dendrites_per_cell) * 2.0f - 1.0f) * ((adv_dendrite_acts[adv_dendrite_index] > 0.0f) * (1.0f - params.leak) + params.leak);
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

        Int_Buffer_View vl_input_cis = history_samples[t].input_cis[vli];

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci_prev = vl_input_cis[visible_column_index];

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi_start_partial = offset.y + diam * (offset.x + diam * (in_ci_prev + vld.size.z * hidden_column_index));

                int wi_value_start = value_num_dendrites_per_cell * wi_start_partial;

                for (int di = 0; di < value_num_dendrites_per_cell; di++) {
                    int value_dendrite_index = di + value_dendrites_start;

                    int wi = di + wi_value_start;

                    vl.value_weights[wi] += value_dendrite_acts[value_dendrite_index];
                }

                for (int hc = 0; hc < hidden_size.z; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int adv_dendrites_start = adv_num_dendrites_per_cell * hidden_cell_index;

                    int wi_adv_start = adv_num_dendrites_per_cell * (hc + hidden_size.z * wi_start_partial);

                    for (int di = 0; di < adv_num_dendrites_per_cell; di++) {
                        int adv_dendrite_index = di + adv_dendrites_start;

                        int wi = di + wi_adv_start;

                        vl.adv_weights[wi] += adv_dendrite_acts[adv_dendrite_index];
                    }
                }
            }
    }
}

void Actor::init_random(
    const Int3 &hidden_size,
    int value_num_dendrites_per_cell,
    int adv_num_dendrites_per_cell,
    int history_capacity,
    const Array<Visible_Layer_Desc> &visible_layer_descs
) {
    this->visible_layer_descs = visible_layer_descs;

    this->hidden_size = hidden_size;
    this->value_num_dendrites_per_cell = value_num_dendrites_per_cell;
    this->adv_num_dendrites_per_cell = adv_num_dendrites_per_cell;

    visible_layers.resize(visible_layer_descs.size());

    // pre-compute dimensions
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;
    int value_num_dendrites = num_hidden_columns * value_num_dendrites_per_cell;
    int adv_num_dendrites = num_hidden_cells * adv_num_dendrites_per_cell;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;
        int num_visible_cells = num_visible_columns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.value_weights.resize(value_num_dendrites * area * vld.size.z);

        for (int i = 0; i < vl.value_weights.size(); i++)
            vl.value_weights[i] = randf(-init_weight_noisef, init_weight_noisef);

        vl.value_weights_delayed = vl.value_weights;

        vl.adv_weights.resize(adv_num_dendrites * area * vld.size.z);

        for (int i = 0; i < vl.adv_weights.size(); i++)
            vl.adv_weights[i] = randf(-init_weight_noisef, init_weight_noisef);

        vl.adv_weights_delayed = vl.adv_weights;
    }

    // hidden cis
    hidden_cis = Int_Buffer(num_hidden_columns, 0);

    value_dendrite_acts.resize(value_num_dendrites);
    adv_dendrite_acts.resize(adv_num_dendrites);
    hidden_advs.resize(num_hidden_cells);
    hidden_probs.resize(num_hidden_cells);

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
    bool learn_enabled,
    float reward,
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

            PARALLEL_FOR
            for (int i = 0; i < num_hidden_columns; i++)
                learn(Int2(i / hidden_size.y, i % hidden_size.y), t, params);
        }

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];

            PARALLEL_FOR
            for (int i = 0; i < vl.value_weights.size(); i++)
                vl.value_weights_delayed[i] += params.delay_rate * (vl.value_weights[i] - vl.value_weights_delayed[i]);

            PARALLEL_FOR
            for (int i = 0; i < vl.adv_weights.size(); i++)
                vl.adv_weights_delayed[i] += params.delay_rate * (vl.adv_weights[i] - vl.adv_weights_delayed[i]);
        }
    }
}

void Actor::clear_state() {
    hidden_cis.fill(0);

    history_size = 0;
}

long Actor::size() const {
    long size = sizeof(Int3) + 2 * sizeof(int) + hidden_cis.size() * sizeof(int) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        size += sizeof(Visible_Layer_Desc) + vl.value_weights.size() * sizeof(float) + vl.adv_weights.size() * sizeof(float);
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
    long size = hidden_cis.size() * sizeof(int) + 2 * sizeof(int);

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

        size += vl.value_weights.size() * sizeof(float) + vl.adv_weights.size() * sizeof(float);
    }

    return size;
}

void Actor::write(
    Stream_Writer &writer
) const {
    writer.write(&hidden_size, sizeof(Int3));
    writer.write(&value_num_dendrites_per_cell, sizeof(int));
    writer.write(&adv_num_dendrites_per_cell, sizeof(int));

    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));

    int num_visible_layers = visible_layers.size();

    writer.write(&num_visible_layers, sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(&vld, sizeof(Visible_Layer_Desc));

        writer.write(&vl.value_weights[0], vl.value_weights.size() * sizeof(float));
        writer.write(&vl.adv_weights[0], vl.adv_weights.size() * sizeof(float));
    }

    writer.write(&history_size, sizeof(int));

    int num_history_samples = history_samples.size();

    writer.write(&num_history_samples, sizeof(int));

    int history_start = history_samples.start;

    writer.write(&history_start, sizeof(int));

    for (int t = 0; t < history_samples.size(); t++) {
        const History_Sample &s = history_samples[t];

        for (int vli = 0; vli < visible_layers.size(); vli++)
            writer.write(&s.input_cis[vli][0], s.input_cis[vli].size() * sizeof(int));

        writer.write(&s.hidden_target_cis_prev[0], s.hidden_target_cis_prev.size() * sizeof(int));

        writer.write(&s.reward, sizeof(float));
    }
}

void Actor::read(
    Stream_Reader &reader
) {
    reader.read(&hidden_size, sizeof(Int3));
    reader.read(&value_num_dendrites_per_cell, sizeof(int));
    reader.read(&adv_num_dendrites_per_cell, sizeof(int));

    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;
    int value_num_dendrites = num_hidden_columns * value_num_dendrites_per_cell;
    int adv_num_dendrites = num_hidden_cells * adv_num_dendrites_per_cell;
    
    hidden_cis.resize(num_hidden_columns);

    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));

    value_dendrite_acts.resize(value_num_dendrites);
    adv_dendrite_acts.resize(adv_num_dendrites);
    hidden_advs.resize(num_hidden_cells);
    hidden_probs.resize(num_hidden_cells);

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

        reader.read(&vl.value_weights[0], vl.value_weights.size() * sizeof(float));

        vl.value_weights_delayed = vl.value_weights;

        vl.adv_weights.resize(adv_num_dendrites * area * vld.size.z);

        reader.read(&vl.adv_weights[0], vl.adv_weights.size() * sizeof(float));

        vl.adv_weights_delayed = vl.adv_weights;
    }

    reader.read(&history_size, sizeof(int));

    int num_history_samples;

    reader.read(&num_history_samples, sizeof(int));

    int history_start;

    reader.read(&history_start, sizeof(int));

    history_samples.resize(num_history_samples);
    history_samples.start = history_start;

    for (int t = 0; t < history_samples.size(); t++) {
        History_Sample &s = history_samples[t];

        s.input_cis.resize(num_visible_layers);

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];

            int num_visible_columns = vld.size.x * vld.size.y;

            s.input_cis[vli].resize(num_visible_columns);

            reader.read(&s.input_cis[vli][0], s.input_cis[vli].size() * sizeof(int));
        }

        s.hidden_target_cis_prev.resize(num_hidden_columns);

        reader.read(&s.hidden_target_cis_prev[0], s.hidden_target_cis_prev.size() * sizeof(int));

        reader.read(&s.reward, sizeof(float));
    }
}

void Actor::write_state(
    Stream_Writer &writer
) const {
    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));

    writer.write(&history_size, sizeof(int));

    int history_start = history_samples.start;

    writer.write(&history_start, sizeof(int));

    for (int t = 0; t < history_samples.size(); t++) {
        const History_Sample &s = history_samples[t];

        for (int vli = 0; vli < visible_layers.size(); vli++)
            writer.write(&s.input_cis[vli][0], s.input_cis[vli].size() * sizeof(int));

        writer.write(&s.hidden_target_cis_prev[0], s.hidden_target_cis_prev.size() * sizeof(int));

        writer.write(&s.reward, sizeof(float));
    }
}

void Actor::read_state(
    Stream_Reader &reader
) {
    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));

    reader.read(&history_size, sizeof(int));

    int history_start;

    reader.read(&history_start, sizeof(int));

    history_samples.start = history_start;

    for (int t = 0; t < history_samples.size(); t++) {
        History_Sample &s = history_samples[t];

        for (int vli = 0; vli < visible_layers.size(); vli++)
            reader.read(&s.input_cis[vli][0], s.input_cis[vli].size() * sizeof(int));

        reader.read(&s.hidden_target_cis_prev[0], s.hidden_target_cis_prev.size() * sizeof(int));

        reader.read(&s.reward, sizeof(float));
    }
}

void Actor::write_weights(
    Stream_Writer &writer
) const {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        writer.write(&vl.value_weights[0], vl.value_weights.size() * sizeof(float));
        writer.write(&vl.adv_weights[0], vl.adv_weights.size() * sizeof(float));
    }
}

void Actor::read_weights(
    Stream_Reader &reader
) {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(&vl.value_weights[0], vl.value_weights.size() * sizeof(float));
        reader.read(&vl.adv_weights[0], vl.adv_weights.size() * sizeof(float));
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
        
            for (int i = 0; i < vl.adv_weights.size(); i++) {
                int d = rand() % actors.size();                

                vl.adv_weights[i] = actors[d]->visible_layers[vli].adv_weights[i];
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
        
            for (int i = 0; i < vl.adv_weights.size(); i++) {
                float total = 0.0f;

                for (int d = 0; d < actors.size(); d++)
                    total += actors[d]->visible_layers[vli].adv_weights[i];

                vl.adv_weights[i] = total / actors.size();
            }
        }

        break;
    }
}
