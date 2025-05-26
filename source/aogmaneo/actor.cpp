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

            dendrite_ps[dendrite_index] = 0.0f;
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

                int wi_start_partial = hidden_size.z * (offset.y + diam * (offset.x + diam * (in_ci + vld.size.z * hidden_column_index)));

                for (int hc = 0; hc < hidden_size.z; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

                    int wi_start = num_dendrites_per_cell * (hc + wi_start_partial);

                    for (int di = 0; di < num_dendrites_per_cell; di++) {
                        int dendrite_index = di + dendrites_start;

                        int wi = di + wi_start;

                        dendrite_ps[dendrite_index] += vl.p_weights[wi];
                    }
                }
            }
    }

    const int half_num_dendrites_per_cell = num_dendrites_per_cell / 2;
    const float dendrite_scale = sqrtf(1.0f / count);
    const float activation_scale = sqrtf(1.0f / num_dendrites_per_cell);

    float max_p = limit_min;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

        float p = 0.0f;

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + dendrites_start;

            float act = dendrite_ps[dendrite_index] * dendrite_scale;

            dendrite_ps[dendrite_index] = (act > 0.0f); // store derivative

            p += max(0.0f, act) * ((di >= half_num_dendrites_per_cell) * 2.0f - 1.0f);
        }

        p *= activation_scale;

        hidden_ps[hidden_cell_index] = p;

        max_p = max(max_p, p);
    }

    // softmax
    float total = 0.0f;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;
    
        hidden_ps[hidden_cell_index] = expf(hidden_ps[hidden_cell_index] - max_p);

        total += hidden_ps[hidden_cell_index];
    }

    float total_inv = 1.0f / max(limit_small, total);

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        hidden_ps[hidden_cell_index] *= total_inv;
    }

    float cusp = randf(state);

    int select_index = 0;
    float sum_so_far = 0.0f;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        sum_so_far += hidden_ps[hidden_cell_index];

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
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    int target_ci = history_samples[t - 1].hidden_target_cis_prev[hidden_column_index];

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + dendrites_start;

            dendrite_qs[dendrite_index] = 0.0f;
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

                int wi_start_partial = hidden_size.z * (offset.y + diam * (offset.x + diam * (in_ci + vld.size.z * hidden_column_index)));

                for (int hc = 0; hc < hidden_size.z; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

                    int wi_start = num_dendrites_per_cell * (hc + wi_start_partial);

                    for (int di = 0; di < num_dendrites_per_cell; di++) {
                        int dendrite_index = di + dendrites_start;

                        int wi = di + wi_start;

                        dendrite_qs[dendrite_index] += vl.q_weights[wi];
                    }
                }
            }
    }

    const int half_num_dendrites_per_cell = num_dendrites_per_cell / 2;
    const float dendrite_scale = sqrtf(1.0f / count);
    const float activation_scale = sqrtf(1.0f / num_dendrites_per_cell);

    float max_q_next = limit_min;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

        float q = 0.0f;

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + dendrites_start;

            // q
            float act = dendrite_qs[dendrite_index] * dendrite_scale;

            dendrite_qs[dendrite_index] = (act > 0.0f); // store derivative

            q += max(0.0f, act) * ((di >= half_num_dendrites_per_cell) * 2.0f - 1.0f);
        }

        q *= activation_scale;

        hidden_qs[hidden_cell_index] = q;

        max_q_next = max(max_q_next, q);
    }

    // softmax q for AWAC weights
    float next_q = 0.0f;

    {
        float total = 0.0f;

        for (int hc = 0; hc < hidden_size.z; hc++) {
            int hidden_cell_index = hc + hidden_cells_start;
        
            hidden_weights[hidden_cell_index] = expf(hidden_qs[hidden_cell_index] - max_q_next);

            total += hidden_weights[hidden_cell_index];
        }

        float total_inv = 1.0f / max(limit_small, total);

        for (int hc = 0; hc < hidden_size.z; hc++) {
            int hidden_cell_index = hc + hidden_cells_start;

            hidden_weights[hidden_cell_index] *= total_inv;

            next_q += hidden_weights[hidden_cell_index] * hidden_qs[hidden_cell_index];
        }
    }

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + dendrites_start;

            dendrite_qs[dendrite_index] = 0.0f;
            dendrite_ps[dendrite_index] = 0.0f;
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

                int wi_start_partial = hidden_size.z * (offset.y + diam * (offset.x + diam * (in_ci_prev + vld.size.z * hidden_column_index)));

                for (int hc = 0; hc < hidden_size.z; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

                    int wi_start = num_dendrites_per_cell * (hc + wi_start_partial);

                    for (int di = 0; di < num_dendrites_per_cell; di++) {
                        int dendrite_index = di + dendrites_start;

                        int wi = di + wi_start;

                        dendrite_qs[dendrite_index] += vl.q_weights[wi];
                        dendrite_ps[dendrite_index] += vl.p_weights[wi];
                    }
                }
            }
    }

    float max_q_prev = limit_min;
    float max_p_prev = limit_min;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

        float q = 0.0f;
        float p = 0.0f;

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + dendrites_start;

            // q
            {
                float act = dendrite_qs[dendrite_index] * dendrite_scale;

                dendrite_qs[dendrite_index] = (act > 0.0f); // store derivative

                q += max(0.0f, act) * ((di >= half_num_dendrites_per_cell) * 2.0f - 1.0f);
            }

            // p
            {
                float act = dendrite_ps[dendrite_index] * dendrite_scale;

                dendrite_ps[dendrite_index] = (act > 0.0f); // store derivative

                p += max(0.0f, act) * ((di >= half_num_dendrites_per_cell) * 2.0f - 1.0f);
            }
        }

        q *= activation_scale;
        p *= activation_scale;

        hidden_qs[hidden_cell_index] = q;
        hidden_ps[hidden_cell_index] = p;

        max_q_prev = max(max_q_prev, q);
        max_p_prev = max(max_p_prev, p);
    }

    float target_q = next_q;

    for (int n = params.n_steps; n >= 1; n--)
        target_q = history_samples[t - n].reward + params.discount * target_q;

    float q_prev = hidden_qs[target_ci + hidden_cells_start];

    float td_error = target_q - q_prev;

    int hidden_cell_index_target = target_ci + hidden_cells_start;

    // softmax q for AWAC weights
    {
        float total = 0.0f;

        for (int hc = 0; hc < hidden_size.z; hc++) {
            int hidden_cell_index = hc + hidden_cells_start;
        
            hidden_weights[hidden_cell_index] = expf(hidden_qs[hidden_cell_index] - max_q_prev);

            total += hidden_weights[hidden_cell_index];
        }

        float total_inv = 1.0f / max(limit_small, total);

        for (int hc = 0; hc < hidden_size.z; hc++) {
            int hidden_cell_index = hc + hidden_cells_start;

            hidden_weights[hidden_cell_index] *= total_inv;
        }
    }

    float weight = hidden_weights[hidden_cell_index_target] * hidden_size.z; // multiplying by hidden_size.z makes this weight value ~1 upon initialization

    // softmax p
    {
        float total = 0.0f;

        for (int hc = 0; hc < hidden_size.z; hc++) {
            int hidden_cell_index = hc + hidden_cells_start;
        
            hidden_ps[hidden_cell_index] = expf(hidden_ps[hidden_cell_index] - max_p_prev);

            total += hidden_ps[hidden_cell_index];
        }

        float total_inv = 1.0f / max(limit_small, total);

        for (int hc = 0; hc < hidden_size.z; hc++) {
            int hidden_cell_index = hc + hidden_cells_start;

            hidden_ps[hidden_cell_index] *= total_inv;
        }
    }

    float q_error = params.qlr * td_error;

    int dendrites_start_target = num_dendrites_per_cell * hidden_cell_index_target;

    for (int di = 0; di < num_dendrites_per_cell; di++) {
        int dendrite_index = di + dendrites_start_target;

        // re-use as deltas
        dendrite_qs[dendrite_index] = q_error * ((di >= half_num_dendrites_per_cell) * 2.0f - 1.0f) * dendrite_qs[dendrite_index];
    }

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

        float p_error = params.plr * weight * ((hc == target_ci) - hidden_ps[hidden_cell_index]);

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + dendrites_start;

            // re-use as deltas
            dendrite_ps[dendrite_index] = p_error * ((di >= half_num_dendrites_per_cell) * 2.0f - 1.0f) * dendrite_ps[dendrite_index];
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

                int wi_start_partial = hidden_size.z * (offset.y + diam * (offset.x + diam * (in_ci_prev + vld.size.z * hidden_column_index)));

                int wi_start_target = num_dendrites_per_cell * (target_ci + wi_start_partial);

                for (int di = 0; di < num_dendrites_per_cell; di++) {
                    int dendrite_index = di + dendrites_start_target;

                    int wi = di + wi_start_target;

                    vl.q_weights[wi] += dendrite_qs[dendrite_index];
                }

                for (int hc = 0; hc < hidden_size.z; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

                    int wi_start = num_dendrites_per_cell * (hc + wi_start_partial);

                    for (int di = 0; di < num_dendrites_per_cell; di++) {
                        int dendrite_index = di + dendrites_start;

                        int wi = di + wi_start;

                        vl.p_weights[wi] += dendrite_ps[dendrite_index];
                    }
                }
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
        const Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;
        int num_visible_cells = num_visible_columns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.q_weights.resize(num_dendrites * area * vld.size.z);
        vl.p_weights.resize(vl.q_weights.size());

        for (int i = 0; i < vl.q_weights.size(); i++) {
            vl.q_weights[i] = randf(-init_weight_noisef, init_weight_noisef);
            vl.p_weights[i] = randf(-init_weight_noisef, init_weight_noisef);
        }
    }

    // hidden cis
    hidden_cis = Int_Buffer(num_hidden_columns, 0);

    hidden_qs.resize(num_hidden_cells);
    hidden_ps.resize(num_hidden_cells);

    hidden_weights.resize(num_hidden_cells);

    dendrite_qs.resize(num_dendrites);
    dendrite_ps.resize(num_dendrites);

    hidden_qs.resize(num_hidden_cells);
    hidden_ps.resize(num_hidden_cells);

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
    if (learn_enabled && history_size > params.n_steps) {
        for (int it = 0; it < params.history_iters; it++) {
            int t = rand() % (history_size - params.n_steps) + params.n_steps;

            PARALLEL_FOR
            for (int i = 0; i < num_hidden_columns; i++)
                learn(Int2(i / hidden_size.y, i % hidden_size.y), t, params);
        }
    }
}

void Actor::clear_state() {
    hidden_cis.fill(0);

    history_size = 0;
}

long Actor::size() const {
    long size = sizeof(Int3) + sizeof(int) + hidden_cis.size() * sizeof(int) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        size += sizeof(Visible_Layer_Desc) + 2 * vl.q_weights.size() * sizeof(float);
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
    long size = hidden_cis.size() * sizeof(int) + sizeof(int);

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

        size += 2 * vl.q_weights.size() * sizeof(float);
    }

    return size;
}

void Actor::write(
    Stream_Writer &writer
) const {
    writer.write(&hidden_size, sizeof(Int3));
    writer.write(&num_dendrites_per_cell, sizeof(int));

    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));

    int num_visible_layers = visible_layers.size();

    writer.write(&num_visible_layers, sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(&vld, sizeof(Visible_Layer_Desc));

        writer.write(&vl.q_weights[0], vl.q_weights.size() * sizeof(float));
        writer.write(&vl.p_weights[0], vl.p_weights.size() * sizeof(float));
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
    reader.read(&num_dendrites_per_cell, sizeof(int));

    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;
    int num_dendrites = num_hidden_cells * num_dendrites_per_cell;
    
    hidden_cis.resize(num_hidden_columns);

    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));

    hidden_qs.resize(num_hidden_cells);
    hidden_ps.resize(num_hidden_cells);

    hidden_weights.resize(num_hidden_cells);

    dendrite_qs.resize(num_dendrites);
    dendrite_ps.resize(num_dendrites);

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

        vl.q_weights.resize(num_dendrites * area * vld.size.z);
        vl.p_weights.resize(vl.q_weights.size());

        reader.read(&vl.q_weights[0], vl.q_weights.size() * sizeof(float));
        reader.read(&vl.p_weights[0], vl.p_weights.size() * sizeof(float));
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

        writer.write(&vl.q_weights[0], vl.q_weights.size() * sizeof(float));
        writer.write(&vl.p_weights[0], vl.p_weights.size() * sizeof(float));
    }
}

void Actor::read_weights(
    Stream_Reader &reader
) {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(&vl.q_weights[0], vl.q_weights.size() * sizeof(float));
        reader.read(&vl.p_weights[0], vl.p_weights.size() * sizeof(float));
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

            for (int i = 0; i < vl.q_weights.size(); i++) {
                int d = rand() % actors.size();                

                vl.q_weights[i] = actors[d]->visible_layers[vli].q_weights[i];
                vl.p_weights[i] = actors[d]->visible_layers[vli].p_weights[i];
            }
        }

        break;
    case merge_average:
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];

            for (int i = 0; i < vl.q_weights.size(); i++) {
                float q_total = 0.0f;
                float p_total = 0.0f;

                for (int d = 0; d < actors.size(); d++) {
                    q_total += actors[d]->visible_layers[vli].q_weights[i];
                    p_total += actors[d]->visible_layers[vli].p_weights[i];
                }

                vl.q_weights[i] = q_total / actors.size();
                vl.p_weights[i] = p_total / actors.size();
            }
        }

        break;
    }
}
