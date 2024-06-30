// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "searcher.h"

using namespace aon;

void Searcher::forward(
    const Int2 &column_pos,
    Int_Buffer_View actual_config_cis,
    float reward,
    bool learn_enabled,
    unsigned long* state,
    const Params &params
) {
    int config_column_index = address2(column_pos, Int2(config_size.x, config_size.y));

    int config_cells_start = config_column_index * config_size.z;

    int diam = radius * 2 + 1;

    // lower corner
    Int2 field_lower_bound(column_pos.x - radius, column_pos.y - radius);

    // bounds of receptive field, clamped to input size
    Int2 iter_lower_bound(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
    Int2 iter_upper_bound(min(config_size.x - 1, column_pos.x + radius), min(config_size.y - 1, column_pos.y + radius));

    int count = (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1);

    const int half_num_dendrites_per_cell = num_dendrites_per_cell / 2;
    const float dendrite_scale = sqrtf(1.0f / count);
    const float activation_scale = sqrtf(1.0f / num_dendrites_per_cell);

    if (learn_enabled) {
        int config_cell_index = actual_config_cis[config_column_index] + config_cells_start;

        int dendrites_start = num_dendrites_per_cell * config_cell_index;

        float error = params.lr * (reward - pred_config_acts[pred_config_cis[config_column_index] + config_cells_start]);

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + dendrites_start;

            dendrite_deltas[dendrite_index] = error * ((di >= half_num_dendrites_per_cell) * 2.0f - 1.0f) * ((dendrite_acts[dendrite_index] > 0.0f) * (1.0f - params.leak) + params.leak);
        }

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(config_size.x, config_size.y));

                int in_ci = pred_config_cis[visible_column_index];

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi_start_partial = config_size.z * (offset.y + diam * (offset.x + diam * (in_ci + config_size.z * config_column_index)));

                for (int hc = 0; hc < config_size.z; hc++) {
                    int config_cell_index = hc + config_cells_start;

                    int dendrites_start = num_dendrites_per_cell * config_cell_index;

                    int wi_start = num_dendrites_per_cell * (hc + wi_start_partial);

                    for (int di = 0; di < num_dendrites_per_cell; di++) {
                        int dendrite_index = di + dendrites_start;

                        int wi = di + wi_start;

                        weights[wi] += dendrite_deltas[dendrite_index];
                    }
                }
            }
    }

    for (int hc = 0; hc < config_size.z; hc++) {
        int config_cell_index = hc + config_cells_start;

        int dendrites_start = num_dendrites_per_cell * config_cell_index;

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + dendrites_start;

            dendrite_acts[dendrite_index] = 0.0f;
        }
    }

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            int visible_column_index = address2(Int2(ix, iy), Int2(config_size.x, config_size.y));

            int in_ci = actual_config_cis[visible_column_index];

            Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

            int wi_start_partial = config_size.z * (offset.y + diam * (offset.x + diam * (in_ci + config_size.z * config_column_index)));

            for (int hc = 0; hc < config_size.z; hc++) {
                int config_cell_index = hc + config_cells_start;

                int dendrites_start = num_dendrites_per_cell * config_cell_index;

                int wi_start = num_dendrites_per_cell * (hc + wi_start_partial);

                for (int di = 0; di < num_dendrites_per_cell; di++) {
                    int dendrite_index = di + dendrites_start;

                    int wi = di + wi_start;

                    dendrite_acts[dendrite_index] += weights[wi];
                }
            }
        }

    int max_index = 0;
    float max_activation = limit_min;

    for (int hc = 0; hc < config_size.z; hc++) {
        int config_cell_index = hc + config_cells_start;

        int dendrites_start = num_dendrites_per_cell * config_cell_index;

        float activation = 0.0f;

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + dendrites_start;

            float act = dendrite_acts[dendrite_index] * dendrite_scale;

            dendrite_acts[dendrite_index] = max(act * params.leak, act); // relu

            activation += dendrite_acts[dendrite_index] * ((di >= half_num_dendrites_per_cell) * 2.0f - 1.0f);
        }

        activation *= activation_scale;

        pred_config_acts[config_cell_index] = activation;

        if (activation > max_activation) {
            max_activation = activation;
            max_index = hc;
        }
    }

    if (randf(state) < params.exploration)
        pred_config_cis[config_column_index] = rand(state) % config_size.z;
    else
        pred_config_cis[config_column_index] = max_index;
}

void Searcher::init_random(
    const Int3 &config_size,
    int num_dendrites_per_cell,
    int radius
) {
    this->config_size = config_size;
    this->num_dendrites_per_cell = num_dendrites_per_cell;
    this->radius = radius;

    // pre-compute dimensions
    int num_config_columns = config_size.x * config_size.y;
    int num_config_cells = num_config_columns * config_size.z;
    int num_dendrites = num_config_cells * num_dendrites_per_cell;

    int diam = radius * 2 + 1;
    int area = diam * diam;

    weights.resize(num_dendrites * area * config_size.z);

    for (int i = 0; i < weights.size(); i++)
        weights[i] = randf(-init_weight_noisef, init_weight_noisef);

    pred_config_cis = Int_Buffer(num_config_columns, 0);

    pred_config_acts = Float_Buffer(num_config_cells, 0.0f);

    dendrite_acts.resize(num_dendrites);
}

void Searcher::step(
    Int_Buffer_View actual_config_cis,
    float reward,
    bool learn_enabled,
    const Params &params
) {
    int num_config_columns = config_size.x * config_size.y;

    unsigned int base_state = rand();

    PARALLEL_FOR
    for (int i = 0; i < num_config_columns; i++) {
        unsigned long state = rand_get_state(base_state + i * rand_subseed_offset);

        forward(Int2(i / config_size.y, i % config_size.y), actual_config_cis, reward, learn_enabled, &state, params);
    }
}

void Searcher::clear_state() {
    pred_config_cis.fill(0);
    pred_config_acts.fill(0.0f);
}

long Searcher::size() const {
    long size = sizeof(Int3) + sizeof(int) + pred_config_cis.size() * sizeof(int) + pred_config_acts.size() * sizeof(float) + dendrite_acts.size() * sizeof(float) + sizeof(int);

    size += weights.size() * sizeof(float);

    return size;
}

long Searcher::state_size() const {
    return pred_config_cis.size() * sizeof(int) + pred_config_acts.size() * sizeof(float);
}

long Searcher::weights_size() const {
    return weights.size() * sizeof(float);
}

void Searcher::write(
    Stream_Writer &writer
) const {
    writer.write(&config_size, sizeof(Int3));
    writer.write(&num_dendrites_per_cell, sizeof(int));

    writer.write(&pred_config_cis[0], pred_config_cis.size() * sizeof(int));
    writer.write(&pred_config_acts[0], pred_config_acts.size() * sizeof(float));
    
    writer.write(&weights[0], weights.size() * sizeof(float));
}

void Searcher::read(
    Stream_Reader &reader
) {
    reader.read(&config_size, sizeof(Int3));
    reader.read(&num_dendrites_per_cell, sizeof(int));

    int num_config_columns = config_size.x * config_size.y;
    int num_config_cells = num_config_columns * config_size.z;
    int num_dendrites = num_config_cells * num_dendrites_per_cell;

    pred_config_cis.resize(num_config_columns);
    pred_config_acts.resize(num_config_cells);

    reader.read(&pred_config_cis[0], pred_config_cis.size() * sizeof(int));
    reader.read(&pred_config_acts[0], pred_config_acts.size() * sizeof(float));

    dendrite_acts.resize(num_dendrites);

    int diam = radius * 2 + 1;
    int area = diam * diam;

    weights.resize(num_dendrites * area * config_size.z);

    reader.read(&weights[0], weights.size() * sizeof(float));
}

void Searcher::write_state(
    Stream_Writer &writer
) const {
    writer.write(&pred_config_cis[0], pred_config_cis.size() * sizeof(int));
    writer.write(&pred_config_acts[0], pred_config_acts.size() * sizeof(float));
}

void Searcher::read_state(
    Stream_Reader &reader
) {
    reader.read(&pred_config_cis[0], pred_config_cis.size() * sizeof(int));
    reader.read(&pred_config_acts[0], pred_config_acts.size() * sizeof(float));
}

void Searcher::write_weights(
    Stream_Writer &writer
) const {
    writer.write(&weights[0], weights.size() * sizeof(float));
}

void Searcher::read_weights(
    Stream_Reader &reader
) {
    reader.read(&weights[0], weights.size() * sizeof(float));
}

void Searcher::merge(
    const Array<Searcher*> &searchers,
    Merge_Mode mode
) {
    switch (mode) {
    case merge_random:
        for (int i = 0; i < weights.size(); i++) {
            int d = rand() % searchers.size();                

            weights[i] = searchers[d]->weights[i];
        }

        break;
    case merge_average:
        for (int i = 0; i < weights.size(); i++) {
            float total = 0.0f;

            for (int d = 0; d < searchers.size(); d++)
                total += searchers[d]->weights[i];

            weights[i] = total / searchers.size();
        }

        break;
    }
}
