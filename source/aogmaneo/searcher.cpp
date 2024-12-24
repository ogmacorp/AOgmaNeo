// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "searcher.h"

using namespace aon;

void Searcher::init_random(
    const Int3 &config_size,
    int num_dendrites
) {
    this->config_size = config_size;
    this->num_dendrites = num_dendrites;

    // pre-compute dimensions
    int num_config_columns = config_size.x * config_size.y;
    int num_config_cells = num_config_columns * config_size.z;

    weights.resize(num_dendrites * num_config_cells);

    for (int i = 0; i < weights.size(); i++)
        weights[i] = randf(-init_weight_noisef, init_weight_noisef);

    config_cis = Int_Buffer(num_config_columns, 0);

    temp_cis.resize(num_config_columns);

    dendrite_acts.resize(num_dendrites);

    dendrite_deltas.resize(num_dendrites);

    max_temps.resize(num_config_columns);
}

void Searcher::step(
    Int_Buffer_View margin_cis,
    float reward,
    bool learn_enabled
) {
    int num_config_columns = config_size.x * config_size.y;

    const int half_num_dendrites = num_dendrites / 2;

    // activate dendrites
    for (int di = 0; di < num_dendrites; di++)
        dendrite_acts[di] = 0.0f;

    for (int i = 0; i < num_config_columns; i++) {
        int config_ci = config_cis[i];

        for (int di = 0; di < num_dendrites; di++) {
            float w = weights[di + num_dendrites * (config_ci + config_size.z * i)];

            dendrite_acts[di] += w;
        }
    }

    // reward prediction
    float activation = 0.0f;

    const float dendrite_scale = sqrtf(1.0f / num_config_columns);
    const float activation_scale = sqrtf(1.0f / num_dendrites);

    for (int di = 0; di < num_dendrites; di++) {
        float act = dendrite_acts[di] * dendrite_scale;

        dendrite_acts[di] = max(act * params.leak, act); // relu

        activation += dendrite_acts[di] * ((di >= half_num_dendrites) * 2.0f - 1.0f);
    }

    activation *= activation_scale;

    if (learn_enabled) { // learn from last activation
        float error = reward - activation;

        for (int di = 0; di < num_dendrites; di++)
            dendrite_deltas[di] = params.lr * error * ((di >= half_num_dendrites) * 2.0f - 1.0f) * ((dendrite_acts[di] > 0.0f) * (1.0f - params.leak) + params.leak);

        for (int i = 0; i < num_config_columns; i++) {
            int config_ci = config_cis[i];

            for (int di = 0; di < num_dendrites; di++) {
                int wi = di + num_dendrites * (config_ci + config_size.z * i);

                weights[wi] += dendrite_deltas[di];
            }
        }
    }

    // determine new representation with gradient of 1
    for (int di = 0; di < num_dendrites; di++)
        dendrite_deltas[di] = ((di >= half_num_dendrites) * 2.0f - 1.0f) * ((dendrite_acts[di] > 0.0f) * (1.0f - params.leak) + params.leak);

    for (int i = 0; i < num_config_columns; i++) {
        int config_ci = 0;
        float max_grad = limit_min;

        for (int cc = 0; cc < config_size.z; cc++) {
            float grad = 0.0f;

            for (int di = 0; di < num_dendrites; di++) {
                int wi = di + num_dendrites * (cc + config_size.z * i);

                grad += weights[wi] * dendrite_deltas[di];
            }

            if (grad > max_grad) {
                max_grad = grad;
                config_ci = cc;
            }
        }

        temp_cis[i] = config_ci;
        max_temps[i] = max_grad;
    }

    config_cis = margin_cis;

    for (int c = 0; c < params.max_dist; c++) {
        int max_index = 0;
        float max_temp = limit_min;

        for (int i = 0; i < num_config_columns; i++) {
            if (max_temps[i] > max_temp) {
                max_temp = max_temps[i];
                max_index = i;
            }
        }
        
        config_cis[max_index] = temp_cis[max_index];
        max_temps[max_index] = limit_min; // reset
    }

    for (int c = 0; c < params.num_explore; c++) {
        int rand_index = rand() % config_cis.size();

        config_cis[rand_index] = rand() % config_size.z;    
    }
}

void Searcher::clear_state() {
    config_cis.fill(0);
}

long Searcher::size() const {
    long size = sizeof(Int3) + sizeof(int) + config_cis.size() * sizeof(int) + dendrite_acts.size() * sizeof(float) + sizeof(int);

    size += weights.size() * sizeof(float);

    return size;
}

long Searcher::state_size() const {
    return config_cis.size() * sizeof(int);
}

long Searcher::weights_size() const {
    return weights.size() * sizeof(float);
}

void Searcher::write(
    Stream_Writer &writer
) const {
    writer.write(&config_size, sizeof(Int3));
    writer.write(&num_dendrites, sizeof(int));

    writer.write(&config_cis[0], config_cis.size() * sizeof(int));
    
    writer.write(&weights[0], weights.size() * sizeof(float));
}

void Searcher::read(
    Stream_Reader &reader
) {
    reader.read(&config_size, sizeof(Int3));
    reader.read(&num_dendrites, sizeof(int));

    int num_config_columns = config_size.x * config_size.y;
    int num_config_cells = num_config_columns * config_size.z;

    config_cis.resize(num_config_columns);

    reader.read(&config_cis[0], config_cis.size() * sizeof(int));

    temp_cis.resize(num_config_columns);

    dendrite_acts.resize(num_dendrites);

    dendrite_deltas.resize(num_dendrites);

    weights.resize(num_dendrites * num_config_cells);

    reader.read(&weights[0], weights.size() * sizeof(float));

    max_temps.resize(num_config_columns);
}

void Searcher::write_state(
    Stream_Writer &writer
) const {
    writer.write(&config_cis[0], config_cis.size() * sizeof(int));
}

void Searcher::read_state(
    Stream_Reader &reader
) {
    reader.read(&config_cis[0], config_cis.size() * sizeof(int));
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
