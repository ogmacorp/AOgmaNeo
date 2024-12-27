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
    const Int3 &config_size
) {
    this->config_size = config_size;

    // pre-compute dimensions
    int num_config_columns = config_size.x * config_size.y;
    int num_config_cells = num_config_columns * config_size.z;

    rewards.resize(num_config_cells);

    for (int i = 0; i < rewards.size(); i++)
        rewards[i] = randf(-init_weight_noisef, init_weight_noisef);

    config_cis = Int_Buffer(num_config_columns, 0);

    temp_cis.resize(num_config_columns);

    max_temps.resize(num_config_columns);
}

void Searcher::step(
    Int_Buffer_View margin_cis,
    float reward,
    bool learn_enabled
) {
    int num_config_columns = config_size.x * config_size.y;

    // reward update
    if (learn_enabled) { // learn from last activation
        for (int i = 0; i < num_config_columns; i++) {
            int config_ci = config_cis[i];

            int wi = config_ci + config_size.z * i;

            rewards[wi] += params.lr * (reward - rewards[wi]);
        }
    }

    for (int i = 0; i < num_config_columns; i++) {
        int config_ci = 0;
        float max_reward = limit_min;

        for (int cc = 0; cc < config_size.z; cc++) {
            float r = rewards[cc + config_size.z * i];

            if (r > max_reward) {
                max_reward = r;
                config_ci = cc;
            }
        }

        temp_cis[i] = config_ci;
        max_temps[i] = max_reward;
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
}

void Searcher::clear_state() {
    config_cis.fill(0);
}

long Searcher::size() const {
    long size = sizeof(Int3) + config_cis.size() * sizeof(int);

    size += rewards.size() * sizeof(float);

    return size;
}

long Searcher::state_size() const {
    return config_cis.size() * sizeof(int);
}

long Searcher::weights_size() const {
    return rewards.size() * sizeof(float);
}

void Searcher::write(
    Stream_Writer &writer
) const {
    writer.write(&config_size, sizeof(Int3));

    writer.write(&config_cis[0], config_cis.size() * sizeof(int));
    
    writer.write(&rewards[0], rewards.size() * sizeof(float));
}

void Searcher::read(
    Stream_Reader &reader
) {
    reader.read(&config_size, sizeof(Int3));

    int num_config_columns = config_size.x * config_size.y;
    int num_config_cells = num_config_columns * config_size.z;

    config_cis.resize(num_config_columns);

    reader.read(&config_cis[0], config_cis.size() * sizeof(int));

    temp_cis.resize(num_config_columns);

    rewards.resize(num_config_cells);

    reader.read(&rewards[0], rewards.size() * sizeof(float));

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
    writer.write(&rewards[0], rewards.size() * sizeof(float));
}

void Searcher::read_weights(
    Stream_Reader &reader
) {
    reader.read(&rewards[0], rewards.size() * sizeof(float));
}

void Searcher::merge(
    const Array<Searcher*> &searchers,
    Merge_Mode mode
) {
    switch (mode) {
    case merge_random:
        for (int i = 0; i < rewards.size(); i++) {
            int d = rand() % searchers.size();                

            rewards[i] = searchers[d]->rewards[i];
        }

        break;
    case merge_average:
        for (int i = 0; i < rewards.size(); i++) {
            float total = 0.0f;

            for (int d = 0; d < searchers.size(); d++)
                total += searchers[d]->rewards[i];

            rewards[i] = total / searchers.size();
        }

        break;
    }
}
