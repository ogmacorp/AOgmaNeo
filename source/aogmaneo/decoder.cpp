// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "decoder.h"

using namespace aon;

void Decoder::forward(
    const Int2 &column_pos,
    const Array<const Int_Buffer*> &input_cis,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        hidden_sums[hidden_cell_index] = 0;
    }

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        // projection
        Float2 h_to_v = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hidden_size.x),
            static_cast<float>(vld.size.y) / static_cast<float>(hidden_size.y));

        Int2 visible_center = project(column_pos, h_to_v);

        // lower corner
        Int2 field_lower_bound(visible_center.x - vld.radius, visible_center.y - vld.radius);

        // bounds of receptive field, clamped to input size
        Int2 iter_lower_bound(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
        Int2 iter_upper_bound(min(vld.size.x - 1, visible_center.x + vld.radius), min(vld.size.y - 1, visible_center.y + vld.radius));

        int receptive_size_x = (iter_upper_bound.x - iter_lower_bound.x + 1);
        int receptive_size_y = (iter_upper_bound.y - iter_lower_bound.y + 1);

        int count = receptive_size_x * receptive_size_y;

        // need at least 2 columns
        if (count < 2)
            continue;

        int num_cell_combinations = vld.size.z * vld.size.z;

        int num_column_combinations = area * (area - 1) / 2; // do not include diagonal

        int num_weights_per_cell = num_column_combinations * num_cell_combinations;

        const Int_Buffer &vl_input_cis = *input_cis[vli];

        // loop through bit pairs
        for (int j = 1; j < count; j++)
            for (int i = 0; i < j; i++) {
                Int2 i_pos(i / receptive_size_y, i % receptive_size_y);
                Int2 j_pos(j / receptive_size_y, j % receptive_size_y);

                Int2 i_pos_full(i_pos.x + iter_lower_bound.x, i_pos.y + iter_lower_bound.y);
                Int2 j_pos_full(j_pos.x + iter_lower_bound.x, j_pos.y + iter_lower_bound.y);

                int i_visible_column_index = address2(i_pos_full, Int2(vld.size.x, vld.size.y));
                int j_visible_column_index = address2(j_pos_full, Int2(vld.size.x, vld.size.y));

                int i_in_ci = vl_input_cis[i_visible_column_index];
                int j_in_ci = vl_input_cis[j_visible_column_index];

                int cell_combination = i_in_ci + j_in_ci * vld.size.z;
                int column_combination = i + j * (j - 1) / 2; // do not include diagonal

                int pair_address = cell_combination + num_cell_combinations * column_combination;

                int wi_start = hidden_size.z * (pair_address + num_weights_per_cell * hidden_column_index);

                for (int hc = 0; hc < hidden_size.z; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int wi = hc + wi_start;

                    int byi = wi / 8;
                    int bi = wi % 8;

                    hidden_sums[hidden_cell_index] += ((vl.weights[byi] & (0x1 << bi)) != 0);
                }
            }
    }

    int max_index = 0;
    int max_activation = 0;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int sum = hidden_sums[hidden_cell_index];

        if (sum > max_activation) {
            max_activation = sum;
            max_index = hc;
        }
    }

    hidden_cis[hidden_column_index] = max_index;
}

void Decoder::learn(
    const Int2 &column_pos,
    const Int_Buffer* hidden_target_cis,
    unsigned long* state,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    int target_ci = (*hidden_target_cis)[hidden_column_index];
    int hidden_ci = hidden_cis[hidden_column_index];

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        // projection
        Float2 h_to_v = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hidden_size.x),
            static_cast<float>(vld.size.y) / static_cast<float>(hidden_size.y));

        Int2 visible_center = project(column_pos, h_to_v);

        // lower corner
        Int2 field_lower_bound(visible_center.x - vld.radius, visible_center.y - vld.radius);

        // bounds of receptive field, clamped to input size
        Int2 iter_lower_bound(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
        Int2 iter_upper_bound(min(vld.size.x - 1, visible_center.x + vld.radius), min(vld.size.y - 1, visible_center.y + vld.radius));

        int receptive_size_x = (iter_upper_bound.x - iter_lower_bound.x + 1);
        int receptive_size_y = (iter_upper_bound.y - iter_lower_bound.y + 1);

        int count = receptive_size_x * receptive_size_y;

        // need at least 2 columns
        if (count < 2)
            continue;

        int num_cell_combinations = vld.size.z * vld.size.z;

        int num_column_combinations = area * (area - 1) / 2; // do not include diagonal

        int num_weights_per_cell = num_column_combinations * num_cell_combinations;

        // loop through bit pairs
        for (int j = 1; j < count; j++)
            for (int i = 0; i < j; i++) {
                Int2 i_pos(i / receptive_size_y, i % receptive_size_y);
                Int2 j_pos(j / receptive_size_y, j % receptive_size_y);

                Int2 i_pos_full(i_pos.x + iter_lower_bound.x, i_pos.y + iter_lower_bound.y);
                Int2 j_pos_full(j_pos.x + iter_lower_bound.x, j_pos.y + iter_lower_bound.y);

                int i_visible_column_index = address2(i_pos_full, Int2(vld.size.x, vld.size.y));
                int j_visible_column_index = address2(j_pos_full, Int2(vld.size.x, vld.size.y));

                int i_in_ci = vl.input_cis_prev[i_visible_column_index];
                int j_in_ci = vl.input_cis_prev[j_visible_column_index];

                int cell_combination = i_in_ci + j_in_ci * vld.size.z;
                int column_combination = i + j * (j - 1) / 2; // do not include diagonal

                int pair_address = cell_combination + num_cell_combinations * column_combination;

                int wi_start = hidden_size.z * (pair_address + num_weights_per_cell * hidden_column_index);

                if (hidden_ci != target_ci && randf(state) < params.forget) {
                    int wi = hidden_ci + wi_start;

                    int byi = wi / 8;
                    int bi = wi % 8;

                    vl.weights[byi] &= ~(0x1 << bi);
                }

                {
                    int wi = target_ci + wi_start;

                    int byi = wi / 8;
                    int bi = wi % 8;

                    vl.weights[byi] |= (0x1 << bi);
                }
            }
    }
}

void Decoder::init_random(
    const Int3 &hidden_size,
    const Array<Visible_Layer_Desc> &visible_layer_descs
) {
    this->visible_layer_descs = visible_layer_descs; 

    this->hidden_size = hidden_size;

    visible_layers.resize(visible_layer_descs.size());

    // pre-compute dimensions
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;
    
    // create layers
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;
        int num_visible_cells = num_visible_columns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        // column pairs
        int num_cell_combinations = vld.size.z * vld.size.z;

        int num_column_combinations = area * (area - 1) / 2; // do not include diagonal

        int num_weights_per_cell = num_column_combinations * num_cell_combinations;

        vl.weights = Byte_Buffer((num_hidden_cells * num_weights_per_cell + 7) / 8, 0);

        vl.input_cis_prev = Int_Buffer(num_visible_columns, 0);
    }

    // hidden cis
    hidden_cis = Int_Buffer(num_hidden_columns, 0);

    hidden_sums.resize(num_hidden_cells);
}

void Decoder::step(
    const Array<const Int_Buffer*> &input_cis,
    const Int_Buffer* hidden_target_cis,
    bool learn_enabled,
    const Params &params
) {
    int num_hidden_columns = hidden_size.x * hidden_size.y;

    if (learn_enabled) {
        unsigned int base_state = rand();

        // learn kernel
        PARALLEL_FOR
        for (int i = 0; i < num_hidden_columns; i++) {
            unsigned long state = rand_get_state(base_state + i * rand_subseed_offset);

            learn(Int2(i / hidden_size.y, i % hidden_size.y), hidden_target_cis, &state, params);
        }
    }

    // forward kernel
    PARALLEL_FOR
    for (int i = 0; i < num_hidden_columns; i++)
        forward(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, params);

    // copy to prevs
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        vl.input_cis_prev = *input_cis[vli];
    }
}

void Decoder::clear_state() {
    hidden_cis.fill(0);

    for (int vli = 0; vli < visible_layers.size(); vli++)
        visible_layers[vli].input_cis_prev.fill(0);
}

int Decoder::size() const {
    int size = sizeof(Int3) + hidden_cis.size() * sizeof(int) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        size += sizeof(Visible_Layer_Desc) + vl.weights.size() * sizeof(Byte) + vl.input_cis_prev.size() * sizeof(int);
    }

    return size;
}

int Decoder::state_size() const {
    int size = hidden_cis.size() * sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += vl.input_cis_prev.size() * sizeof(int);
    }

    return size;
}

void Decoder::write(
    Stream_Writer &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hidden_size), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));
    
    int num_visible_layers = visible_layers.size();

    writer.write(reinterpret_cast<const void*>(&num_visible_layers), sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(Visible_Layer_Desc));

        writer.write(reinterpret_cast<const void*>(&vl.weights[0]), vl.weights.size() * sizeof(Byte));

        writer.write(reinterpret_cast<const void*>(&vl.input_cis_prev[0]), vl.input_cis_prev.size() * sizeof(int));
    }
}

void Decoder::read(
    Stream_Reader &reader
) {
    reader.read(reinterpret_cast<void*>(&hidden_size), sizeof(Int3));

    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;

    hidden_cis.resize(num_hidden_columns);

    reader.read(reinterpret_cast<void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));

    hidden_sums.resize(num_hidden_cells);

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

        // column pairs
        int num_cell_combinations = vld.size.z * vld.size.z;

        int num_column_combinations = area * (area - 1) / 2; // do not include diagonal

        int num_weights_per_cell = num_column_combinations * num_cell_combinations;

        vl.weights.resize((num_hidden_cells * num_weights_per_cell + 7) / 8);

        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(Byte));

        vl.input_cis_prev.resize(num_visible_columns);

        reader.read(reinterpret_cast<void*>(&vl.input_cis_prev[0]), vl.input_cis_prev.size() * sizeof(int));
    }
}

void Decoder::write_state(
    Stream_Writer &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        writer.write(reinterpret_cast<const void*>(&vl.input_cis_prev[0]), vl.input_cis_prev.size() * sizeof(int));
    }
}

void Decoder::read_state(
    Stream_Reader &reader
) {
    reader.read(reinterpret_cast<void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(reinterpret_cast<void*>(&vl.input_cis_prev[0]), vl.input_cis_prev.size() * sizeof(int));
    }
}
