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
    const Array<Int_Buffer_View> &input_cis,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    int num_hidden_columns = hidden_size.x * hidden_size.y;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        hidden_sums[hidden_cell_index] = 0;
    }

    int total_num_combinations_used = 0;
    int count = column_addresses.size();

    int num_column_combinations = count * (count - 1) / 2;

    for (int j = 1; j < count; j++)
        for (int i = 0; i < j; i++) {
            Int3 i_address = column_addresses[i];
            Int3 j_address = column_addresses[j];

            Visible_Layer &i_vl = visible_layers[i_address.z];
            const Visible_Layer_Desc &i_vld = visible_layer_descs[i_address.z];

            Visible_Layer &j_vl = visible_layers[j_address.z];
            const Visible_Layer_Desc &j_vld = visible_layer_descs[j_address.z];

            // projection
            Float2 i_h_to_v = Float2(static_cast<float>(i_vld.size.x) / static_cast<float>(hidden_size.x),
                static_cast<float>(i_vld.size.y) / static_cast<float>(hidden_size.y));

            Float2 j_h_to_v = Float2(static_cast<float>(j_vld.size.x) / static_cast<float>(hidden_size.x),
                static_cast<float>(j_vld.size.y) / static_cast<float>(hidden_size.y));

            Int2 i_visible_center = project(column_pos, i_h_to_v);
            Int2 j_visible_center = project(column_pos, j_h_to_v);

            Int2 i_pos(i_visible_center.x + i_address.x, i_visible_center.y + i_address.y);

            if (!in_bounds0(i_pos, Int2(i_vld.size.x, i_vld.size.y)))
                continue;

            Int2 j_pos(j_visible_center.x + j_address.x, j_visible_center.y + j_address.y);

            if (!in_bounds0(j_pos, Int2(j_vld.size.x, j_vld.size.y)))
                continue;

            int i_in_ci = input_cis[i_address.z][address2(i_pos, Int2(i_vld.size.x, i_vld.size.y))];
            int j_in_ci = input_cis[j_address.z][address2(j_pos, Int2(j_vld.size.x, j_vld.size.y))];

            int column_combination = i + j * (j - 1) / 2; // do not include diagonal

            int index = i_in_ci + max_vld_size_z * (j_in_ci + max_vld_size_z * column_combination);

            int wi_start = hidden_size.z * (hidden_column_index + num_hidden_columns * index);

            for (int hc = 0; hc < hidden_size.z; hc++) {
                int hidden_cell_index = hc + hidden_cells_start;

                int wi = hc + wi_start;

                hidden_sums[hidden_cell_index] += weights[wi];
            }

            total_num_combinations_used++;
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

    float total = 0.0f;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        hidden_acts[hidden_cell_index] = expf(static_cast<float>(hidden_sums[hidden_cell_index] - max_activation) / (total_num_combinations_used * 255) * params.scale);

        total += hidden_acts[hidden_cell_index];
    }

    float total_inv = 1.0f / max(limit_small, total);

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        hidden_acts[hidden_cell_index] *= total_inv;
    }

    hidden_cis[hidden_column_index] = max_index;
}

void Decoder::learn(
    const Int2 &column_pos,
    Int_Buffer_View hidden_target_cis,
    unsigned long* state,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    int num_hidden_columns = hidden_size.x * hidden_size.y;

    int target_ci = hidden_target_cis[hidden_column_index];
    int hidden_ci = hidden_cis[hidden_column_index];

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        hidden_deltas[hidden_cell_index] = rand_roundf(params.lr * 255.0f * ((hc == target_ci) - hidden_acts[hidden_cell_index]), state);
    }

    int count = column_addresses.size();

    for (int j = 1; j < count; j++)
        for (int i = 0; i < j; i++) {
            Int3 i_address = column_addresses[i];
            Int3 j_address = column_addresses[j];

            Visible_Layer &i_vl = visible_layers[i_address.z];
            const Visible_Layer_Desc &i_vld = visible_layer_descs[i_address.z];

            Visible_Layer &j_vl = visible_layers[j_address.z];
            const Visible_Layer_Desc &j_vld = visible_layer_descs[j_address.z];

            // projection
            Float2 i_h_to_v = Float2(static_cast<float>(i_vld.size.x) / static_cast<float>(hidden_size.x),
                static_cast<float>(i_vld.size.y) / static_cast<float>(hidden_size.y));

            Float2 j_h_to_v = Float2(static_cast<float>(j_vld.size.x) / static_cast<float>(hidden_size.x),
                static_cast<float>(j_vld.size.y) / static_cast<float>(hidden_size.y));

            Int2 i_visible_center = project(column_pos, i_h_to_v);
            Int2 j_visible_center = project(column_pos, j_h_to_v);

            Int2 i_pos(i_visible_center.x + i_address.x, i_visible_center.y + i_address.y);

            if (!in_bounds0(i_pos, Int2(i_vld.size.x, i_vld.size.y)))
                continue;

            Int2 j_pos(j_visible_center.x + j_address.x, j_visible_center.y + j_address.y);

            if (!in_bounds0(j_pos, Int2(j_vld.size.x, j_vld.size.y)))
                continue;

            int i_in_ci = i_vl.input_cis_prev[address2(i_pos, Int2(i_vld.size.x, i_vld.size.y))];
            int j_in_ci = j_vl.input_cis_prev[address2(j_pos, Int2(j_vld.size.x, j_vld.size.y))];

            unsigned long column_combination = i + j * (j - 1) / 2; // do not include diagonal

            unsigned long weight_index = i_in_ci + max_vld_size_z * (j_in_ci + max_vld_size_z * column_combination);

            int index = i_in_ci + max_vld_size_z * (j_in_ci + max_vld_size_z * column_combination);

            int wi_start = hidden_size.z * (hidden_column_index + num_hidden_columns * index);

            for (int hc = 0; hc < hidden_size.z; hc++) {
                int hidden_cell_index = hc + hidden_cells_start;

                int wi = hc + wi_start;

                weights[wi] = min(255, max(0, weights[wi] + hidden_deltas[hidden_cell_index]));
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
    
    int count = 0;
    max_vld_size_z = 0;

    // create layers
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

        max_vld_size_z = max(max_vld_size_z, vld.size.z);

        int num_visible_columns = vld.size.x * vld.size.y;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        count += area;

        vl.input_cis_prev = Int_Buffer(num_visible_columns, 0);
    }

    assert(count >= 2); // Need at least 2 columns of input

    column_addresses.resize(count);

    // assign combinations
    int running_count = 0;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

        for (int dx = -vld.radius; dx <= vld.radius; dx++) {
            for (int dy = -vld.radius; dy <= vld.radius; dy++) {
                column_addresses[running_count] = Int3(dx, dy, vli);

                running_count++;
            }
        }
    }
    
    // hidden cis
    hidden_cis = Int_Buffer(num_hidden_columns, 0);

    hidden_sums.resize(num_hidden_cells);

    hidden_acts = Float_Buffer(num_hidden_cells, 0.0f);

    hidden_deltas.resize(num_hidden_cells);

    int num_locations = num_hidden_cells * count * max_vld_size_z;

    weights.resize(num_hidden_cells * num_locations);

    for (int i = 0; i < weights.size(); i++)
        weights[i] = 127 + (rand() % init_weight_noisei) - init_weight_noisei / 2;
}

void Decoder::step(
    const Array<Int_Buffer_View> &input_cis,
    Int_Buffer_View hidden_target_cis,
    bool learn_enabled,
    const Params &params
) {
    int num_hidden_columns = hidden_size.x * hidden_size.y;

    if (learn_enabled) {
        // learn kernel
        unsigned int base_state = rand();

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

        vl.input_cis_prev = input_cis[vli];
    }
}

void Decoder::clear_state() {
    hidden_cis.fill(0);

    for (int vli = 0; vli < visible_layers.size(); vli++)
        visible_layers[vli].input_cis_prev.fill(0);
}

long Decoder::size() const {
    long size = sizeof(Int3) + hidden_cis.size() * sizeof(int) + hidden_acts.size() * sizeof(float) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        size += sizeof(Visible_Layer_Desc) + vl.input_cis_prev.size() * sizeof(int);
    }

    size += weights.size() * sizeof(Byte);

    return size;
}

long Decoder::state_size() const {
    long size = hidden_cis.size() * sizeof(int) + hidden_acts.size() * sizeof(float);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += vl.input_cis_prev.size() * sizeof(int);
    }

    return size;
}

long Decoder::weights_size() const {
    return weights.size() * sizeof(Byte);
}

void Decoder::write(
    Stream_Writer &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hidden_size), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hidden_acts[0]), hidden_acts.size() * sizeof(float));
    
    int num_visible_layers = visible_layers.size();

    writer.write(reinterpret_cast<const void*>(&num_visible_layers), sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(Visible_Layer_Desc));

        writer.write(reinterpret_cast<const void*>(&vl.input_cis_prev[0]), vl.input_cis_prev.size() * sizeof(int));
    }

    writer.write(reinterpret_cast<const void*>(&column_addresses[0]), column_addresses.size() * sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&weights[0]), weights.size() * sizeof(Byte));
}

void Decoder::read(
    Stream_Reader &reader
) {
    reader.read(reinterpret_cast<void*>(&hidden_size), sizeof(Int3));

    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;

    hidden_cis.resize(num_hidden_columns);
    hidden_acts.resize(num_hidden_cells);

    reader.read(reinterpret_cast<void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hidden_acts[0]), hidden_acts.size() * sizeof(float));

    hidden_sums.resize(num_hidden_cells);

    int num_visible_layers;

    reader.read(reinterpret_cast<void*>(&num_visible_layers), sizeof(int));

    visible_layers.resize(num_visible_layers);
    visible_layer_descs.resize(num_visible_layers);

    int count = 0;
    max_vld_size_z = 0;
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        Visible_Layer_Desc &vld = visible_layer_descs[vli];

        reader.read(reinterpret_cast<void*>(&vld), sizeof(Visible_Layer_Desc));

        max_vld_size_z = max(max_vld_size_z, vld.size.z);

        int num_visible_columns = vld.size.x * vld.size.y;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        count += area;

        vl.input_cis_prev.resize(num_visible_columns);

        reader.read(reinterpret_cast<void*>(&vl.input_cis_prev[0]), vl.input_cis_prev.size() * sizeof(int));
    }

    column_addresses.resize(count);

    reader.read(reinterpret_cast<void*>(&column_addresses[0]), column_addresses.size() * sizeof(Int3));

    int num_locations = num_hidden_cells * count * max_vld_size_z;

    weights.resize(num_hidden_cells * num_locations);

    reader.read(reinterpret_cast<void*>(&weights[0]), weights.size() * sizeof(Byte));
}

void Decoder::write_state(
    Stream_Writer &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&hidden_acts[0]), hidden_acts.size() * sizeof(float));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        writer.write(reinterpret_cast<const void*>(&vl.input_cis_prev[0]), vl.input_cis_prev.size() * sizeof(int));
    }
}

void Decoder::read_state(
    Stream_Reader &reader
) {
    reader.read(reinterpret_cast<void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&hidden_acts[0]), hidden_acts.size() * sizeof(float));

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(reinterpret_cast<void*>(&vl.input_cis_prev[0]), vl.input_cis_prev.size() * sizeof(int));
    }
}

void Decoder::write_weights(
    Stream_Writer &writer
) const {
    writer.write(reinterpret_cast<const void*>(&weights[0]), weights.size() * sizeof(Byte));
}

void Decoder::read_weights(
    Stream_Reader &reader
) {
    reader.read(reinterpret_cast<void*>(&weights[0]), weights.size() * sizeof(Byte));
}

void Decoder::merge(
    const Array<Decoder*> &decoders,
    Merge_Mode mode
) {
    switch (mode) {
    case merge_random:
        for (int i = 0; i < weights.size(); i++) {
            int d = rand() % decoders.size();                

            weights[i] = decoders[d]->weights[i];
        }

        break;
    case merge_average:
        for (int i = 0; i < weights.size(); i++) {
            float total = 0.0f;

            for (int d = 0; d < decoders.size(); d++)
                total += decoders[d]->weights[i];

            weights[i] = roundf(total / decoders.size());
        }

        break;
    }
}
