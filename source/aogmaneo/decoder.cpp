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

    int num_dendrites_per_column = hidden_size.z * num_dendrites;

    int hidden_cells_start = hidden_column_index * hidden_size.z;
    int hidden_dendrites_start = hidden_column_index * num_dendrites_per_column;

    for (int di = 0; di < num_dendrites_per_column; di++) {
        int hidden_dendritic_index = di + hidden_dendrites_start;

        hidden_matches[hidden_dendritic_index] = 0.0f;
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

        int sub_count = (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1);

        float influence = 1.0f / (sub_count * 255);

        const Int_Buffer &vl_input_cis = *input_cis[vli];

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci = vl_input_cis[visible_column_index];

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi_start = num_dendrites_per_column * (offset.y + diam * (offset.x + diam * (in_ci + vld.size.z * hidden_column_index)));

                for (int di = 0; di < num_dendrites_per_column; di++) {
                    int hidden_dendritic_index = di + hidden_dendrites_start;

                    int wi = di + wi_start;

                    hidden_matches[hidden_dendritic_index] += vl.weights[wi] * influence;
                }
            }
    }

    for (int di = 0; di < num_dendrites_per_column; di++) {
        int hidden_dendritic_index = di + hidden_dendrites_start;

        hidden_matches[hidden_dendritic_index] /= visible_layers.size();
    }

    int max_overall_index = 0;
    float max_overall_match = 0.0f;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int hidden_dendritic_cells_start = num_dendrites * hidden_cell_index;

        int max_index = -1;
        float max_activation = 0.0f;

        int max_complete_index = 0;
        float max_complete_match = 0.0f;

        for (int di = 0; di < num_dendrites; di++) {
            int hidden_dendritic_index = di + hidden_dendritic_cells_start;

            float match = hidden_matches[hidden_dendritic_index];

            float activation = match / (params.choice + hidden_totals[hidden_dendritic_index]);

            if (match >= params.vigilance) {
                if (activation > max_activation) {
                    max_activation = activation;
                    max_index = di;
                }
            }

            if (match > max_complete_match) {
                max_complete_match = match;
                max_complete_index = di;
            }
        }

        learn_dis[hidden_cell_index] = max_index;

        if (max_complete_match > max_overall_match) {
            max_overall_match = max_complete_match;
            max_overall_index = hc;
        }
    }

    hidden_cis[hidden_column_index] = max_overall_index;
}

void Decoder::learn(
    const Int2 &column_pos,
    const Int_Buffer* hidden_target_cis,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int num_dendrites_per_column = hidden_size.z * num_dendrites;

    int hidden_cells_start = hidden_column_index * hidden_size.z;
    int hidden_dendrites_start = hidden_column_index * num_dendrites_per_column;

    int target_ci = (*hidden_target_cis)[hidden_column_index];

    int hidden_cell_index_target = target_ci + hidden_cells_start;

    int learn_di = learn_dis[hidden_cell_index_target];

    if (learn_di == -1)
        return;

    int hidden_dendritic_index = learn_di + hidden_dendrites_start;

    float total = 0.0f;
    float total_importance = 0.0f;

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

        int sub_total = 0;
        int sub_count = (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1) * vld.size.z;

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci = vl.input_cis_prev[visible_column_index];

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                for (int vc = 0; vc < vld.size.z; vc++) {
                    int wi = learn_di + num_dendrites * (target_ci + hidden_size.z * (offset.y + diam * (offset.x + diam * (vc + vld.size.z * hidden_column_index))));

                    if (vc != in_ci)
                        vl.weights[wi] = max(0, vl.weights[wi] - ceilf(params.lr * vl.weights[wi]));

                    sub_total += vl.weights[wi];
                }
            }

        total += static_cast<float>(sub_total) / (sub_count * 255);
    }

    total /= visible_layers.size();

    hidden_totals[hidden_dendritic_index] = total;
}

void Decoder::init_random(
    const Int3 &hidden_size,
    int num_dendrites,
    const Array<Visible_Layer_Desc> &visible_layer_descs
) {
    this->visible_layer_descs = visible_layer_descs; 

    this->hidden_size = hidden_size;
    this->num_dendrites = num_dendrites;

    visible_layers.resize(visible_layer_descs.size());

    // pre-compute dimensions
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;
    int num_hidden_dendrites = num_hidden_cells * num_dendrites;
    
    // create layers
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;
        int num_visible_cells = num_visible_columns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(num_hidden_dendrites * area * vld.size.z);

        for (int i = 0; i < vl.weights.size(); i++)
            vl.weights[i] = 255 - (rand() % init_weight_noise);

        vl.input_cis_prev = Int_Buffer(num_visible_columns, 0);
    }

    // hidden cis
    hidden_cis = Int_Buffer(num_hidden_columns, 0);

    learn_dis.resize(num_hidden_cells);

    hidden_matches.resize(num_hidden_dendrites);

    hidden_totals = Float_Buffer(num_hidden_dendrites, 1.0f);
}

void Decoder::activate(
    const Array<const Int_Buffer*> &input_cis,
    const Params &params
) {
    int num_hidden_columns = hidden_size.x * hidden_size.y;

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

void Decoder::learn(
    const Int_Buffer* hidden_target_cis,
    const Params &params
) {
    int num_hidden_columns = hidden_size.x * hidden_size.y;

    // learn kernel
    PARALLEL_FOR
    for (int i = 0; i < num_hidden_columns; i++)
        learn(Int2(i / hidden_size.y, i % hidden_size.y), hidden_target_cis, params);
}

void Decoder::clear_state() {
    hidden_cis.fill(0);

    for (int vli = 0; vli < visible_layers.size(); vli++)
        visible_layers[vli].input_cis_prev.fill(0);
}

int Decoder::size() const {
    int size = sizeof(Int3) + sizeof(int) + hidden_cis.size() * sizeof(int) + hidden_totals.size() * sizeof(float) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        size += sizeof(Visible_Layer_Desc) + vl.weights.size() * sizeof(Byte) + vl.input_cis_prev.size() * sizeof(int);
    }

    return size;
}

int Decoder::state_size() const {
    int size = hidden_cis.size() * sizeof(int) + hidden_totals.size() * sizeof(float);

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
    writer.write(reinterpret_cast<const void*>(&num_dendrites), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));

    writer.write(reinterpret_cast<const void*>(&hidden_totals[0]), hidden_totals.size() * sizeof(float));
    
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
    reader.read(reinterpret_cast<void*>(&num_dendrites), sizeof(int));

    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;
    int num_hidden_dendrites = num_hidden_cells * num_dendrites;

    hidden_cis.resize(num_hidden_columns);

    reader.read(reinterpret_cast<void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));

    learn_dis.resize(num_hidden_cells);

    hidden_matches.resize(num_hidden_dendrites);

    hidden_totals.resize(num_hidden_dendrites);

    reader.read(reinterpret_cast<void*>(&hidden_totals[0]), hidden_totals.size() * sizeof(float));

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

        vl.weights.resize(num_hidden_dendrites * area * vld.size.z);

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
