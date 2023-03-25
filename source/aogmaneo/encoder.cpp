// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "encoder.h"

using namespace aon;

void Encoder::forward(
    const Int2 &column_pos,
    const Array<const Int_Buffer*> &input_cis,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    int max_index = -1;
    float max_activation = limit_min;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        float sum = 0.0f;

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

            float sub_sum = 0.0f;
            int sub_count = (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1);

            for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                    int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                    int in_ci = (*input_cis[vli])[visible_column_index];

                    Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                    int wi = offset.y + diam * (offset.x + diam * hidden_cell_index);

                    float in_value = (in_ci + 0.5f) / vld.size.z;

                    float delta = in_value - vl.protos[wi];

                    sub_sum -= delta * delta;
                }

            sum += sub_sum / sub_count * vl.importance;
        }

        if (sum > max_activation || max_index == -1) {
            max_activation = sum;
            max_index = hc;
        }
    }

    hidden_max_acts[hidden_column_index] = max_activation;

    hidden_cis[hidden_column_index] = max_index;
}

void Encoder::inhibit(
    const Int2 &column_pos,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    hidden_peaks[hidden_column_index] = true;

    float max_activation = hidden_max_acts[hidden_column_index];

    for (int dcx = -params.l_radius; dcx <= params.l_radius; dcx++)
        for (int dcy = -params.l_radius; dcy <= params.l_radius; dcy++) {
            if (dcx == 0 && dcy == 0)
                continue;

            Int2 other_column_pos(column_pos.x + dcx, column_pos.y + dcy);

            if (in_bounds0(other_column_pos, Int2(hidden_size.x, hidden_size.y))) {
                int other_hidden_column_index = address2(other_column_pos, Int2(hidden_size.x, hidden_size.y));

                if (hidden_max_acts[other_hidden_column_index] >= max_activation) {
                    hidden_peaks[hidden_column_index] = false;
                    return;
                }
            }
        }
}

void Encoder::learn(
    const Int2 &column_pos,
    const Array<const Int_Buffer*> &input_cis,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    // Search for peaks in range
    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++) {
            Int2 search_pos(column_pos.x + dx, column_pos.y + dy);

            if (in_bounds0(search_pos, Int2(hidden_size.x, hidden_size.y))) {
                if (hidden_peaks[address2(search_pos, Int2(hidden_size.x, hidden_size.y))])
                    goto learn;
            }
        }

    return;

learn:
    for (int dhc = -1; dhc <= 1; dhc++) {
        int hc = hidden_cis[hidden_column_index] + dhc;

        if (hc < 0 || hc >= hidden_size.z)
            continue;

        int hidden_cell_index = hc + hidden_cells_start;

        float rate = hidden_rates[hidden_cell_index];

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

            for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                    int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                    int in_ci = (*input_cis[vli])[visible_column_index];

                    Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                    int wi = offset.y + diam * (offset.x + diam * hidden_cell_index);

                    float in_value = (in_ci + 0.5f) / vld.size.z;

                    vl.protos[wi] += rate * (in_value - vl.protos[wi]);
                }
        }

        hidden_rates[hidden_cell_index] *= 1.0f - params.lr;
    }
}

void Encoder::init_random(
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

        vl.protos.resize(num_hidden_cells * area * vld.size.z);

        for (int i = 0; i < vl.protos.size(); i++)
            vl.protos[i] = randf(0.0f, 1.0f);
    }

    hidden_cis = Int_Buffer(num_hidden_columns, 0);

    hidden_max_acts.resize(num_hidden_columns);

    hidden_rates = Float_Buffer(num_hidden_cells, 0.5f);

    hidden_peaks.resize(num_hidden_columns);
}

void Encoder::step(
    const Array<const Int_Buffer*> &input_cis,
    bool learn_enabled,
    const Params &params
) {
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    
    #pragma omp parallel for
    for (int i = 0; i < num_hidden_columns; i++)
        forward(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, params);

    if (learn_enabled) {
        #pragma omp parallel for
        for (int i = 0; i < num_hidden_columns; i++)
            inhibit(Int2(i / hidden_size.y, i % hidden_size.y), params);

        #pragma omp parallel for
        for (int i = 0; i < num_hidden_columns; i++)
            learn(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, params);
    }
}

int Encoder::size() const {
    int size = sizeof(Int3) + hidden_cis.size() * sizeof(int) + hidden_rates.size() * sizeof(float) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += sizeof(Visible_Layer_Desc) + vl.protos.size() * sizeof(float) + sizeof(float);
    }

    return size;
}

int Encoder::state_size() const {
    return hidden_cis.size() * sizeof(int);
}

void Encoder::write(
    Stream_Writer &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hidden_size), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));

    writer.write(reinterpret_cast<const void*>(&hidden_rates[0]), hidden_rates.size() * sizeof(float));

    int num_visible_layers = visible_layers.size();

    writer.write(reinterpret_cast<const void*>(&num_visible_layers), sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(Visible_Layer_Desc));

        writer.write(reinterpret_cast<const void*>(&vl.protos[0]), vl.protos.size() * sizeof(float));

        writer.write(reinterpret_cast<const void*>(&vl.importance), sizeof(float));
    }
}

void Encoder::read(
    Stream_Reader &reader
) {
    reader.read(reinterpret_cast<void*>(&hidden_size), sizeof(Int3));

    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;

    hidden_cis.resize(num_hidden_columns);

    reader.read(reinterpret_cast<void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));

    hidden_max_acts.resize(num_hidden_columns);

    hidden_rates.resize(num_hidden_cells);

    reader.read(reinterpret_cast<void*>(&hidden_rates[0]), hidden_rates.size() * sizeof(float));

    hidden_peaks.resize(num_hidden_columns);

    int num_visible_layers = visible_layers.size();

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

        vl.protos.resize(num_hidden_cells * area);

        reader.read(reinterpret_cast<void*>(&vl.protos[0]), vl.protos.size() * sizeof(float));

        reader.read(reinterpret_cast<void*>(&vl.importance), sizeof(float));
    }
}

void Encoder::write_state(
    Stream_Writer &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));
}

void Encoder::read_state(
    Stream_Reader &reader
) {
    reader.read(reinterpret_cast<void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));
}
