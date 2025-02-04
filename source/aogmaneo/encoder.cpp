// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2025 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "encoder.h"

using namespace aon;

void Encoder::forward_spatial(
    const Int2 &column_pos,
    const Array<Int_Buffer_View> &input_cis,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        hidden_acts[hidden_cell_index] = 0.0f;
    }

    float count = 0.0f;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        if (vl.importance == 0.0f)
            continue;

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

        count += vl.importance * sub_count;

        const float vld_size_z_inv = 1.0f / vld.size.z;

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci = input_cis[vli][visible_column_index];

                float in_value = (in_ci + 0.5f) * vld_size_z_inv;

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi_start = hidden_size.z * (offset.y + diam * (offset.x + diam * hidden_column_index));

                for (int hc = 0; hc < hidden_size.z; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int wi = hc + wi_start;

                    float diff = in_value - vl.protos[wi];

                    hidden_acts[hidden_cell_index] -= abs(diff) * vl.importance;
                }
            }
    }

    int max_index = 0;
    float max_activation = limit_min;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        hidden_acts[hidden_cell_index] /= max(limit_small, count);

        if (hidden_acts[hidden_cell_index] > max_activation) {
            max_activation = hidden_acts[hidden_cell_index];
            max_index = hc;
        }
    }

    hidden_cis[hidden_column_index] = max_index;

    int hidden_cell_index_max = max_index + hidden_cells_start;

    hidden_comparisons[hidden_column_index] = (0.5f + params.choice - hidden_resources[hidden_cell_index_max]) * hidden_acts[hidden_cell_index_max];
}

void Encoder::forward_recurrent(
    const Int2 &column_pos,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int full_column_size = hidden_size.z * temporal_size;

    int temporal_cells_start = hidden_column_index * temporal_size;
    int full_cells_start = hidden_column_index * full_column_size;

    int hidden_ci = hidden_cis[hidden_column_index];

    for (int tc = 0; tc < temporal_size; tc++) {
        int temporal_cell_index = tc + temporal_cells_start;

        temporal_acts[temporal_cell_index] = 0.0f;
    }

    int diam = recurrent_radius * 2 + 1;

    // lower corner
    Int2 field_lower_bound(column_pos.x - recurrent_radius, column_pos.y - recurrent_radius);

    // bounds of receptive field, clamped to input size
    Int2 iter_lower_bound(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
    Int2 iter_upper_bound(min(hidden_size.x - 1, column_pos.x + recurrent_radius), min(hidden_size.y - 1, column_pos.y + recurrent_radius));

    int count = (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1);

    const float full_column_size_inv = 1.0f / full_column_size;

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            int other_hidden_column_index = address2(Int2(ix, iy), Int2(hidden_size.x, hidden_size.y));

            int in_ci = temporal_cis_prev[other_hidden_column_index];
            int hidden_ci_prev = hidden_cis_prev[other_hidden_column_index];

            float in_value = ((in_ci % temporal_size) + 0.5f) * temporal_size;

            Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

            for (int tc = 0; tc < temporal_size; tc++) {
                int temporal_cell_index = tc + temporal_cells_start;

                int fc = tc + hidden_ci * temporal_size;

                int wi = fc + full_column_size * (offset.y + diam * (offset.x + diam * (hidden_ci_prev + hidden_size.z * hidden_column_index)));

                float diff = in_value - recurrent_protos[wi];

                temporal_acts[temporal_cell_index] -= abs(diff);
            }
        }

    int max_index = 0;
    float max_activation = limit_min;

    const float byte_inv = 1.0f / 255.0f;

    for (int tc = 0; tc < temporal_size; tc++) {
        int temporal_cell_index = tc + temporal_cells_start;

        int full_cell_index = tc + hidden_ci * temporal_size + full_cells_start;

        temporal_acts[temporal_cell_index] /= count;

        if (temporal_acts[temporal_cell_index] > max_activation) {
            max_activation = temporal_acts[temporal_cell_index];
            max_index = tc;
        }
    }

    temporal_cis[hidden_column_index] = max_index + hidden_ci * temporal_size;
}

void Encoder::learn(
    const Int2 &column_pos,
    const Array<Int_Buffer_View> &input_cis,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int full_column_size = hidden_size.z * temporal_size;

    int hidden_cells_start = hidden_column_index * hidden_size.z;
    int temporal_cells_start = hidden_column_index * temporal_size;
    int full_cells_start = hidden_column_index * full_column_size;

    int hidden_ci = hidden_cis[hidden_column_index];

    int hidden_cell_index_max = hidden_ci + hidden_cells_start;

    float hidden_max = hidden_comparisons[hidden_column_index];

    int num_higher = 0;
    int count = 1; // 1 since we skip looping over self

    for (int dcx = -params.l_radius; dcx <= params.l_radius; dcx++)
        for (int dcy = -params.l_radius; dcy <= params.l_radius; dcy++) {
            if (dcx == 0 && dcy == 0)
                continue;

            Int2 other_column_pos(column_pos.x + dcx, column_pos.y + dcy);

            if (in_bounds0(other_column_pos, Int2(hidden_size.x, hidden_size.y))) {
                int other_hidden_column_index = address2(other_column_pos, Int2(hidden_size.x, hidden_size.y));

                if (hidden_comparisons[other_hidden_column_index] >= hidden_max)
                    num_higher++;

                count++;
            }
        }

    float ratio = static_cast<float>(num_higher) / static_cast<float>(count);

    if (ratio <= params.active_ratio) {
        for (int dhc = -params.n_radius; dhc <= params.n_radius; dhc++) {
            int hc = hidden_ci + dhc;

            if (hc < 0 || hc >= hidden_size.z)
                continue;

            int hidden_cell_index = hc + hidden_cells_start;

            float rate = hidden_resources[hidden_cell_index] * powf(params.falloff, abs(dhc));

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

                const float vld_size_z_inv = 1.0f / vld.size.z;

                for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                    for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                        int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                        int in_ci = input_cis[vli][visible_column_index];

                        Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                        int wi = hc + hidden_size.z * (offset.y + diam * (offset.x + diam * hidden_column_index));

                        float in_value = (in_ci + 0.5f) * vld_size_z_inv;

                        float diff = in_value - vl.protos[wi];

                        vl.protos[wi] += rate * diff;
                    }
            }

            hidden_resources[hidden_cell_index] -= params.lr * rate;
        }
    }

    // recurrent
    int temporal_ci = temporal_cis[hidden_column_index];

    const float full_column_size_inv = 1.0f / full_column_size;

    for (int dfc = -params.n_radius; dfc <= params.n_radius; dfc++) {
        int fc = temporal_ci + dfc;

        if (fc < 0 || fc >= full_column_size || (fc / temporal_size) != hidden_ci)
            continue;

        int full_cell_index = fc + full_cells_start;

        float rate = temporal_resources[full_cell_index] * powf(params.falloff, abs(dfc));

        int diam = recurrent_radius * 2 + 1;

        // lower corner
        Int2 field_lower_bound(column_pos.x - recurrent_radius, column_pos.y - recurrent_radius);

        // bounds of receptive field, clamped to input size
        Int2 iter_lower_bound(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
        Int2 iter_upper_bound(min(hidden_size.x - 1, column_pos.x + recurrent_radius), min(hidden_size.y - 1, column_pos.y + recurrent_radius));

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int other_hidden_column_index = address2(Int2(ix, iy), Int2(hidden_size.x, hidden_size.y));

                int in_ci = temporal_cis_prev[other_hidden_column_index];
                int hidden_ci_prev = hidden_cis_prev[other_hidden_column_index];

                float in_value = ((in_ci % temporal_size) + 0.5f) * temporal_size;

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi = fc + full_column_size * (offset.y + diam * (offset.x + diam * (hidden_ci_prev + hidden_size.z * hidden_column_index)));
                
                float diff = in_value - recurrent_protos[wi];

                recurrent_protos[wi] += rate * diff;
            }

        temporal_resources[full_cell_index] -= params.lr * rate;
    }
}

void Encoder::init_random(
    const Int3 &hidden_size,
    int temporal_size,
    int recurrent_radius,
    const Array<Visible_Layer_Desc> &visible_layer_descs
) {
    this->visible_layer_descs = visible_layer_descs;

    this->hidden_size = hidden_size;
    this->temporal_size = temporal_size;
    this->recurrent_radius = recurrent_radius;

    visible_layers.resize(visible_layer_descs.size());

    int full_column_size = hidden_size.z * temporal_size;

    // pre-compute dimensions
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;
    int num_temporal_cells = num_hidden_columns * temporal_size;
    int num_full_cells = num_hidden_columns * full_column_size;

    // create layers
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.protos.resize(num_hidden_cells * area);

        for (int i = 0; i < vl.protos.size(); i++)
            vl.protos[i] = randf();
    }

    hidden_cis = Int_Buffer(num_hidden_columns, 0);
    hidden_cis_prev.resize(num_hidden_columns);
    temporal_cis = Int_Buffer(num_hidden_columns, 0);
    temporal_cis_prev.resize(num_hidden_columns);

    hidden_resources = Float_Buffer(num_hidden_cells, 0.5f);
    temporal_resources = Float_Buffer(num_full_cells, 0.5f);

    hidden_acts.resize(num_hidden_cells);
    temporal_acts.resize(num_full_cells);

    hidden_comparisons.resize(num_hidden_columns);

    int diam = recurrent_radius * 2 + 1;
    int area = diam * diam;

    recurrent_protos.resize(num_full_cells * area * hidden_size.z);

    for (int i = 0; i < recurrent_protos.size(); i++)
        recurrent_protos[i] = randf();
}

void Encoder::step(
    const Array<Int_Buffer_View> &input_cis,
    bool learn_enabled,
    const Params &params
) {
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    
    hidden_cis_prev = hidden_cis;
    temporal_cis_prev = temporal_cis;

    PARALLEL_FOR
    for (int i = 0; i < num_hidden_columns; i++)
        forward_spatial(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, params);

    PARALLEL_FOR
    for (int i = 0; i < num_hidden_columns; i++)
        forward_recurrent(Int2(i / hidden_size.y, i % hidden_size.y), params);

    if (learn_enabled) {
        PARALLEL_FOR
        for (int i = 0; i < num_hidden_columns; i++)
            learn(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, params);
    }
}

void Encoder::clear_state() {
    hidden_cis.fill(0);
    temporal_cis.fill(0);
}

long Encoder::size() const {
    long size = sizeof(Int3) + 2 * sizeof(int) + 2 * hidden_cis.size() * sizeof(int) + hidden_resources.size() * sizeof(float) + temporal_resources.size() * sizeof(float) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += sizeof(Visible_Layer_Desc) + vl.protos.size() * sizeof(float) + sizeof(float);
    }

    size += recurrent_protos.size() * sizeof(float);

    return size;
}

long Encoder::state_size() const {
    return 2 * hidden_cis.size() * sizeof(int);
}

long Encoder::weights_size() const {
    long size = 0;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += vl.protos.size() * sizeof(float);
    }

    size += recurrent_protos.size() * sizeof(float);

    return size;
}

void Encoder::write(
    Stream_Writer &writer
) const {
    writer.write(&hidden_size, sizeof(Int3));
    writer.write(&temporal_size, sizeof(int));
    writer.write(&recurrent_radius, sizeof(int));

    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    writer.write(&temporal_cis[0], temporal_cis.size() * sizeof(int));

    writer.write(&hidden_resources[0], hidden_resources.size() * sizeof(float));
    writer.write(&temporal_resources[0], temporal_resources.size() * sizeof(float));

    int num_visible_layers = visible_layers.size();

    writer.write(&num_visible_layers, sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(&vld, sizeof(Visible_Layer_Desc));

        writer.write(&vl.protos[0], vl.protos.size() * sizeof(float));

        writer.write(&vl.importance, sizeof(float));
    }

    writer.write(&recurrent_protos[0], recurrent_protos.size() * sizeof(float));
}

void Encoder::read(
    Stream_Reader &reader
) {
    reader.read(&hidden_size, sizeof(Int3));
    reader.read(&temporal_size, sizeof(int));
    reader.read(&recurrent_radius, sizeof(int));

    int full_column_size = hidden_size.z * temporal_size;

    // pre-compute dimensions
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;
    int num_temporal_cells = num_hidden_columns * temporal_size;
    int num_full_cells = num_hidden_columns * full_column_size;

    hidden_cis.resize(num_hidden_columns);
    temporal_cis.resize(num_hidden_columns);

    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    reader.read(&temporal_cis[0], temporal_cis.size() * sizeof(int));

    hidden_cis_prev.resize(num_hidden_columns);
    temporal_cis_prev.resize(num_hidden_columns);

    hidden_resources.resize(num_hidden_cells);
    temporal_resources.resize(num_full_cells);

    reader.read(&hidden_resources[0], hidden_resources.size() * sizeof(float));
    reader.read(&temporal_resources[0], temporal_resources.size() * sizeof(float));

    hidden_acts.resize(num_hidden_cells);
    temporal_acts.resize(num_full_cells);

    hidden_comparisons.resize(num_hidden_columns);

    int num_visible_layers = visible_layers.size();

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

        vl.protos.resize(num_hidden_cells * area);

        reader.read(&vl.protos[0], vl.protos.size() * sizeof(float));

        reader.read(&vl.importance, sizeof(float));
    }

    int diam = recurrent_radius * 2 + 1;
    int area = diam * diam;

    recurrent_protos.resize(num_full_cells * area * hidden_size.z);

    reader.read(&recurrent_protos[0], recurrent_protos.size() * sizeof(float));
}

void Encoder::write_state(
    Stream_Writer &writer
) const {
    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    writer.write(&temporal_cis[0], temporal_cis.size() * sizeof(int));
}

void Encoder::read_state(
    Stream_Reader &reader
) {
    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    reader.read(&temporal_cis[0], temporal_cis.size() * sizeof(int));
}

void Encoder::write_weights(
    Stream_Writer &writer
) const {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        writer.write(&vl.protos[0], vl.protos.size() * sizeof(float));
    }
}

void Encoder::read_weights(
    Stream_Reader &reader
) {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(&vl.protos[0], vl.protos.size() * sizeof(float));
    }
}

void Encoder::merge(
    const Array<Encoder*> &encoders,
    Merge_Mode mode
) {
    switch (mode) {
    case merge_random:
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];
        
            for (int i = 0; i < vl.protos.size(); i++) {
                int e = rand() % encoders.size();                

                vl.protos[i] = encoders[e]->visible_layers[vli].protos[i];
            }
        }

        break;
    case merge_average:
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];
        
            for (int i = 0; i < vl.protos.size(); i++) {
                float total = 0.0f;

                for (int e = 0; e < encoders.size(); e++)
                    total += encoders[e]->visible_layers[vli].protos[i];

                vl.protos[i] = total / encoders.size();
            }
        }

        break;
    }
}
