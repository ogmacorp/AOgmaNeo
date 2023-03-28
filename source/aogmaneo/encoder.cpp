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
    unsigned int* state,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    int max_index = -1;
    float max_activation = 0.0f;

    int max_complete_index = 0;
    float max_complete_activation = 0.0f;

    for (int hc = 0; hc < hidden_commits[hidden_column_index]; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        float sum = 0.0f;
        float total_importance = 0.0f;

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];

            if (!vl.use_input)
                continue;

            if (vl.needs_update) {
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

                int sub_sum = 0;
                int sub_count = 0;

                for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                    for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                        int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                        int in_ci = vl.input_cis[visible_column_index];

                        if (in_ci == -1)
                            continue;

                        Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                        int wi = offset.y + diam * (offset.x + diam * hidden_cell_index);

                        sub_sum += (vl.weight_indices[wi] == in_ci) * vl.weights[wi];
                        sub_count++;
                    }

                if (sub_count == 0) // No input, assume it's fine
                    vl.hidden_partial_acts[hidden_cell_index] = 1.0f;
                else
                    vl.hidden_partial_acts[hidden_cell_index] = (sub_sum / 255.0f) / max(1, sub_count);
            }

            sum += vl.hidden_partial_acts[hidden_cell_index] * vl.importance;
            total_importance += vl.importance;
        }

        sum /= max(0.0001f, total_importance);

        float activation = sum / (params.choice + hidden_totals[hidden_cell_index]);

        if (sum >= params.vigilance) {
            if (activation > max_activation || max_index == -1) {
                max_activation = activation;
                max_index = hc;
            }
        }

        if (activation > max_complete_activation || max_complete_index == -1) {
            max_complete_activation = activation;
            max_complete_index = hc;
        }
    }

    hidden_cis[hidden_column_index] = max_index;

    if (max_index == -1 && hidden_commits[hidden_column_index] < hidden_size.z) {
        // commit
        max_index = hidden_commits[hidden_column_index];

        max_activation = randf(state) * 0.0001f; // small random tie breaker
    }

    learn_cis[hidden_column_index] = max_index;

    hidden_max_acts[hidden_column_index] = max_activation;
}

void Encoder::learn(
    const Int2 &column_pos,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    int learn_ci = learn_cis[hidden_column_index];

    if (learn_ci == -1)
        return;

    float max_activation = hidden_max_acts[hidden_column_index];

    for (int dcx = -params.l_radius; dcx <= params.l_radius; dcx++)
        for (int dcy = -params.l_radius; dcy <= params.l_radius; dcy++) {
            if (dcx == 0 && dcy == 0)
                continue;

            Int2 other_column_pos(column_pos.x + dcx, column_pos.y + dcy);

            if (in_bounds0(other_column_pos, Int2(hidden_size.x, hidden_size.y))) {
                int other_hidden_column_index = address2(other_column_pos, Int2(hidden_size.x, hidden_size.y));

                if (hidden_max_acts[other_hidden_column_index] >= max_activation)
                    return;
            }
        }

    int hidden_cell_index_max = learn_ci + hidden_cells_start;

    bool commit = (hidden_totals[hidden_cell_index_max] == limit_max);

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
        int sub_count = (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1);

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci = vl.input_cis[visible_column_index];

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi = offset.y + diam * (offset.x + diam * hidden_cell_index_max);

                if (vl.use_input) {
                    if (commit) {
                        vl.weight_indices[wi] = in_ci;

                        if (in_ci == -1)
                            vl.weights[wi] = 0;
                    }
                    else if (vl.weight_indices[wi] != in_ci || in_ci == -1)
                        vl.weights[wi] = max(0, vl.weights[wi] - ceilf(params.lr * vl.weights[wi]));
                }

                sub_total += vl.weights[wi];
            }

        total += (sub_total / 255.0f) / sub_count * vl.importance;
        total_importance += vl.importance;
    }

    total /= max(0.0001f, total_importance);

    hidden_totals[hidden_cell_index_max] = total;

    if (commit)
        hidden_commits[hidden_column_index]++;
}

void Encoder::reconstruct(
    const Int2 &column_pos,
    const Int_Buffer* other_commits,
    int vli,
    const Params &params
) {
    Visible_Layer &vl = visible_layers[vli];
    Visible_Layer_Desc &vld = visible_layer_descs[vli];

    int diam = vld.radius * 2 + 1;

    int visible_column_index = address2(column_pos, Int2(vld.size.x, vld.size.y));

    int visible_cells_start = visible_column_index * vld.size.z;

    // projection
    Float2 v_to_h = Float2(static_cast<float>(hidden_size.x) / static_cast<float>(vld.size.x),
        static_cast<float>(hidden_size.y) / static_cast<float>(vld.size.y));

    Float2 h_to_v = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hidden_size.x),
        static_cast<float>(vld.size.y) / static_cast<float>(hidden_size.y));
                
    Int2 reverse_radii(ceilf(v_to_h.x * (vld.radius * 2 + 1) * 0.5f), ceilf(v_to_h.y * (vld.radius * 2 + 1) * 0.5f));

    Int2 hidden_center = project(column_pos, v_to_h);

    // lower corner
    Int2 field_lower_bound(hidden_center.x - reverse_radii.x, hidden_center.y - reverse_radii.y);

    // bounds of receptive field, clamped to input size
    Int2 iter_lower_bound(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
    Int2 iter_upper_bound(min(hidden_size.x - 1, hidden_center.x + reverse_radii.x), min(hidden_size.y - 1, hidden_center.y + reverse_radii.y));
    
    // find current max
    int max_index = -1;
    float max_activation = 0.0f;

    int num_commits = (other_commits == nullptr ? vld.size.z : (*other_commits)[visible_column_index]);

    for (int vc = 0; vc < num_commits; vc++) {
        int visible_cell_index = vc + visible_cells_start;

        int sum = 0;
        int count = 0;

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                Int2 hidden_pos = Int2(ix, iy);

                int hidden_column_index = address2(hidden_pos, Int2(hidden_size.x, hidden_size.y));

                Int2 visible_center = project(hidden_pos, h_to_v);

                if (in_bounds(column_pos, Int2(visible_center.x - vld.radius, visible_center.y - vld.radius), Int2(visible_center.x + vld.radius + 1, visible_center.y + vld.radius + 1))) {
                    Int2 offset(column_pos.x - visible_center.x + vld.radius, column_pos.y - visible_center.y + vld.radius);

                    if (hidden_cis[hidden_column_index] == -1)
                        continue;

                    int hidden_cell_index_max = hidden_cis[hidden_column_index] + hidden_column_index * hidden_size.z;

                    int wi = offset.y + diam * (offset.x + diam * hidden_cell_index_max);

                    sum += (vc == vl.weight_indices[wi]) * vl.weights[wi];
                    count++;
                }
            }

        if (count != 0) {
            float recon = (sum / 255.0f) / count;

            if (recon >= params.min_recon) {
                if (recon > max_activation || max_index == -1) {
                    max_activation = recon;
                    max_index = vc;
                }
            }
        }
    }

    vl.recon_cis[visible_column_index] = max_index;
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

        vl.weight_indices = Int_Buffer(num_hidden_cells * area, -1);
        vl.weights = Byte_Buffer(vl.weight_indices.size(), 255);

        vl.hidden_partial_acts.resize(num_hidden_cells, 0.0f);

        vl.input_cis = Int_Buffer(num_visible_columns, 0);
        vl.recon_cis = Int_Buffer(num_visible_columns, 0);
    }

    hidden_cis = Int_Buffer(num_hidden_columns, -1);

    learn_cis = Int_Buffer(num_hidden_columns, -1);

    hidden_totals = Float_Buffer(num_hidden_cells, limit_max);

    hidden_max_acts.resize(num_hidden_columns);

    hidden_commits = Int_Buffer(num_hidden_columns, 0);
}

void Encoder::set_input_cis(
    const Int_Buffer* input_cis,
    int vli
) {
    Visible_Layer &vl = visible_layers[vli];

    if (input_cis == nullptr)
        vl.use_input = false;
    else {
        vl.use_input = true;
        vl.needs_update = true;
        vl.input_cis = *input_cis;
    }
}

void Encoder::activate(
    const Params &params
) {
    int num_hidden_columns = hidden_size.x * hidden_size.y;

    unsigned int base_state = rand();

    #pragma omp parallel for
    for (int i = 0; i < num_hidden_columns; i++) {
        unsigned int state = base_state + i * 12345;

        forward(Int2(i / hidden_size.y, i % hidden_size.y), &state, params);
    }

    // clear update flags
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        if (vl.use_input)
            vl.needs_update = false;
    }
}

void Encoder::learn(
    const Params &params
) {
    int num_hidden_columns = hidden_size.x * hidden_size.y;

    #pragma omp parallel for
    for (int i = 0; i < num_hidden_columns; i++)
        learn(Int2(i / hidden_size.y, i % hidden_size.y), params);

    // reset update flags
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        vl.needs_update = true;
    }
}

void Encoder::reconstruct(
    const Int_Buffer* other_commits,
    int vli,
    const Params &params
) {
    Visible_Layer &vl = visible_layers[vli];
    const Visible_Layer_Desc &vld = visible_layer_descs[vli];

    int num_visible_columns = vld.size.x * vld.size.y;

    #pragma omp parallel for
    for (int i = 0; i < num_visible_columns; i++)
        reconstruct(Int2(i / vld.size.y, i % vld.size.y), other_commits, vli, params);
}

int Encoder::size() const {
    int size = sizeof(Int3) + hidden_cis.size() * sizeof(int) + hidden_totals.size() * sizeof(float) + hidden_commits.size() * sizeof(int) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += sizeof(Visible_Layer_Desc) + vl.weight_indices.size() * sizeof(int) + vl.weights.size() * sizeof(Byte) + 2 * vl.input_cis.size() * sizeof(int) + sizeof(float);
    }

    return size;
}

int Encoder::state_size() const {
    int size = hidden_cis.size() * sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += 2 * vl.input_cis.size() * sizeof(int);
    }

    return size;
}

void Encoder::write(
    Stream_Writer &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hidden_size), sizeof(Int3));

    writer.write(reinterpret_cast<const void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));

    writer.write(reinterpret_cast<const void*>(&hidden_totals[0]), hidden_totals.size() * sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hidden_commits[0]), hidden_commits.size() * sizeof(int));

    int num_visible_layers = visible_layers.size();

    writer.write(reinterpret_cast<const void*>(&num_visible_layers), sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(Visible_Layer_Desc));

        writer.write(reinterpret_cast<const void*>(&vl.weight_indices[0]), vl.weight_indices.size() * sizeof(int));
        writer.write(reinterpret_cast<const void*>(&vl.weights[0]), vl.weights.size() * sizeof(Byte));

        writer.write(reinterpret_cast<const void*>(&vl.input_cis[0]), vl.input_cis.size() * sizeof(int));
        writer.write(reinterpret_cast<const void*>(&vl.recon_cis[0]), vl.recon_cis.size() * sizeof(int));

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

    learn_cis = Int_Buffer(num_hidden_columns, -1);

    hidden_totals.resize(num_hidden_cells);

    reader.read(reinterpret_cast<void*>(&hidden_totals[0]), hidden_totals.size() * sizeof(float));

    hidden_max_acts.resize(num_hidden_columns);

    hidden_commits.resize(num_hidden_columns);

    reader.read(reinterpret_cast<void*>(&hidden_commits[0]), hidden_commits.size() * sizeof(int));

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

        vl.weight_indices.resize(num_hidden_cells * area);
        vl.weights.resize(vl.weight_indices.size());

        reader.read(reinterpret_cast<void*>(&vl.weight_indices[0]), vl.weight_indices.size() * sizeof(int));
        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(Byte));

        vl.hidden_partial_acts.resize(num_hidden_cells);

        vl.input_cis.resize(num_visible_columns);
        vl.recon_cis.resize(num_visible_columns);

        reader.read(reinterpret_cast<void*>(&vl.input_cis[0]), vl.input_cis.size() * sizeof(int));
        reader.read(reinterpret_cast<void*>(&vl.recon_cis[0]), vl.recon_cis.size() * sizeof(int));

        reader.read(reinterpret_cast<void*>(&vl.importance), sizeof(float));

        vl.use_input = false;
        vl.needs_update = true;
    }
}

void Encoder::write_state(
    Stream_Writer &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        writer.write(reinterpret_cast<const void*>(&vl.input_cis[0]), vl.input_cis.size() * sizeof(int));
        writer.write(reinterpret_cast<const void*>(&vl.recon_cis[0]), vl.recon_cis.size() * sizeof(int));
    }
}

void Encoder::read_state(
    Stream_Reader &reader
) {
    reader.read(reinterpret_cast<void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(reinterpret_cast<void*>(&vl.input_cis[0]), vl.input_cis.size() * sizeof(int));
        reader.read(reinterpret_cast<void*>(&vl.recon_cis[0]), vl.recon_cis.size() * sizeof(int));

        vl.use_input = false;
        vl.needs_update = true;
    }
}
