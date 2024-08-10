// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "encoder.h"
#include <iostream>

using namespace aon;

void Encoder::bind_inputs(
    const Int2 &column_pos,
    int vli
) {
    Visible_Layer &vl = visible_layers[vli];
    const Visible_Layer_Desc &vld = visible_layer_descs[vli];

    assert(vl.use_input);

    int visible_column_index = address2(column_pos, Int2(vld.size.x, vld.size.y));

    int visible_cell_index_max = vl.input_cis[visible_column_index] + vld.size.z * visible_column_index;

    int visible_vecs_start = vec_size * visible_column_index;
    int visible_codes_start = vec_size * visible_cell_index_max;

    for (int vi = 0; vi < vec_size; vi++) {
        int visible_vec_index = vi + visible_vecs_start;
        int visible_code_index = vi + visible_codes_start;

        vl.input_vecs[visible_vec_index] = vl.visible_code_vecs[visible_code_index] * vl.visible_pos_vecs[visible_vec_index];
    }
}

void Encoder::forward(
    const Int2 &column_pos
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_size.z * hidden_column_index;
    int hidden_vecs_start = vec_size * hidden_column_index;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        if (!vl.use_input)
            continue;

        if (!vl.up_to_date) {
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

            for (int vi = 0; vi < vec_size; vi++) {
                int hidden_vec_index = vi + hidden_vecs_start;

                vl.hidden_bundles[hidden_vec_index] = 0.0f;
            }

            for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                    int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                    int visible_vecs_start = visible_column_index * vec_size;

                    Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                    int wi = offset.y + diam * (offset.x + diam * hidden_column_index);

                    for (int vi = 0; vi < vec_size; vi++) {
                        int hidden_vec_index = vi + hidden_vecs_start;
                        int visible_vec_index = vi + visible_vecs_start;

                        vl.hidden_bundles[hidden_vec_index] += vl.input_vecs[visible_vec_index] * vl.weights[wi];
                    }
                }
        }
    }

    for (int vi = 0; vi < vec_size; vi++) {
        int hidden_vec_index = vi + hidden_vecs_start;

        float hidden_bundle = 0.0f;

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];

            if (!vl.use_input)
                continue;

            hidden_bundle += vl.hidden_bundles[hidden_vec_index] * vl.importance;
        }

        hidden_vecs[hidden_vec_index] = (hidden_bundle > 0.0f) * 2 - 1;
    }

    int max_index = 0;
    int max_similarity = limit_min;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int hidden_codes_start = vec_size * hidden_cell_index;

        int similarity = 0;

        for (int vi = 0; vi < vec_size; vi++) {
            int hidden_vec_index = vi + hidden_vecs_start;
            int hidden_code_index = vi + hidden_codes_start;

            similarity += hidden_vecs[hidden_vec_index] * ((hidden_code_vecs[hidden_code_index] > 0.0f) * 2 - 1);
        }

        if (similarity > max_similarity) {
            max_similarity = similarity;
            max_index = hc;
        }
    }

    hidden_cis[hidden_column_index] = max_index;
    hidden_comparisons[hidden_column_index] = max_similarity;
}

void Encoder::learn(
    const Int2 &column_pos,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_size.z * hidden_column_index;
    int hidden_vecs_start = vec_size * hidden_column_index;

    int hidden_compare = hidden_comparisons[hidden_column_index];

    int num_higher = 0;
    int count = 0;

    for (int dcx = -params.l_radius; dcx <= params.l_radius; dcx++)
        for (int dcy = -params.l_radius; dcy <= params.l_radius; dcy++) {
            if (dcx == 0 && dcy == 0)
                continue;

            Int2 other_column_pos(column_pos.x + dcx, column_pos.y + dcy);

            if (in_bounds0(other_column_pos, Int2(hidden_size.x, hidden_size.y))) {
                int other_hidden_column_index = address2(other_column_pos, Int2(hidden_size.x, hidden_size.y));

                if (hidden_comparisons[other_hidden_column_index] >= hidden_compare)
                    num_higher++;

                count++;
            }
        }

    float ratio = static_cast<float>(num_higher) / static_cast<float>(max(1, count));

    if (ratio > params.active_ratio)
        return;

    int hidden_ci = hidden_cis[hidden_column_index];

    int hidden_cell_index_max = hidden_ci + hidden_cells_start;

    int hidden_codes_start_max = vec_size * hidden_cell_index_max;

    // learn codes
    for (int vi = 0; vi < vec_size; vi++) {
        int hidden_vec_index = vi + hidden_vecs_start;
        int hidden_code_index = vi + hidden_codes_start_max;

         hidden_code_vecs[hidden_code_index] += params.clr * (hidden_vecs[hidden_vec_index] - hidden_code_vecs[hidden_code_index]);
    }

    // learn weights
    const float vec_size_inv = 1.0f / vec_size;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        if (!vl.use_input)
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

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int visible_vecs_start = visible_column_index * vec_size;

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int similarity = 0;

                for (int vi = 0; vi < vec_size; vi++) {
                    int visible_vec_index = vi + visible_vecs_start;
                    int hidden_code_index = vi + hidden_codes_start_max;

                    similarity += vl.input_vecs[visible_vec_index] * ((hidden_code_vecs[hidden_code_index] > 0.0f) * 2 - 1);
                }

                int wi = offset.y + diam * (offset.x + diam * hidden_column_index);

                vl.weights[wi] += params.wlr * (similarity * vec_size_inv - vl.weights[wi]);
            }
    }
}

void Encoder::reconstruct(
    const Int2 &column_pos,
    int vli
) {
    Visible_Layer &vl = visible_layers[vli];
    const Visible_Layer_Desc &vld = visible_layer_descs[vli];

    int diam = vld.radius * 2 + 1;

    int visible_column_index = address2(column_pos, Int2(vld.size.x, vld.size.y));

    int visible_cells_start = visible_column_index * vld.size.z;
    int visible_vecs_start = visible_column_index * vec_size;

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

    for (int vi = 0; vi < vec_size; vi++) {
        int visible_vec_index = vi + visible_vecs_start;

        vl.visible_bundles[visible_vec_index] = 0.0f;
    }

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            Int2 hidden_pos = Int2(ix, iy);

            int hidden_column_index = address2(hidden_pos, Int2(hidden_size.x, hidden_size.y));

            Int2 visible_center = project(hidden_pos, h_to_v);

            if (in_bounds(column_pos, Int2(visible_center.x - vld.radius, visible_center.y - vld.radius), Int2(visible_center.x + vld.radius + 1, visible_center.y + vld.radius + 1))) {
                int hidden_cell_index_max = hidden_cis[hidden_column_index] + hidden_column_index * hidden_size.z;

                int hidden_codes_start_max = vec_size * hidden_cell_index_max;

                Int2 offset(column_pos.x - visible_center.x + vld.radius, column_pos.y - visible_center.y + vld.radius);

                int wi = offset.y + diam * (offset.x + diam * hidden_column_index);

                for (int vi = 0; vi < vec_size; vi++) {
                    int visible_vec_index = vi + visible_vecs_start;
                    int hidden_code_index = vi + hidden_codes_start_max;

                    // un-bind and superimpose
                    vl.visible_bundles[visible_vec_index] += ((hidden_code_vecs[hidden_code_index] > 0.0f) * 2 - 1) * vl.weights[wi];
                }
            }
        }

    for (int vi = 0; vi < vec_size; vi++) {
        int visible_vec_index = vi + visible_vecs_start;

        vl.recon_vecs[visible_vec_index] = ((vl.visible_bundles[visible_vec_index] > 0.0f) * 2 - 1) * vl.visible_pos_vecs[visible_vec_index];
    }

    int max_index = 0;
    int max_similarity = limit_min;

    for (int vc = 0; vc < vld.size.z; vc++) {
        int visible_cell_index = vc + visible_cells_start;

        int visible_codes_start = vec_size * visible_cell_index;

        int similarity = 0;

        for (int vi = 0; vi < vec_size; vi++) {
            int visible_vec_index = vi + visible_vecs_start;
            int visible_code_index = vi + visible_codes_start;

            similarity += vl.recon_vecs[visible_vec_index] * vl.visible_code_vecs[visible_code_index];
        }

        if (similarity > max_similarity) {
            max_similarity = similarity;
            max_index = vc;
        }
    }

    vl.recon_cis[visible_column_index] = max_index;
}

void Encoder::init_random(
    const Int3 &hidden_size,
    int vec_size,
    float positional_scale,
    const Array<Visible_Layer_Desc> &visible_layer_descs
) {
    this->visible_layer_descs = visible_layer_descs;

    this->hidden_size = hidden_size;
    this->vec_size = vec_size;

    visible_layers.resize(visible_layer_descs.size());

    // pre-compute dimensions
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;
        int num_visible_cells = num_visible_columns * vld.size.z;

        vl.visible_code_vecs.resize(vec_size * num_visible_cells);

        for (int i = 0; i < vl.visible_code_vecs.size(); i++)
            vl.visible_code_vecs[i] = (rand() % 2) * 2 - 1;

        // generate temporary positional matrix
        Float_Buffer embedding(vec_size * 3);

        for (int i = 0; i < embedding.size(); i++)
            embedding[i] = rand_normalf() * positional_scale;

        vl.visible_pos_vecs.resize(vec_size * num_visible_columns);

        for (int x = 0; x < vld.size.x; x++)
            for (int y = 0; y < vld.size.y; y++) {
                int visible_column_index = y + x * vld.size.y;

                int visible_vecs_start = vec_size * visible_column_index;

                for (int vi = 0; vi < vec_size; vi++) {
                    int visible_vec_index = vi + visible_vecs_start;

                    vl.visible_pos_vecs[visible_vec_index] = (modf(embedding[visible_vec_index * 3] * x + embedding[visible_vec_index * 3 + 1] * y + embedding[visible_vec_index * 3 + 2], pi) > 0.0f) * 2 - 1;
                }
            }

        vl.visible_bundles.resize(vec_size * num_visible_columns);
        vl.hidden_bundles.resize(vec_size * num_hidden_columns);

        vl.input_cis = Int_Buffer(num_visible_columns, 0);
        vl.recon_cis = Int_Buffer(num_visible_columns, 0);

        vl.input_vecs.resize(vec_size * num_visible_columns);
        vl.recon_vecs.resize(vec_size * num_visible_columns);

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(num_hidden_columns * area);

        for (int i = 0; i < vl.weights.size(); i++)
            vl.weights[i] = 1.0f - randf() * init_weight_noisef;
    }

    hidden_cis = Int_Buffer(num_hidden_columns, 0);

    hidden_code_vecs.resize(vec_size * num_hidden_cells);

    for (int i = 0; i < hidden_code_vecs.size(); i++)
        hidden_code_vecs[i] = randf(-init_weight_noisef, init_weight_noisef);

    hidden_vecs.resize(vec_size * num_hidden_columns);

    hidden_comparisons.resize(num_hidden_columns);
}

void Encoder::step(
    bool learn_enabled,
    const Params &params
) {
    int num_hidden_columns = hidden_size.x * hidden_size.y;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        if (!vl.use_input || vl.up_to_date)
            continue;

        int num_visible_columns = vld.size.x * vld.size.y;

        PARALLEL_FOR
        for (int i = 0; i < num_visible_columns; i++)
            bind_inputs(Int2(i / vld.size.y, i % vld.size.y), vli);
    }
    
    PARALLEL_FOR
    for (int i = 0; i < num_hidden_columns; i++)
        forward(Int2(i / hidden_size.y, i % hidden_size.y));

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        vl.up_to_date = true;
    }

    if (learn_enabled) {
        PARALLEL_FOR
        for (int i = 0; i < num_hidden_columns; i++)
            learn(Int2(i / hidden_size.y, i % hidden_size.y), params);
    }
}

void Encoder::reconstruct(
    int vli
) {
    const Visible_Layer_Desc &vld = visible_layer_descs[vli];

    int num_visible_columns = vld.size.x * vld.size.y;

    PARALLEL_FOR
    for (int i = 0; i < num_visible_columns; i++)
        reconstruct(Int2(i / vld.size.y, i % vld.size.y), vli);
}

void Encoder::clear_state() {
    hidden_cis.fill(0);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        vl.recon_cis.fill(0);
    }
}

long Encoder::size() const {
    long size = sizeof(Int3) + sizeof(int) + hidden_cis.size() * sizeof(int) + hidden_code_vecs.size() * sizeof(float) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += sizeof(Visible_Layer_Desc) + vl.weights.size() * sizeof(Byte) + vl.recon_cis.size() * sizeof(int) + sizeof(float);
    }

    return size;
}

long Encoder::state_size() const {
    long size = hidden_cis.size() * sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += vl.recon_cis.size() * sizeof(int);
    }

    return size;
}

long Encoder::weights_size() const {
    long size = hidden_code_vecs.size() * sizeof(float);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += vl.weights.size() * sizeof(float);
    }

    return size;
}

void Encoder::write(
    Stream_Writer &writer
) const {
    writer.write(&hidden_size, sizeof(Int3));
    writer.write(&vec_size, sizeof(int));

    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    writer.write(&hidden_code_vecs[0], hidden_code_vecs.size() * sizeof(float));

    int num_visible_layers = visible_layers.size();

    writer.write(&num_visible_layers, sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(&vld, sizeof(Visible_Layer_Desc));

        writer.write(&vl.visible_code_vecs[0], vl.visible_code_vecs.size() * sizeof(S_Byte));
        writer.write(&vl.visible_pos_vecs[0], vl.visible_pos_vecs.size() * sizeof(S_Byte));

        writer.write(&vl.weights[0], vl.weights.size() * sizeof(float));

        writer.write(&vl.recon_cis[0], vl.recon_cis.size() * sizeof(int));

        writer.write(&vl.importance, sizeof(float));
    }
}

void Encoder::read(
    Stream_Reader &reader
) {
    reader.read(&hidden_size, sizeof(Int3));
    reader.read(&vec_size, sizeof(int));

    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;

    hidden_cis.resize(num_hidden_columns);
    hidden_code_vecs.resize(vec_size * num_hidden_cells);

    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    reader.read(&hidden_code_vecs[0], hidden_code_vecs.size() * sizeof(float));

    hidden_vecs.resize(vec_size * num_hidden_columns);

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

        vl.visible_code_vecs.resize(vec_size * num_visible_cells);
        vl.visible_pos_vecs.resize(vec_size * num_visible_columns);

        reader.read(&vl.visible_code_vecs[0], vl.visible_code_vecs.size() * sizeof(S_Byte));
        reader.read(&vl.visible_pos_vecs[0], vl.visible_pos_vecs.size() * sizeof(S_Byte));

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(num_hidden_columns * area);

        reader.read(&vl.weights[0], vl.weights.size() * sizeof(float));

        vl.visible_bundles.resize(vec_size * num_visible_columns);
        vl.hidden_bundles.resize(vec_size * num_hidden_columns);

        vl.input_cis = Int_Buffer(num_visible_columns, 0);

        vl.recon_cis.resize(num_visible_columns);

        reader.read(&vl.recon_cis[0], vl.recon_cis.size() * sizeof(int));

        reader.read(&vl.importance, sizeof(float));
    }
}

void Encoder::write_state(
    Stream_Writer &writer
) const {
    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        writer.write(&vl.recon_cis[0], vl.recon_cis.size() * sizeof(int));
    }
}

void Encoder::read_state(
    Stream_Reader &reader
) {
    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(&vl.recon_cis[0], vl.recon_cis.size() * sizeof(int));
    }
}

void Encoder::write_weights(
    Stream_Writer &writer
) const {
    writer.write(&hidden_code_vecs[0], hidden_code_vecs.size() * sizeof(float));

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        writer.write(&vl.weights[0], vl.weights.size() * sizeof(float));
    }
}

void Encoder::read_weights(
    Stream_Reader &reader
) {
    reader.read(&hidden_code_vecs[0], hidden_code_vecs.size() * sizeof(float));

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(&vl.weights[0], vl.weights.size() * sizeof(float));
    }
}
