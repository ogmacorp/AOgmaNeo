// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "encoder.h"

using namespace aon;

void Encoder::bind_inputs(
    const Int2 &column_pos,
    int vli,
    const Params &params
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

        vl.input_vecs[visible_vec_index] = ((vl.input_code_vecs[visible_code_index] > 0.0f) * 2 - 1) * vl.input_pos_vecs[visible_vec_index];
    }
}

void Encoder::forward(
    const Int2 &column_pos,
    const Params &params
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

                vl.hidden_bundles[hidden_vec_index] = 0;
            }

            for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                    int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                    int in_ci = vl.input_cis[visible_column_index];

                    int visible_vecs_start = visible_column_index * vec_size;

                    for (int vi = 0; vi < vec_size; vi++) {
                        int hidden_vec_index = vi + hidden_vecs_start;
                        int visible_vec_index = vi + visible_vecs_start;

                        vl.hidden_bundles[hidden_vec_index] += vl.input_vecs[visible_vec_index];
                    }
                }
        }
    }

    for (int vi = 0; vi < vec_size; vi++) {
        int hidden_vec_index = vi + hidden_vecs_start;

        int hidden_bundle = 0;

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];

            if (!vl.use_input)
                continue;

            hidden_bundle += vl.hidden_bundles[hidden_vec_index];
        }

        hidden_vecs[hidden_vec_index] = (hidden_bundle > 0) * 2 - 1;
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

    // learn
    for (int vi = 0; vi < vec_size; vi++) {
        int hidden_vec_index = vi + hidden_vecs_start;
        int hidden_code_index = vi + hidden_codes_start_max;

         hidden_code_vecs[hidden_code_index] += params.lr * (hidden_vecs[hidden_vec_index] - hidden_code_vecs[hidden_code_index]);
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

        vl.visible_bundles[visible_vec_index] = 0;
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

                for (int vi = 0; vi < vec_size; vi++) {
                    int visible_vec_index = vi + visible_vecs_start;
                    int hidden_code_index = vi + hidden_codes_start_max;

                    // un-bind and superimpose
                    vl.visible_bundles[visible_vec_index] += ((hidden_code_vecs[hidden_code_index] > 0.0f) * 2 - 1) * vl.input_pos_vecs[visible_vec_index];
                }
            }
        }

    for (int vi = 0; vi < vec_size; vi++) {
        int visible_vec_index = vi + visible_vecs_start;

        vl.recon_vecs[visible_vec_index] = (vl.visible_bundles[visible_vec_index] > 0) * 2 - 1;
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

            similarity += vl.recon_vecs[visible_vec_index] * ((hidden_code_vecs[visible_code_index] > 0.0f) * 2 - 1);
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
    const Array<Visible_Layer_Desc> &visible_layer_descs
) {
    this->visible_layer_descs = visible_layer_descs;

    this->hidden_size = hidden_size;

    visible_layers.resize(visible_layer_descs.size());

    // pre-compute dimensions
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;

    int total_num_visible_columns = 0;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;
        int num_visible_cells = num_visible_columns * vld.size.z;

        total_num_visible_columns += num_visible_columns;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(num_hidden_cells * area * vld.size.z);

        for (int i = 0; i < vl.weights.size(); i++)
            vl.weights[i] = 127 + (rand() % 127);

        vl.hidden_sums.resize(num_hidden_cells);

        vl.input_cis = Int_Buffer(num_visible_columns, 0);
        vl.recon_cis = Int_Buffer(num_visible_columns, 0);

        vl.recon_deltas.resize(num_visible_cells);
        vl.recon_sums.resize(num_visible_cells);
    }

    hidden_cis = Int_Buffer(num_hidden_columns, 0);

    // generate helper buffers for parallelization
    visible_pos_vlis.resize(total_num_visible_columns);

    int index = 0;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;

        for (int i = 0; i < num_visible_columns; i++) {
            visible_pos_vlis[index] = Int3(i / vld.size.y, i % vld.size.y, vli);
            index++;
        }
    }
}

void Encoder::step(
    bool learn_enabled,
    const Params &params
) {
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    
    PARALLEL_FOR
    for (int i = 0; i < num_hidden_columns; i++)
        forward(Int2(i / hidden_size.y, i % hidden_size.y), params);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        vl.up_to_date = true;
    }

    if (learn_enabled) {
        unsigned int base_state = rand();

        PARALLEL_FOR
        for (int i = 0; i < visible_pos_vlis.size(); i++) {
            Int2 pos = Int2(visible_pos_vlis[i].x, visible_pos_vlis[i].y);
            int vli = visible_pos_vlis[i].z;

            unsigned long state = rand_get_state(base_state + i * rand_subseed_offset);

            learn(pos, vli, &state, params);
        }
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
    long size = sizeof(Int3) + hidden_cis.size() * sizeof(int) + sizeof(int);

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
    long size = 0;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += vl.weights.size() * sizeof(Byte);
    }

    return size;
}

void Encoder::write(
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

    int num_visible_layers = visible_layers.size();

    reader.read(reinterpret_cast<void*>(&num_visible_layers), sizeof(int));

    visible_layers.resize(num_visible_layers);
    visible_layer_descs.resize(num_visible_layers);
    
    int total_num_visible_columns = 0;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        Visible_Layer_Desc &vld = visible_layer_descs[vli];

        reader.read(reinterpret_cast<void*>(&vld), sizeof(Visible_Layer_Desc));

        int num_visible_columns = vld.size.x * vld.size.y;
        int num_visible_cells = num_visible_columns * vld.size.z;

        total_num_visible_columns += num_visible_columns;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(num_hidden_cells * area * vld.size.z);

        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(Byte));

        vl.hidden_sums.resize(num_hidden_cells);

        vl.input_cis = Int_Buffer(num_visible_columns, 0);

        vl.recon_cis.resize(num_visible_columns);

        reader.read(reinterpret_cast<void*>(&vl.recon_cis[0]), vl.recon_cis.size() * sizeof(int));

        vl.recon_deltas.resize(num_visible_cells);
        vl.recon_sums.resize(num_visible_cells);

        reader.read(reinterpret_cast<void*>(&vl.importance), sizeof(float));
    }

    // generate helper buffers for parallelization
    visible_pos_vlis.resize(total_num_visible_columns);

    int index = 0;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;

        for (int i = 0; i < num_visible_columns; i++) {
            visible_pos_vlis[index] = Int3(i / vld.size.y, i % vld.size.y, vli);
            index++;
        }
    }
}

void Encoder::write_state(
    Stream_Writer &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        writer.write(reinterpret_cast<const void*>(&vl.recon_cis[0]), vl.recon_cis.size() * sizeof(int));
    }
}

void Encoder::read_state(
    Stream_Reader &reader
) {
    reader.read(reinterpret_cast<void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(reinterpret_cast<void*>(&vl.recon_cis[0]), vl.recon_cis.size() * sizeof(int));
    }
}

void Encoder::write_weights(
    Stream_Writer &writer
) const {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        writer.write(reinterpret_cast<const void*>(&vl.weights[0]), vl.weights.size() * sizeof(Byte));
    }
}

void Encoder::read_weights(
    Stream_Reader &reader
) {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(reinterpret_cast<void*>(&vl.weights[0]), vl.weights.size() * sizeof(Byte));
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
        
            for (int i = 0; i < vl.weights.size(); i++) {
                int e = rand() % encoders.size();                

                vl.weights[i] = encoders[e]->visible_layers[vli].weights[i];
            }
        }

        break;
    case merge_average:
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];
        
            for (int i = 0; i < vl.weights.size(); i++) {
                float total = 0.0f;

                for (int e = 0; e < encoders.size(); e++)
                    total += encoders[e]->visible_layers[vli].weights[i];

                vl.weights[i] = roundf(total / encoders.size());
            }
        }

        break;
    }
}
