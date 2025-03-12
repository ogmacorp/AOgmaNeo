// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2025 Ogma Intelligent Systems Corp. All rights reserved.
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

        total_importance += vl.importance;

        Int_Buffer_View vl_input_cis = input_cis[vli];

        for (int hc = 0; hc < hidden_size.z; hc++) {
            int hidden_cell_index = hc + hidden_cells_start;

            int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

            for (int di = 0; di < num_dendrites_per_cell; di++) {
                int dendrite_index = di + dendrites_start;

                vl.dendrite_sums[dendrite_index] = 0.0f;
                vl.dendrite_totals[dendrite_index] = 0.0f;
            }
        }

        const float vld_size_z_inv = 1.0f / vld.size.z;

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci = vl_input_cis[visible_column_index];

                float in_value = (in_ci + 0.5f) * vld_size_z_inv;

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi_start_partial = hidden_size.z * (offset.y + diam * (offset.x + diam * hidden_column_index));

                for (int hc = 0; hc < hidden_size.z; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

                    int wi_start = num_dendrites_per_cell * (hc + wi_start_partial);

                    for (int di = 0; di < num_dendrites_per_cell; di++) {
                        int dendrite_index = di + dendrites_start;

                        int wi = di + wi_start;

                        vl.dendrite_sums[dendrite_index] += min(vl.weights0[wi], in_value) + min(vl.weights1[wi], 1.0f - in_value);
                        vl.dendrite_totals[dendrite_index] += vl.weights0[wi] + vl.weights1[wi];
                    }
                }
            }
    }

    int max_compare_index = 0;
    float max_compare_activation = 0.0f;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

        int max_index = -1;
        float max_activation = 0.0f;

        int max_complete_index = 0;
        float max_complete_activation = 0.0f;

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + dendrites_start;

            float sum = 0.0f;
            float total = 0.0f;

            bool all_match = true;

            for (int vli = 0; vli < visible_layers.size(); vli++) {
                Visible_Layer &vl = visible_layers[vli];
                const Visible_Layer_Desc &vld = visible_layer_descs[vli];

                int sub_count = vl.dendrite_counts[hidden_column_index];

                float sub_sum = vl.dendrite_sums[dendrite_index];
                float sub_total = vl.dendrite_totals[dendrite_index];

                float match = sub_sum / sub_count;

                float vigilance = 1.0f - params.mismatch / vld.size.z;

                if (vl.importance > 0.0f && match < vigilance)
                    all_match = false;

                sum += sub_sum * vl.importance;
                total += sub_total * vl.importance;
            }

            sum /= max(limit_small, total_importance);
            total /= max(limit_small, total_importance);

            float activation = sum / (params.choice + total);

            if (all_match && activation > max_activation) {
                max_activation = activation;
                max_index = di;
            }

            if (activation > max_complete_activation) {
                max_complete_activation = activation;
                max_complete_index = di;
            }
        }

        hidden_dis[hidden_cell_index] = (max_index == -1 ? max_complete_index : max_index);

        if (max_complete_activation > max_compare_activation) {
            max_compare_activation = max_complete_activation;
            max_compare_index = hc;
        }
    }

    hidden_cis[hidden_column_index] = max_compare_index;
}

void Decoder::learn(
    const Int2 &column_pos,
    const Array<Int_Buffer_View> &input_cis,
    const Int_Buffer_View hidden_target_cis,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    int target_ci = hidden_target_cis[hidden_column_index];
    int hidden_ci = hidden_cis[hidden_column_index];

    if (hidden_ci == target_ci)
        return;

    int hidden_cell_index_target = target_ci + hidden_cells_start;

    int hidden_di_target = hidden_dis[hidden_cell_index_target];

    if (hidden_di_target == -1)
        return;

    int dendrite_index_target = hidden_di_target + num_dendrites_per_cell * hidden_cell_index_target;

    float rate = (dendrite_commits[dendrite_index_target] ? params.lr : 1.0f);

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

        Int_Buffer_View vl_input_cis = input_cis[vli];

        const float vld_size_z_inv = 1.0f / vld.size.z;

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci = vl_input_cis[visible_column_index];

                float in_value = (in_ci + 0.5f) * vld_size_z_inv;

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi = hidden_di_target + num_dendrites_per_cell * (target_ci + hidden_size.z * (offset.y + diam * (offset.x + diam * hidden_column_index)));

                vl.weights0[wi] += rate * min(0.0f, in_value - vl.weights0[wi]);
                vl.weights1[wi] += rate * min(0.0f, 1.0f - in_value - vl.weights1[wi]);
            }
    }

    dendrite_commits[dendrite_index_target] = true;
}

void Decoder::init_random(
    const Int3 &hidden_size,
    int num_dendrites_per_cell,
    const Array<Visible_Layer_Desc> &visible_layer_descs
) {
    this->visible_layer_descs = visible_layer_descs; 

    this->hidden_size = hidden_size;
    this->num_dendrites_per_cell = num_dendrites_per_cell;

    visible_layers.resize(visible_layer_descs.size());

    // pre-compute dimensions
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;
    int num_dendrites = num_hidden_cells * num_dendrites_per_cell;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;
        int num_visible_cells = num_visible_columns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights0.resize(num_dendrites * area);
        vl.weights1.resize(vl.weights0.size());

        for (int i = 0; i < vl.weights0.size(); i++) {
            vl.weights0[i] = 1.0f - randf() * init_weight_noisef;
            vl.weights1[i] = 1.0f - randf() * init_weight_noisef;
        }

        vl.dendrite_sums.resize(num_dendrites);
        vl.dendrite_totals.resize(num_dendrites);
        vl.dendrite_counts.resize(num_hidden_columns);
    }

    hidden_cis = Int_Buffer(num_hidden_columns, 0);

    hidden_dis = Int_Buffer(num_hidden_cells, -1);

    dendrite_commits = Byte_Buffer(num_dendrites, false);

    // init totals
    for (int i = 0; i < num_hidden_columns; i++) {
        Int2 column_pos(i / hidden_size.y, i % hidden_size.y);

        int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

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

            vl.dendrite_counts[hidden_column_index] = (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1);
        }
    }
}

void Decoder::activate(
    const Array<Int_Buffer_View> &input_cis,
    const Params &params
) {
    int num_hidden_columns = hidden_size.x * hidden_size.y;

    PARALLEL_FOR
    for (int i = 0; i < num_hidden_columns; i++)
        forward(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, params);
}

void Decoder::learn(
    const Array<Int_Buffer_View> &input_cis,
    Int_Buffer_View hidden_target_cis,
    const Params &params
) {
    int num_hidden_columns = hidden_size.x * hidden_size.y;

    PARALLEL_FOR
    for (int i = 0; i < num_hidden_columns; i++)
        learn(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, hidden_target_cis, params);
}

void Decoder::clear_state() {
    hidden_cis.fill(0);
    hidden_dis.fill(-1);
}

long Decoder::size() const {
    long size = sizeof(Int3) + sizeof(int) + hidden_cis.size() * sizeof(int) + hidden_dis.size() * sizeof(int) + dendrite_commits.size() * sizeof(Byte) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        size += sizeof(Visible_Layer_Desc) + 2 * vl.weights0.size() * sizeof(float) + vl.dendrite_counts.size() * sizeof(int) + sizeof(float);
    }

    return size;
}

long Decoder::state_size() const {
    return hidden_cis.size() * sizeof(int) + hidden_dis.size() * sizeof(int);
}

long Decoder::weights_size() const {
    int size = 0;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += vl.weights0.size() * sizeof(float);
    }

    return size;
}

void Decoder::write(
    Stream_Writer &writer
) const {
    writer.write(&hidden_size, sizeof(Int3));
    writer.write(&num_dendrites_per_cell, sizeof(int));

    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    writer.write(&hidden_dis[0], hidden_dis.size() * sizeof(int));
    writer.write(&dendrite_commits[0], dendrite_commits.size() * sizeof(Byte));
    
    int num_visible_layers = visible_layers.size();

    writer.write(&num_visible_layers, sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(&vld, sizeof(Visible_Layer_Desc));

        writer.write(&vl.weights0[0], vl.weights0.size() * sizeof(float));
        writer.write(&vl.weights1[0], vl.weights1.size() * sizeof(float));

        writer.write(&vl.dendrite_counts[0], vl.dendrite_counts.size() * sizeof(int));

        writer.write(&vl.importance, sizeof(float));
    }
}

void Decoder::read(
    Stream_Reader &reader
) {
    reader.read(&hidden_size, sizeof(Int3));
    reader.read(&num_dendrites_per_cell, sizeof(int));

    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;
    int num_dendrites = num_hidden_cells * num_dendrites_per_cell;

    hidden_cis.resize(num_hidden_columns);
    hidden_dis.resize(num_hidden_cells);
    dendrite_commits.resize(num_dendrites);

    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    reader.read(&hidden_dis[0], hidden_dis.size() * sizeof(int));
    reader.read(&dendrite_commits[0], dendrite_commits.size() * sizeof(Byte));

    int num_visible_layers;

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

        vl.weights0.resize(num_dendrites * area);

        reader.read(&vl.weights0[0], vl.weights0.size() * sizeof(float));
        reader.read(&vl.weights1[0], vl.weights1.size() * sizeof(float));

        vl.dendrite_sums.resize(num_dendrites);
        vl.dendrite_totals.resize(num_dendrites);

        vl.dendrite_counts.resize(num_hidden_columns);

        reader.read(&vl.dendrite_counts[0], vl.dendrite_counts.size() * sizeof(int));

        reader.read(&vl.importance, sizeof(float));
    }
}

void Decoder::write_state(
    Stream_Writer &writer
) const {
    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    writer.write(&hidden_dis[0], hidden_dis.size() * sizeof(int));
}

void Decoder::read_state(
    Stream_Reader &reader
) {
    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    reader.read(&hidden_dis[0], hidden_dis.size() * sizeof(int));
}

void Decoder::write_weights(
    Stream_Writer &writer
) const {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        writer.write(&vl.weights0[0], vl.weights0.size() * sizeof(float));
        writer.write(&vl.weights1[0], vl.weights1.size() * sizeof(float));
    }
}

void Decoder::read_weights(
    Stream_Reader &reader
) {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(&vl.weights0[0], vl.weights0.size() * sizeof(float));
        reader.read(&vl.weights1[0], vl.weights1.size() * sizeof(float));
    }
}

void Decoder::merge(
    const Array<Decoder*> &decoders,
    Merge_Mode mode
) {
    switch (mode) {
    case merge_random:
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];
        
            for (int i = 0; i < vl.weights0.size(); i++) {
                int d = rand() % decoders.size();                

                vl.weights0[i] = decoders[d]->visible_layers[vli].weights0[i];
                vl.weights1[i] = decoders[d]->visible_layers[vli].weights1[i];
            }
        }

        break;
    case merge_average:
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];
        
            for (int i = 0; i < vl.weights0.size(); i++) {
                float total0 = 0.0f;
                float total1 = 0.0f;

                for (int d = 0; d < decoders.size(); d++) {
                    total0 += decoders[d]->visible_layers[vli].weights0[i];
                    total1 += decoders[d]->visible_layers[vli].weights1[i];
                }

                vl.weights0[i] = total0 / decoders.size();
                vl.weights1[i] = total1 / decoders.size();
            }
        }

        break;
    }
}
