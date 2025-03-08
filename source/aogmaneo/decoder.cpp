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

    float count = 0.0f;
    float count_all = 0.0f;
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

        int sub_count = vl.dendrite_counts[hidden_column_index];

        count += vl.importance * sub_count;
        count_all += vl.importance * sub_count * vld.size.z;

        total_importance += vl.importance;

        Int_Buffer_View vl_input_cis = input_cis[vli];

        for (int hc = 0; hc < hidden_size.z; hc++) {
            int hidden_cell_index = hc + hidden_cells_start;

            int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

            for (int di = 0; di < num_dendrites_per_cell; di++) {
                int dendrite_index = di + dendrites_start;

                vl.dendrite_sums0[dendrite_index] = 0;
                vl.dendrite_sums1[dendrite_index] = 0;
            }
        }

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci = vl_input_cis[visible_column_index];

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi_start_partial = hidden_size.z * (offset.y + diam * (offset.x + diam * (in_ci + vld.size.z * hidden_column_index)));

                for (int hc = 0; hc < hidden_size.z; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

                    int wi_start = num_dendrites_per_cell * (hc + wi_start_partial);

                    for (int di = 0; di < num_dendrites_per_cell; di++) {
                        int dendrite_index = di + dendrites_start;

                        int wi = di + wi_start;

                        vl.dendrite_sums0[dendrite_index] += vl.weights0[wi];
                        vl.dendrite_sums1[dendrite_index] += vl.weights1[wi];
                    }
                }
            }
    }

    count /= max(limit_small, total_importance);
    count_all /= max(limit_small, total_importance);

    int max_compare_index = 0;
    float max_compare_activation = limit_min;

    const float byte_inv = 1.0f / 255.0f;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

        int max_index0 = -1;
        int max_index1 = -1;
        float max_activation0 = 0.0f;
        float max_activation1 = 0.0f;

        int max_complete_index0 = 0;
        int max_complete_index1 = 0;
        float max_complete_activation0 = 0.0f;
        float max_complete_activation1 = 0.0f;

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + dendrites_start;

            float sum0 = 0.0f;
            float sum1 = 0.0f;
            float total0 = 0.0f;
            float total1 = 0.0f;

            bool all_match0 = true;
            bool all_match1 = true;

            for (int vli = 0; vli < visible_layers.size(); vli++) {
                Visible_Layer &vl = visible_layers[vli];
                const Visible_Layer_Desc &vld = visible_layer_descs[vli];

                float influence = vl.importance * byte_inv;

                int sub_count = vl.dendrite_counts[hidden_column_index];
                int sub_count_except = sub_count * (vld.size.z - 1);
                int sub_count_all = sub_count * vld.size.z;

                float complemented0 = (sub_count_all - vl.dendrite_totals0[hidden_cell_index] * byte_inv) - (sub_count - vl.dendrite_sums0[hidden_cell_index] * byte_inv);
                float complemented1 = (sub_count_all - vl.dendrite_totals1[hidden_cell_index] * byte_inv) - (sub_count - vl.dendrite_sums1[hidden_cell_index] * byte_inv);

                float match0 = complemented0 / sub_count_except;
                float match1 = complemented1 / sub_count_except;

                float vigilance = 1.0f - params.mismatch / vld.size.z;

                if (vl.importance > 0.0f) {
                    if (match0 < vigilance)
                        all_match0 = false;

                    if (match1 < vigilance)
                        all_match1 = false;
                }

                sum0 += vl.dendrite_sums0[dendrite_index] * influence;
                total0 += vl.dendrite_totals0[dendrite_index] * influence;

                sum1 += vl.dendrite_sums1[dendrite_index] * influence;
                total1 += vl.dendrite_totals1[dendrite_index] * influence;
            }

            sum0 /= max(limit_small, total_importance);
            total0 /= max(limit_small, total_importance);

            sum1 /= max(limit_small, total_importance);
            total1 /= max(limit_small, total_importance);

            float complemented0 = (count_all - total0) - (count - sum0);
            float complemented1 = (count_all - total1) - (count - sum1);

            float activation0 = complemented0 / (params.choice + count_all - total0);
            float activation1 = complemented1 / (params.choice + count_all - total1);

            if (all_match0 && activation0 > max_activation0) {
                max_activation0 = activation0;
                max_index0 = di;
            }

            if (all_match1 && activation1 > max_activation1) {
                max_activation1 = activation1;
                max_index1 = di;
            }

            if (activation0 > max_complete_activation0) {
                max_complete_activation0 = activation0;
                max_complete_index0 = di;
            }

            if (activation1 > max_complete_activation1) {
                max_complete_activation1 = activation1;
                max_complete_index1 = di;
            }
        }

        hidden_dis0[hidden_cell_index] = (max_index0 == -1 ? max_complete_index0 : max_index0);
        hidden_dis1[hidden_cell_index] = (max_index1 == -1 ? max_complete_index1 : max_index1);

        float compare_activation = max_complete_activation0 - max_complete_activation1;

        if (compare_activation > max_compare_activation) {
            max_compare_activation = compare_activation;
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
    int hidden_cell_index_max = hidden_ci + hidden_cells_start;

    int hidden_di_target0 = hidden_dis0[hidden_cell_index_target];
    int hidden_di_max1 = hidden_dis1[hidden_cell_index_max];

    if (hidden_di_target0 == -1 || hidden_di_max1 == -1)
        return;

    int dendrites_start_target = num_dendrites_per_cell * hidden_cell_index_target;
    int dendrites_start_max = num_dendrites_per_cell * hidden_cell_index_max;

    int dendrite_index_target0 = hidden_di_target0 + dendrites_start_target;
    int dendrite_index_max1 = hidden_di_max1 + dendrites_start_max;

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

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci = vl_input_cis[visible_column_index];

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                {
                    int wi = hidden_di_target0 + num_dendrites_per_cell * (target_ci + hidden_size.z * (offset.y + diam * (offset.x + diam * (in_ci + vld.size.z * hidden_column_index))));

                    Byte w_old = vl.weights0[wi];

                    vl.weights0[wi] = min(255, vl.weights0[wi] + ceilf(params.lr * (255.0f - vl.weights0[wi])));

                    vl.dendrite_totals0[dendrite_index_target0] += vl.weights0[wi] - w_old;
                }

                {
                    int wi = hidden_di_max1 + num_dendrites_per_cell * (hidden_ci + hidden_size.z * (offset.y + diam * (offset.x + diam * (in_ci + vld.size.z * hidden_column_index))));

                    Byte w_old = vl.weights1[wi];

                    vl.weights1[wi] = min(255, vl.weights1[wi] + ceilf(params.lr * (255.0f - vl.weights1[wi])));

                    vl.dendrite_totals1[dendrite_index_max1] += vl.weights1[wi] - w_old;
                }
            }
    }
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

        vl.weights0.resize(num_dendrites * area * vld.size.z);
        vl.weights1.resize(vl.weights0.size());

        for (int i = 0; i < vl.weights0.size(); i++) {
            vl.weights0[i] = (rand() % init_weight_noisei);
            vl.weights1[i] = (rand() % init_weight_noisei);
        }

        vl.dendrite_sums0.resize(num_dendrites);
        vl.dendrite_sums1.resize(num_dendrites);
        vl.dendrite_totals0.resize(num_dendrites);
        vl.dendrite_totals1.resize(num_dendrites);
        vl.dendrite_counts.resize(num_hidden_columns);
    }

    hidden_cis = Int_Buffer(num_hidden_columns, 0);

    hidden_dis0 = Int_Buffer(num_hidden_cells, -1);
    hidden_dis1 = Int_Buffer(num_hidden_cells, -1);

    // init totals and counts
    for (int i = 0; i < num_hidden_columns; i++) {
        Int2 column_pos(i / hidden_size.y, i % hidden_size.y);

        int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

        int hidden_cells_start = hidden_column_index * hidden_size.z;

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

            for (int hc = 0; hc < hidden_size.z; hc++) {
                int hidden_cell_index = hc + hidden_cells_start;

                int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

                for (int di = 0; di < num_dendrites_per_cell; di++) {
                    int dendrite_index = di + dendrites_start;

                    int total0 = 0;
                    int total1 = 0;

                    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                            Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                            for (int vc = 0; vc < vld.size.z; vc++) {
                                int wi = di + num_dendrites_per_cell * (hc + hidden_size.z * (offset.y + diam * (offset.x + diam * (vc + vld.size.z * hidden_column_index))));

                                total0 += vl.weights0[wi];
                                total1 += vl.weights1[wi];
                            }
                        }

                    vl.dendrite_totals0[dendrite_index] = total0;
                    vl.dendrite_totals1[dendrite_index] = total1;
                }
            }
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
    hidden_dis0.fill(-1);
    hidden_dis1.fill(-1);
}

long Decoder::size() const {
    long size = sizeof(Int3) + sizeof(int) + hidden_cis.size() * sizeof(int) + 2 * hidden_dis0.size() * sizeof(int) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        size += sizeof(Visible_Layer_Desc) + 2 * vl.weights0.size() * sizeof(Byte) + 2 * vl.dendrite_totals0.size() * sizeof(int) + vl.dendrite_counts.size() * sizeof(int) + sizeof(float);
    }

    return size;
}

long Decoder::state_size() const {
    return hidden_cis.size() * sizeof(int) + 2 * hidden_dis0.size() * sizeof(int);
}

long Decoder::weights_size() const {
    int size = 0;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += 2 * vl.weights0.size() * sizeof(Byte);
    }

    return size;
}

void Decoder::write(
    Stream_Writer &writer
) const {
    writer.write(&hidden_size, sizeof(Int3));
    writer.write(&num_dendrites_per_cell, sizeof(int));

    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    writer.write(&hidden_dis0[0], hidden_dis0.size() * sizeof(int));
    writer.write(&hidden_dis1[0], hidden_dis1.size() * sizeof(int));
    
    int num_visible_layers = visible_layers.size();

    writer.write(&num_visible_layers, sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(&vld, sizeof(Visible_Layer_Desc));

        writer.write(&vl.weights0[0], vl.weights0.size() * sizeof(Byte));
        writer.write(&vl.weights1[0], vl.weights1.size() * sizeof(Byte));

        writer.write(&vl.dendrite_totals0[0], vl.dendrite_totals0.size() * sizeof(int));
        writer.write(&vl.dendrite_totals1[0], vl.dendrite_totals1.size() * sizeof(int));
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
    hidden_dis0.resize(num_hidden_cells);
    hidden_dis1.resize(num_hidden_cells);

    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    reader.read(&hidden_dis0[0], hidden_dis0.size() * sizeof(int));
    reader.read(&hidden_dis1[0], hidden_dis1.size() * sizeof(int));

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

        vl.weights0.resize(num_dendrites * area * vld.size.z);
        vl.weights1.resize(vl.weights0.size());

        reader.read(&vl.weights0[0], vl.weights0.size() * sizeof(Byte));
        reader.read(&vl.weights1[0], vl.weights1.size() * sizeof(Byte));

        vl.dendrite_sums0.resize(num_dendrites);
        vl.dendrite_sums1.resize(num_dendrites);

        vl.dendrite_totals0.resize(num_dendrites);
        vl.dendrite_totals1.resize(num_dendrites);

        reader.read(&vl.dendrite_totals0[0], vl.dendrite_totals0.size() * sizeof(int));
        reader.read(&vl.dendrite_totals1[0], vl.dendrite_totals1.size() * sizeof(int));

        vl.dendrite_counts.resize(num_hidden_columns);

        reader.read(&vl.dendrite_counts[0], vl.dendrite_counts.size() * sizeof(int));

        reader.read(&vl.importance, sizeof(float));
    }
}

void Decoder::write_state(
    Stream_Writer &writer
) const {
    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    writer.write(&hidden_dis0[0], hidden_dis0.size() * sizeof(int));
    writer.write(&hidden_dis1[0], hidden_dis1.size() * sizeof(int));
}

void Decoder::read_state(
    Stream_Reader &reader
) {
    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    reader.read(&hidden_dis0[0], hidden_dis0.size() * sizeof(int));
    reader.read(&hidden_dis1[0], hidden_dis1.size() * sizeof(int));
}

void Decoder::write_weights(
    Stream_Writer &writer
) const {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        writer.write(&vl.weights0[0], vl.weights0.size() * sizeof(Byte));
        writer.write(&vl.weights1[0], vl.weights1.size() * sizeof(Byte));
    }
}

void Decoder::read_weights(
    Stream_Reader &reader
) {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(&vl.weights0[0], vl.weights0.size() * sizeof(Byte));
        reader.read(&vl.weights1[0], vl.weights1.size() * sizeof(Byte));
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

                vl.weights0[i] = roundf(total0 / decoders.size());
                vl.weights1[i] = roundf(total1 / decoders.size());
            }
        }

        break;
    }

    int num_hidden_columns = hidden_size.x * hidden_size.y;

    // re-init totals
    for (int i = 0; i < num_hidden_columns; i++) {
        Int2 column_pos(i / hidden_size.y, i % hidden_size.y);

        int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

        int hidden_cells_start = hidden_column_index * hidden_size.z;

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

            for (int hc = 0; hc < hidden_size.z; hc++) {
                int hidden_cell_index = hc + hidden_cells_start;

                int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

                for (int di = 0; di < num_dendrites_per_cell; di++) {
                    int dendrite_index = di + dendrites_start;

                    int total0 = 0;
                    int total1 = 0;

                    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                            Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                            for (int vc = 0; vc < vld.size.z; vc++) {
                                int wi = di + num_dendrites_per_cell * (hc + hidden_size.z * (offset.y + diam * (offset.x + diam * (vc + vld.size.z * hidden_column_index))));

                                total0 += vl.weights0[wi];
                                total1 += vl.weights1[wi];
                            }
                        }

                    vl.dendrite_totals0[dendrite_index] = total0;
                    vl.dendrite_totals1[dendrite_index] = total1;
                }
            }
        }
    }
}
