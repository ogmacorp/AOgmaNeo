// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "encoder.h"

using namespace aon;

void Encoder::forward(
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

        int sub_count = (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1);

        count += vl.importance * sub_count;
        count_all += vl.importance * sub_count * vld.size.z;

        total_importance += vl.importance;

        Int_Buffer_View vl_input_cis = input_cis[vli];

        for (int hc = 0; hc < hidden_size.z; hc++) {
            int hidden_cell_index = hc + hidden_cells_start;

            vl.hidden_sums0[hidden_cell_index] = 0;
            vl.hidden_sums1[hidden_cell_index] = 0;
        }

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci = vl_input_cis[visible_column_index];

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi_start = hidden_size.z * (offset.y + diam * (offset.x + diam * (in_ci + vld.size.z * hidden_column_index)));

                for (int hc = 0; hc < hidden_size.z; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int wi = hc + wi_start;

                    vl.hidden_sums0[hidden_cell_index] += vl.weights0[wi];
                    vl.hidden_sums1[hidden_cell_index] += vl.weights1[wi];
                }
            }
    }

    count /= max(limit_small, total_importance);
    count_all /= max(limit_small, total_importance);

    int max_index = -1;
    float max_activation = 0.0f;

    int max_complete_index = 0;
    float max_complete_activation = 0.0f;

    const float byte_inv = 1.0f / 255.0f;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float total = 0.0f;
        float total1 = 0.0f;

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];

            float influence = vl.importance * byte_inv;

            sum0 += vl.hidden_sums0[hidden_cell_index] * influence;
            sum1 += vl.hidden_sums1[hidden_cell_index] * influence;
            total += vl.hidden_totals[hidden_cell_index] * influence;
            total1 += vl.hidden_totals1[hidden_cell_index] * influence;
        }

        sum0 /= max(limit_small, total_importance);
        sum1 /= max(limit_small, total_importance);
        total /= max(limit_small, total_importance);
        total1 /= max(limit_small, total_importance);

        // complement
        sum1 = total1 - sum1;

        float match = (sum0 + sum1) / count_all;

        float activation = (sum0 + sum1) / (params.choice + total);

        if (match >= params.vigilance && activation > max_activation) {
            max_activation = activation;
            max_index = hc;
        }

        if (activation > max_complete_activation) {
            max_complete_activation = activation;
            max_complete_index = hc;
        }
    }

    hidden_comparisons[hidden_column_index] = (max_index == -1 ? 0.0f : max_complete_activation);

    hidden_cis[hidden_column_index] = max_complete_index;

    learn_cis[hidden_column_index] = max_index;
}

void Encoder::learn(
    const Int2 &column_pos,
    const Array<Int_Buffer_View> &input_cis,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    int hidden_ci = learn_cis[hidden_column_index];

    if (hidden_ci == -1)
        return;

    float hidden_max = hidden_comparisons[hidden_column_index];

    int num_higher = 0;
    int count = 1; // start at 1 since self is skipped

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

    if (ratio > params.active_ratio)
        return;

    int hidden_cell_index_max = hidden_ci + hidden_cells_start;

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

                for (int vc = 0; vc < vld.size.z; vc++) {
                    if (vc == in_ci)
                        continue;

                    int wi = hidden_ci + hidden_size.z * (offset.y + diam * (offset.x + diam * (vc + vld.size.z * hidden_column_index)));

                    Byte w_old0 = vl.weights0[wi];

                    vl.weights0[wi] = max(0, vl.weights0[wi] - ceilf(params.lr * vl.weights0[wi]));

                    vl.hidden_totals[hidden_cell_index_max] += vl.weights0[wi] - w_old0;
                }

                int wi = hidden_ci + hidden_size.z * (offset.y + diam * (offset.x + diam * (in_ci + vld.size.z * hidden_column_index)));

                Byte w_old1 = vl.weights1[wi];

                vl.weights1[wi] = max(0, vl.weights1[wi] - ceilf(params.lr * vl.weights1[wi]));

                vl.hidden_totals[hidden_cell_index_max] += vl.weights1[wi] - w_old1;
                vl.hidden_totals1[hidden_cell_index_max] += vl.weights1[wi] - w_old1;
            }
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

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights0.resize(num_hidden_cells * area * vld.size.z);
        vl.weights1.resize(vl.weights0.size());

        for (int i = 0; i < vl.weights0.size(); i++) {
            vl.weights0[i] = 255 - (rand() % init_weight_noisei);
            vl.weights1[i] = 255 - (rand() % init_weight_noisei);
        }

        vl.hidden_sums0.resize(num_hidden_cells);
        vl.hidden_sums1.resize(num_hidden_cells);
        vl.hidden_totals.resize(num_hidden_cells);
        vl.hidden_totals1.resize(num_hidden_cells);
    }

    hidden_cis = Int_Buffer(num_hidden_columns, 0);

    learn_cis.resize(num_hidden_columns);

    hidden_comparisons.resize(num_hidden_columns);

    // init totals
    for (int i = 0; i < num_hidden_columns; i++) {
        Int2 column_pos(i / hidden_size.y, i % hidden_size.y);

        int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

        int hidden_cells_start = hidden_column_index * hidden_size.z;

        for (int hc = 0; hc < hidden_size.z; hc++) {
            int hidden_cell_index = hc + hidden_cells_start;

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
                int sub_total1 = 0;

                for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                    for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                        int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                        Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                        for (int vc = 0; vc < vld.size.z; vc++) {
                            int wi = hc + hidden_size.z * (offset.y + diam * (offset.x + diam * (vc + vld.size.z * hidden_column_index)));

                            sub_total += vl.weights0[wi] + vl.weights1[wi];
                            sub_total1 += vl.weights1[wi];
                        }
                    }

                vl.hidden_totals[hidden_cell_index] = sub_total;
                vl.hidden_totals1[hidden_cell_index] = sub_total1;
            }
        }
    }
}

void Encoder::step(
    const Array<Int_Buffer_View> &input_cis,
    bool learn_enabled,
    const Params &params
) {
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    
    PARALLEL_FOR
    for (int i = 0; i < num_hidden_columns; i++)
        forward(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, params);

    if (learn_enabled) {
        PARALLEL_FOR
        for (int i = 0; i < num_hidden_columns; i++)
            learn(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, params);
    }
}

void Encoder::clear_state() {
    hidden_cis.fill(0);
}

long Encoder::size() const {
    long size = sizeof(Int3) + hidden_cis.size() * sizeof(int) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += sizeof(Visible_Layer_Desc) + 2 * vl.weights0.size() * sizeof(Byte) + 2 * vl.hidden_totals.size() * sizeof(int) + sizeof(float);
    }

    return size;
}

long Encoder::state_size() const {
    return hidden_cis.size() * sizeof(int);
}

long Encoder::weights_size() const {
    long size = 0;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += 2 * vl.weights0.size() * sizeof(Byte);
    }

    return size;
}

void Encoder::write(
    Stream_Writer &writer
) const {
    writer.write(&hidden_size, sizeof(Int3));

    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));

    int num_visible_layers = visible_layers.size();

    writer.write(&num_visible_layers, sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(&vld, sizeof(Visible_Layer_Desc));

        writer.write(&vl.weights0[0], vl.weights0.size() * sizeof(Byte));
        writer.write(&vl.weights1[0], vl.weights1.size() * sizeof(Byte));

        writer.write(&vl.hidden_totals[0], vl.hidden_totals.size() * sizeof(int));
        writer.write(&vl.hidden_totals1[0], vl.hidden_totals1.size() * sizeof(int));

        writer.write(&vl.importance, sizeof(float));
    }
}

void Encoder::read(
    Stream_Reader &reader
) {
    reader.read(&hidden_size, sizeof(Int3));

    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;

    hidden_cis.resize(num_hidden_columns);

    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));

    learn_cis.resize(num_hidden_columns);

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

        vl.weights0.resize(num_hidden_cells * area * vld.size.z);
        vl.weights1.resize(vl.weights0.size());

        reader.read(&vl.weights0[0], vl.weights0.size() * sizeof(Byte));
        reader.read(&vl.weights1[0], vl.weights1.size() * sizeof(Byte));

        vl.hidden_sums0.resize(num_hidden_cells);
        vl.hidden_sums1.resize(num_hidden_cells);

        vl.hidden_totals.resize(num_hidden_cells);
        vl.hidden_totals1.resize(num_hidden_cells);

        reader.read(&vl.hidden_totals[0], vl.hidden_totals.size() * sizeof(int));
        reader.read(&vl.hidden_totals1[0], vl.hidden_totals1.size() * sizeof(int));

        reader.read(&vl.importance, sizeof(float));
    }
}

void Encoder::write_state(
    Stream_Writer &writer
) const {
    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));
}

void Encoder::read_state(
    Stream_Reader &reader
) {
    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));
}

void Encoder::write_weights(
    Stream_Writer &writer
) const {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        writer.write(&vl.weights0[0], vl.weights0.size() * sizeof(Byte));
        writer.write(&vl.weights1[0], vl.weights1.size() * sizeof(Byte));
    }
}

void Encoder::read_weights(
    Stream_Reader &reader
) {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(&vl.weights0[0], vl.weights0.size() * sizeof(Byte));
        reader.read(&vl.weights1[0], vl.weights1.size() * sizeof(Byte));
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
        
            for (int i = 0; i < vl.weights0.size(); i++) {
                int e = rand() % encoders.size();                

                vl.weights0[i] = encoders[e]->visible_layers[vli].weights0[i];
                vl.weights1[i] = encoders[e]->visible_layers[vli].weights1[i];
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

                for (int e = 0; e < encoders.size(); e++) {
                    total0 += encoders[e]->visible_layers[vli].weights0[i];
                    total1 += encoders[e]->visible_layers[vli].weights1[i];
                }

                vl.weights0[i] = roundf(total0 / encoders.size());
                vl.weights1[i] = roundf(total1 / encoders.size());
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

        for (int hc = 0; hc < hidden_size.z; hc++) {
            int hidden_cell_index = hc + hidden_cells_start;

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
                int sub_total1 = 0;

                for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                    for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                        int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                        Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                        for (int vc = 0; vc < vld.size.z; vc++) {
                            int wi = hc + hidden_size.z * (offset.y + diam * (offset.x + diam * (vc + vld.size.z * hidden_column_index)));

                            sub_total += vl.weights0[wi] + vl.weights1[wi];
                            sub_total1 += vl.weights1[wi];
                        }
                    }

                vl.hidden_totals[hidden_cell_index] = sub_total;
                vl.hidden_totals1[hidden_cell_index] = sub_total1;
            }
        }
    }
}
