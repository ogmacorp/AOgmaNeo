// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
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

                vl.dendrite_sums[dendrite_index] = 0;
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

                        vl.dendrite_sums[dendrite_index] += vl.weights[wi];
                    }
                }
            }
    }

    int max_compare_index = 0;
    float max_compare_activation = limit_min;

    const float byte_inv = 1.0f / 255.0f;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

        int max_index = 0;
        float max_activation = limit_min;

        float compare_activation = limit_min;

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + dendrites_start;

            float sum = 0.0f;
            float total = 0.0f;

            for (int vli = 0; vli < visible_layers.size(); vli++) {
                Visible_Layer &vl = visible_layers[vli];

                float influence = vl.importance * byte_inv;

                sum += vl.dendrite_sums[dendrite_index] * influence;
                total += vl.dendrite_totals[dendrite_index] * influence;
            }

            sum /= max(limit_small, total_importance);
            total /= max(limit_small, total_importance);

            float activation = 2.0f * sum - total;

            if (activation > max_activation) {
                max_activation = activation;
                compare_activation = sum;
                max_index = di;
            }
        }

        hidden_dis[hidden_cell_index] = max_index;

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
    float strength,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    int target_ci = hidden_target_cis[hidden_column_index];
    int hidden_ci = hidden_cis[hidden_column_index];

    int hidden_cell_index_target = target_ci + hidden_cells_start;
    int hidden_cell_index_max = hidden_ci + hidden_cells_start;

    int hidden_di_target = hidden_dis[hidden_cell_index_target];
    int hidden_di_max = hidden_dis[hidden_cell_index_max];

    if (hidden_di_target == -1)
        return;

    int dendrite_index_target = hidden_di_target + num_dendrites_per_cell * hidden_cell_index_target;
    int dendrite_index_max = hidden_di_max + num_dendrites_per_cell * hidden_cell_index_max;

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
                    int wi = hidden_di_target + num_dendrites_per_cell * (target_ci + hidden_size.z * (offset.y + diam * (offset.x + diam * (in_ci + vld.size.z * hidden_column_index))));

                    Byte w_old = vl.weights[wi];

                    vl.weights[wi] = min(255, vl.weights[wi] + static_cast<int>(params.lr * max(0.0f, strength * 255.0f - vl.weights[wi])));

                    vl.dendrite_totals[dendrite_index_target] += vl.weights[wi] - w_old;
                }

                if (hidden_ci != target_ci) {
                    int wi = hidden_di_max + num_dendrites_per_cell * (hidden_ci + hidden_size.z * (offset.y + diam * (offset.x + diam * (in_ci + vld.size.z * hidden_column_index))));

                    Byte w_old = vl.weights[wi];

                    vl.weights[wi] = max(0, vl.weights[wi] - static_cast<int>(params.fr * vl.weights[wi]));

                    vl.dendrite_totals[dendrite_index_max] += vl.weights[wi] - w_old;
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

        vl.weights.resize(num_dendrites * area * vld.size.z);

        for (int i = 0; i < vl.weights.size(); i++)
            vl.weights[i] = (rand() % init_weight_noisei);

        vl.dendrite_sums.resize(num_dendrites);
        vl.dendrite_totals.resize(num_dendrites);
    }

    hidden_cis = Int_Buffer(num_hidden_columns, 0);

    hidden_dis = Int_Buffer(num_hidden_cells, -1);

    // init totals
    for (int i = 0; i < num_hidden_columns; i++) {
        Int2 column_pos(i / hidden_size.y, i % hidden_size.y);

        int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

        int hidden_cells_start = hidden_column_index * hidden_size.z;

        for (int hc = 0; hc < hidden_size.z; hc++) {
            int hidden_cell_index = hc + hidden_cells_start;

            int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

            for (int di = 0; di < num_dendrites_per_cell; di++) {
                int dendrite_index = di + dendrites_start;

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

                    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                            int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                            Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                            for (int vc = 0; vc < vld.size.z; vc++) {
                                int wi = di + num_dendrites_per_cell * (hc + hidden_size.z * (offset.y + diam * (offset.x + diam * (vc + vld.size.z * hidden_column_index))));

                                sub_total += vl.weights[wi];
                            }
                        }

                    vl.dendrite_totals[dendrite_index] = sub_total;
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
    float strength,
    const Params &params
) {
    int num_hidden_columns = hidden_size.x * hidden_size.y;

    PARALLEL_FOR
    for (int i = 0; i < num_hidden_columns; i++)
        learn(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, hidden_target_cis, strength, params);
}

void Decoder::clear_state() {
    hidden_cis.fill(0);
    hidden_dis.fill(-1);
}

long Decoder::size() const {
    long size = sizeof(Int3) + sizeof(int) + hidden_cis.size() * sizeof(int) + hidden_dis.size() * sizeof(int) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        size += sizeof(Visible_Layer_Desc) + vl.weights.size() * sizeof(Byte) + vl.dendrite_totals.size() * sizeof(int) + sizeof(float);
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

        size += vl.weights.size() * sizeof(Byte);
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
    
    int num_visible_layers = visible_layers.size();

    writer.write(&num_visible_layers, sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(&vld, sizeof(Visible_Layer_Desc));

        writer.write(&vl.weights[0], vl.weights.size() * sizeof(Byte));

        writer.write(&vl.dendrite_totals[0], vl.dendrite_totals.size() * sizeof(int));

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

    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));
    reader.read(&hidden_dis[0], hidden_dis.size() * sizeof(int));

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

        vl.weights.resize(num_dendrites * area * vld.size.z);

        reader.read(&vl.weights[0], vl.weights.size() * sizeof(Byte));

        vl.dendrite_sums.resize(num_dendrites);

        vl.dendrite_totals.resize(num_dendrites);

        reader.read(&vl.dendrite_totals[0], vl.dendrite_totals.size() * sizeof(int));

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

        writer.write(&vl.weights[0], vl.weights.size() * sizeof(Byte));
    }
}

void Decoder::read_weights(
    Stream_Reader &reader
) {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(&vl.weights[0], vl.weights.size() * sizeof(Byte));
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
        
            for (int i = 0; i < vl.weights.size(); i++) {
                int d = rand() % decoders.size();                

                vl.weights[i] = decoders[d]->visible_layers[vli].weights[i];
            }
        }

        break;
    case merge_average:
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];
        
            for (int i = 0; i < vl.weights.size(); i++) {
                float total = 0.0f;

                for (int d = 0; d < decoders.size(); d++)
                    total += decoders[d]->visible_layers[vli].weights[i];

                vl.weights[i] = roundf(total / decoders.size());
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

            int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

            for (int di = 0; di < num_dendrites_per_cell; di++) {
                int dendrite_index = di + dendrites_start;

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

                    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                            int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                            Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                            for (int vc = 0; vc < vld.size.z; vc++) {
                                int wi = di + num_dendrites_per_cell * (hc + hidden_size.z * (offset.y + diam * (offset.x + diam * (vc + vld.size.z * hidden_column_index))));

                                sub_total += vl.weights[wi];
                            }
                        }

                    vl.dendrite_totals[dendrite_index] = sub_total;
                }
            }
        }
    }
}
