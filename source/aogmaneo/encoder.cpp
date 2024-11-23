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
    bool learn_enabled,
    unsigned long* state,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int num_hidden_cells_per_column = hidden_size.z * hidden_size.w;

    int hidden_minis_start = hidden_size.z * hidden_column_index;

    int hidden_cells_start = hidden_column_index * num_hidden_cells_per_column;

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

        for (int hc = 0; hc < num_hidden_cells_per_column; hc++) {
            int hidden_cell_index = hc + hidden_cells_start;

            vl.hidden_sums[hidden_cell_index] = 0;
        }

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int visible_minis_start = vld.size.z * visible_column_index;

                int wi_start_partial = vld.size.z * (offset.y + diam * (offset.x + diam * hidden_column_index));

                for (int vmc = 0; vmc < vld.size.z; vmc++) {
                    int visible_mini_index = vmc + visible_minis_start;

                    int in_ci = vl_input_cis[visible_mini_index];

                    int wi_start = num_hidden_cells_per_column * (in_ci + vld.size.w * (vmc + wi_start_partial));

                    for (int hc = 0; hc < num_hidden_cells_per_column; hc++) {
                        int hidden_cell_index = hc + hidden_cells_start;

                        int wi = hc + wi_start;

                        vl.hidden_sums[hidden_cell_index] += vl.weights[wi];
                    }
                }
            }
    }

    const float byte_inv = 1.0f / 255.0f;

    int mini_max_index = 0;
    float mini_max_activation = limit_min;

    // find max of each minicolumn
    for (int hmc = 0; hmc < hidden_size.z; hmc++) {
        int hidden_mini_index = hmc + hidden_minis_start;

        int hidden_mini_cells_start = hidden_size.w * hidden_mini_index;

        int max_index = 0;
        float max_activation = limit_min;

        for (int hcc = 0; hcc < hidden_size.w; hcc++) {
            int hidden_cell_index = hcc + hidden_mini_cells_start;

            float sum = 0.0f;

            for (int vli = 0; vli < visible_layers.size(); vli++) {
                Visible_Layer &vl = visible_layers[vli];
                const Visible_Layer_Desc &vld = visible_layer_descs[vli];

                float influence = vl.importance * byte_inv;

                sum += vl.hidden_sums[hidden_cell_index] * influence;
            }

            if (sum > max_activation) {
                max_activation = sum;
                max_index = hcc;
            }
        }

        hidden_cis[hidden_mini_index] = max_index;

        if (max_activation > mini_max_activation) {
            mini_max_activation = max_activation;
            mini_max_index = hmc;
        }
    }

    if (learn_enabled) {
        int hidden_mini_index_max = mini_max_index + hidden_minis_start;

        int hidden_cell_index_max = hidden_cis[hidden_mini_index_max] + hidden_size.w * mini_max_index;

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

                    int visible_minis_start = vld.size.z * visible_column_index;

                    Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                    for (int vmc = 0; vmc < vld.size.z; vmc++) {
                        int visible_mini_index = vmc + visible_minis_start;

                        int in_ci = vl_input_cis[visible_mini_index];

                        for (int vcc = 0; vcc < vld.size.w; vcc++) {
                            int wi = hidden_cell_index_max + num_hidden_cells_per_column * (vcc + vld.size.w * (vmc + vld.size.z * (offset.y + diam * (offset.x + diam * hidden_column_index))));

                            vl.weights[wi] = min(255, max(0, vl.weights[wi] + roundf(params.lr * ((vcc == in_ci) * 255.0f - vl.weights[wi]))));
                        }
                    }
                }
        }
    }
}

void Encoder::init_random(
    const Int4 &hidden_size,
    const Array<Visible_Layer_Desc> &visible_layer_descs
) {
    this->visible_layer_descs = visible_layer_descs;

    this->hidden_size = hidden_size;

    visible_layers.resize(visible_layer_descs.size());

    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells_per_column = hidden_size.z * hidden_size.w;
    int num_hidden_cells = num_hidden_columns * num_hidden_cells_per_column;

    // create layers
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights.resize(num_hidden_cells * area * vld.size.z * vld.size.w);

        for (int i = 0; i < vl.weights.size(); i++)
            vl.weights[i] = (255 - rand() % init_weight_noisei);

        vl.hidden_sums.resize(num_hidden_cells);
    }

    hidden_cis = Int_Buffer(num_hidden_columns * hidden_size.z, 0);
}

void Encoder::step(
    const Array<Int_Buffer_View> &input_cis,
    bool learn_enabled,
    const Params &params
) {
    int num_hidden_columns = hidden_size.x * hidden_size.y;

    unsigned int base_state = rand();
    
    PARALLEL_FOR
    for (int i = 0; i < num_hidden_columns; i++) {
        unsigned long state = rand_get_state(base_state + i * rand_subseed_offset);

        forward(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, learn_enabled, &state, params);
    }
}

void Encoder::clear_state() {
    hidden_cis.fill(0);
}

long Encoder::size() const {
    long size = sizeof(Int4) + hidden_cis.size() * sizeof(int) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += sizeof(Visible_Layer_Desc) + vl.weights.size() * sizeof(Byte) + sizeof(float);
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

        size += vl.weights.size() * sizeof(Byte);
    }

    return size;
}

void Encoder::write(
    Stream_Writer &writer
) const {
    writer.write(&hidden_size, sizeof(Int4));

    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));

    int num_visible_layers = visible_layers.size();

    writer.write(&num_visible_layers, sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(&vld, sizeof(Visible_Layer_Desc));

        writer.write(&vl.weights[0], vl.weights.size() * sizeof(Byte));

        writer.write(&vl.importance, sizeof(float));
    }
}

void Encoder::read(
    Stream_Reader &reader
) {
    reader.read(&hidden_size, sizeof(Int4));

    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells_per_column = hidden_size.z * hidden_size.w;
    int num_hidden_cells = num_hidden_columns * num_hidden_cells_per_column;

    hidden_cis.resize(num_hidden_columns * hidden_size.z);

    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));

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

        vl.weights.resize(num_hidden_cells * area * vld.size.z * vld.size.w);

        reader.read(&vl.weights[0], vl.weights.size() * sizeof(Byte));

        vl.hidden_sums.resize(num_hidden_cells);

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

        writer.write(&vl.weights[0], vl.weights.size() * sizeof(Byte));
    }
}

void Encoder::read_weights(
    Stream_Reader &reader
) {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(&vl.weights[0], vl.weights.size() * sizeof(Byte));
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
