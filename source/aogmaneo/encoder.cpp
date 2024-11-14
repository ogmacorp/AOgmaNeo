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
    const Array<Int_Buffer_View> &input_cis,
    Float_Buffer_View errors,
    bool learn_enabled,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    float error = errors[hidden_column_index];

    if (learn_enabled) {
        int hidden_ci_prev = hidden_cis[hidden_column_index];

        int hidden_cell_index_prev = hidden_ci_prev + hidden_cells_start;

        int delta = roundf(255.0f * tanhf(params.lr * error));

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

                    int in_ci_prev = vl.input_cis_prev[visible_column_index];

                    Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                    int wi = hidden_ci_prev + hidden_size.z * (offset.y + diam * (offset.x + diam * (in_ci_prev + vld.size.z * hidden_column_index)));

                    vl.weights[wi] = min(255, max(0, vl.weights[wi] + delta));
                }
        }
    }

    float count = 0.0f;
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

        total_importance += vl.importance;

        Int_Buffer_View vl_input_cis = input_cis[vli];

        for (int hc = 0; hc < hidden_size.z; hc++) {
            int hidden_cell_index = hc + hidden_cells_start;

            vl.hidden_sums[hidden_cell_index] = 0;
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

                    vl.hidden_sums[hidden_cell_index] += vl.weights[wi];
                }
            }
    }

    count /= max(limit_small, total_importance);

    int max_index = 0;
    float max_activation = limit_min;

    const float byte_inv = 1.0f / 255.0f;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        float sum = 0.0f;

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];

            float influence = vl.importance * byte_inv;

            sum += vl.hidden_sums[hidden_cell_index] * influence;
        }

        sum /= max(limit_small, total_importance);

        if (sum > max_activation) {
            max_activation = sum;
            max_index = hc;
        }
    }

    hidden_cis[hidden_column_index] = max_index;
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

        vl.weights.resize(num_hidden_cells * area * vld.size.z);

        for (int i = 0; i < vl.weights.size(); i++)
            vl.weights[i] = 255 - (rand() % init_weight_noisei);

        vl.hidden_sums.resize(num_hidden_cells);

        vl.input_cis_prev = Int_Buffer(num_visible_columns, 0);
    }

    hidden_cis = Int_Buffer(num_hidden_columns, 0);
}

void Encoder::step(
    const Array<Int_Buffer_View> &input_cis,
    Float_Buffer_View errors,
    bool learn_enabled,
    const Params &params
) {
    int num_hidden_columns = hidden_size.x * hidden_size.y;

    PARALLEL_FOR
    for (int i = 0; i < num_hidden_columns; i++)
        forward(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, errors, learn_enabled, params);

    // copy to prevs
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        vl.input_cis_prev = input_cis[vli];
    }
}

void Encoder::clear_state() {
    hidden_cis.fill(0);

    for (int vli = 0; vli < visible_layers.size(); vli++)
        visible_layers[vli].input_cis_prev.fill(0);
}

long Encoder::size() const {
    long size = sizeof(Int3) + hidden_cis.size() * sizeof(int) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += sizeof(Visible_Layer_Desc) + vl.weights.size() * sizeof(Byte) + vl.input_cis_prev.size() * sizeof(int) + sizeof(float);
    }

    return size;
}

long Encoder::state_size() const {
    long size = hidden_cis.size() * sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += vl.input_cis_prev.size() * sizeof(int);
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
    writer.write(&hidden_size, sizeof(Int3));

    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));

    int num_visible_layers = visible_layers.size();

    writer.write(&num_visible_layers, sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(&vld, sizeof(Visible_Layer_Desc));

        writer.write(&vl.weights[0], vl.weights.size() * sizeof(Byte));

        writer.write(&vl.input_cis_prev[0], vl.input_cis_prev.size() * sizeof(int));

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

        vl.weights.resize(num_hidden_cells * area * vld.size.z);

        reader.read(&vl.weights[0], vl.weights.size() * sizeof(Byte));

        vl.hidden_sums.resize(num_hidden_cells);

        vl.input_cis_prev.resize(num_visible_columns);

        reader.read(&vl.input_cis_prev[0], vl.input_cis_prev.size() * sizeof(int));

        reader.read(&vl.importance, sizeof(float));
    }
}

void Encoder::write_state(
    Stream_Writer &writer
) const {
    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        writer.write(&vl.input_cis_prev[0], vl.input_cis_prev.size() * sizeof(int));
    }
}

void Encoder::read_state(
    Stream_Reader &reader
) {
    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(&vl.input_cis_prev[0], vl.input_cis_prev.size() * sizeof(int));
    }
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
